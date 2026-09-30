"""让 LLM 识别目标币种并在配置允许范围内选择工具与计算器。"""

import json  # 构造模型提示词中的配置清单。
import re  # 校验模型输出的币种标识。
from datetime import datetime, timezone  # 为整轮请求生成一次 UTC 截止时间。
from typing import Any  # 接收项目现有的聊天模型对象。

from pydantic import BaseModel, ConfigDict, Field  # 解析模型返回的严格 JSON。

from app.crypto_analysis.crypto_analysis_contract import _HISTORICAL  # 沿用原入口对历史问题的限制。
from app.crypto_analysis.crypto_dimension_catalog import DimensionCatalog  # 使用本轮配置快照。
from app.crypto_analysis.crypto_metric_runtime import CALCULATOR_REGISTRY, TOOL_REGISTRY  # 核对实际注册的能力。


_ASSET_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")  # 接受规范化的单个币种符号。


def create_request_scope(question: str, thread_id: str, trace_id: str) -> dict[str, str]:  # 在请求入口固定共享范围。
    return {  # 维度路径随后由 LLM 补入目标币种。
        "as_of": datetime.now(timezone.utc).isoformat(),  # 整轮共用的 UTC 截止时点。
        "thread_id": thread_id,  # 跨请求会话标识。
        "trace_id": trace_id,  # 当前请求追踪标识。
    }  # 返回不可由 LLM 改写的初始范围。


class MetricSelection(BaseModel):  # LLM 对一项必算指标作出的选择。
    model_config = ConfigDict(extra="forbid")  # 禁止任务自行覆盖标的或截止时间。
    dimension_id: str  # 所属维度编号。
    metric_id: str  # 配置中已有的指标标识。
    tool_id: str | None  # 已接入指标选择工具；待接入指标填写空值。
    calculator_id: str | None  # 已接入指标选择计算器；待接入指标填写空值。


class AnalysisPlan(BaseModel):  # 一轮维度分析的模型计划。
    model_config = ConfigDict(extra="forbid")  # 不接受计划外字段。
    target_asset: str  # 单个规范化目标币种。
    metrics: list[MetricSelection] = Field(min_length=1)  # 全部必算指标的选择。


def _planning_catalog(catalog: DimensionCatalog) -> list[dict[str, Any]]:  # 生成模型可见的最小配置。
    overview = []  # 按维度顺序组织指标。
    for dimension_id in catalog.enabled_dimensions:  # 每个启用维度都参与计划。
        dimension = catalog.dimensions[dimension_id]  # 取得本维配置。
        overview.append({  # 只提供工具决策所需字段。
            "dimension_id": dimension_id,  # 维度编号。
            "name": dimension.name,  # 维度名称。
            "metrics": [  # 本维全部必算指标。
                {  # 一项指标的可用能力。
                    "metric_id": metric.id,  # 指标标识。
                    "kind": metric.kind,  # 数值或文本类型。
                    "status": metric.status,  # 当前是否已接入。
                    "allowed_tools": metric.allowed_tools,  # 可选工具集合。
                    "allowed_calculators": metric.allowed_calculators,  # 可选计算器集合。
                } for metric in dimension.metrics  # 遍历不同数量的指标。
            ],  # 结束本维指标列表。
        })  # 结束本维概览。
    return overview  # 返回维度计划输入。


def validate_plan(plan: AnalysisPlan, catalog: DimensionCatalog) -> dict[str, Any]:  # 只做已确认的三项计划校验。
    if not _ASSET_PATTERN.fullmatch(plan.target_asset):  # 第一层固定单个规范化标的。
        raise ValueError("计划中的目标币种格式无效")  # 拒绝含交易对或多个币种的值。
    expected = {  # 配置决定完整的必算指标集合。
        (dimension_id, metric.id)  # 用维度和指标的组合避免重名。
        for dimension_id in catalog.enabled_dimensions  # 遍历启用维度。
        for metric in catalog.dimensions[dimension_id].metrics  # 遍历各维指标。
    }  # 完整指标集合。
    actual = [(item.dimension_id, item.metric_id) for item in plan.metrics]  # 读取 LLM 的计划清单。
    if len(actual) != len(set(actual)) or set(actual) != expected:  # 检查遗漏、重复和多余指标。
        raise ValueError("计划未完整覆盖配置中的维度必算指标")  # 不自行补齐计划。
    for item in plan.metrics:  # 检查每项选择是否在配置允许列表内。
        metric = next(metric for metric in catalog.dimensions[item.dimension_id].metrics if metric.id == item.metric_id)  # 读取对应配置。
        if metric.status == "pending":  # 未接入指标仍须列入维度计划。
            if item.tool_id is not None or item.calculator_id is not None:  # 待接入项不允许伪选工具。
                raise ValueError(f"{item.dimension_id}.{item.metric_id} 尚未接入工具")  # 保留缺失边界。
        elif item.tool_id not in metric.allowed_tools or item.calculator_id not in metric.allowed_calculators:  # 已接入项只选允许能力。
            raise ValueError(f"{item.dimension_id}.{item.metric_id} 选择了未允许的工具或计算器")  # 拒绝越界计划。
    return {  # 所有任务共同使用这一份请求范围。
        "target_asset": plan.target_asset,  # LLM 识别出的单个标的。
        "metrics": [item.model_dump() for item in plan.metrics],  # 通过校验的指标计划。
    }  # 截止时间不由 LLM 生成，执行时从请求范围注入。


async def create_analysis_plan(llm: Any, question: str, catalog: DimensionCatalog) -> dict[str, Any]:  # 调用 LLM 形成计划。
    if _HISTORICAL.search(question):  # 只在维度分析路径保留历史问题限制。
        raise ValueError("当前仅支持当前时点分析，暂不支持历史截止时间")  # 普通主题研究不受此限制。
    for dimension_id in catalog.enabled_dimensions:  # 先核对机器配置引用的能力实际存在。
        for metric in catalog.dimensions[dimension_id].metrics:  # 逐项核对已接入能力。
            if metric.status == "ready" and (set(metric.allowed_tools) - set(TOOL_REGISTRY) or set(metric.allowed_calculators) - set(CALCULATOR_REGISTRY)):  # 注册表必须覆盖配置。
                raise ValueError(f"config_error：{dimension_id}.{metric.id} 引用了未注册的能力")  # 不把配置错误交给模型处理。
    overview = _planning_catalog(catalog)  # 提取模型可见的工具候选。
    prompt = (  # 要求模型通过结构化输出生成受限计划。
        "你是加密货币分析规划员。识别问题中的单个目标币种，未指定时取 BTC；多个币种或无法识别时不要猜测。"
        "必须为下列全部维度和必算指标各输出一次选择。ready 指标从 allowed_tools 和 allowed_calculators 各选一个；"
        "pending 指标的 tool_id 与 calculator_id 均写 null。通过结构化输出返回 target_asset 和 metrics；"
        "metrics 每项只含 dimension_id、metric_id、tool_id、calculator_id。不要生成 as_of，执行器会注入统一截止时间。"
        f"\n配置：{json.dumps(overview, ensure_ascii=False)}\n用户问题：{question}"  # 提供完整配置与原问题。
    )  # 完成提示词。
    structured_llm = llm.with_structured_output(AnalysisPlan, method="function_calling", tool_choice="required")  # 智谱仅支持 auto，覆盖 ChatOpenAI 默认的具名工具选择。
    plan = await structured_llm.ainvoke(prompt)  # 调用结构化模型，不解析自由文本。
    if not isinstance(plan, AnalysisPlan):  # 模型未调用结构化工具时不猜测内容。
        raise ValueError("规划模型未返回 AnalysisPlan 结构化结果")  # 阻止空计划进入执行图。
    return validate_plan(plan, catalog)  # 执行三项已确认的计划校验。
