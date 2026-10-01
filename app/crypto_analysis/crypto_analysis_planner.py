"""识别目标币种与研究重点，工具选择交给各维度的 Agent。"""

import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_contract import _HISTORICAL
from app.crypto_analysis.crypto_dimension_catalog import DimensionCatalog
from app.tracing import estimate_tokens, trace_generation


_ASSET_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")


def create_request_scope(question: str, thread_id: str, trace_id: str) -> dict[str, str]:
    """在请求入口固定 UTC 截止时间和追踪标识。"""
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "trace_id": trace_id,
    }


class AnalysisPlan(BaseModel):
    """一轮分析的目标与重点，不提前指定取数或计算工具。"""

    model_config = ConfigDict(extra="forbid")
    target_asset: str
    analysis_focus: str = Field(min_length=1)


def validate_plan(plan: AnalysisPlan) -> dict[str, str]:
    """保持单币种约束，并要求规划明确用户的研究重点。"""
    if not _ASSET_PATTERN.fullmatch(plan.target_asset):
        raise ValueError("计划中的目标币种格式无效")
    if not plan.analysis_focus.strip():
        raise ValueError("分析计划缺少研究重点")
    return plan.model_dump()


async def create_analysis_plan(llm: Any, question: str, catalog: DimensionCatalog) -> dict[str, str]:
    """让模型识别币种与研究重点，后续由维度 Agent 实时选择工具。"""
    if _HISTORICAL.search(question):
        raise ValueError("当前仅支持当前时点分析，暂不支持历史截止时间")
    overview = [
        {
            "dimension_id": dimension_id,
            "name": catalog.dimensions[dimension_id].name,
            "metrics": [metric.label for metric in catalog.dimensions[dimension_id].metrics],
        }
        for dimension_id in catalog.enabled_dimensions
    ]
    prompt = (
        "你是加密货币分析规划员。识别用户问题中的单个目标币种，未指定时取 BTC；"
        "多个币种或无法识别时不要猜测。返回 target_asset 和 analysis_focus。"
        "analysis_focus 概括用户真正想研究的问题、关注点及需要说明的判断边界，保留原意。"
        "本阶段不选择工具或计算器；各维度 Agent 会根据问题和工具结果自行决策。"
        "不要生成或修改数据截止时间，不宣称尚未取得的数据或来源。"
        "仅返回符合下列 JSON Schema 的 JSON 对象，不输出 Markdown 或额外说明。"
        f"\nJSON Schema：{json.dumps(AnalysisPlan.model_json_schema(), ensure_ascii=False)}"
        f"\n本轮分析范围：{json.dumps(overview, ensure_ascii=False)}\n用户问题：{question}"
    )
    # 规划只提取字段，使用 JSON 模式，避免把普通文本输出误当作工具调用结果。
    structured_llm = llm.with_structured_output(
        AnalysisPlan, method="json_mode"
    )
    with trace_generation("crypto_analysis_plan", model=settings.fast_model) as generation:
        plan = await structured_llm.ainvoke(prompt)
        if isinstance(plan, AnalysisPlan):
            generation.usage = {
                "input": estimate_tokens(prompt),
                "output": estimate_tokens(plan.model_dump_json()),
            }
    if not isinstance(plan, AnalysisPlan):
        raise ValueError("规划模型未返回 AnalysisPlan 结构化结果")
    return validate_plan(plan)
