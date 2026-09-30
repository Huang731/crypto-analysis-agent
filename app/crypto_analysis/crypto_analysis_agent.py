"""维度通用分析图：LLM 规划、按维执行、汇总报告。"""

import json  # 将结构化事实交给报告模型并记录发现。
import operator  # 合并并行维度节点返回的列表。
from typing import Annotated, Any  # 声明图状态与模型类型。
from typing_extensions import TypedDict  # 定义 LangGraph 共享状态。

from langchain_core.messages import AIMessage, HumanMessage  # 读取用户问题并追加最终报告。
from langgraph.graph import END, START, StateGraph  # 构建父图。
from langgraph.graph.message import add_messages  # 合并消息列表。
from langgraph.types import Send  # 动态分派不同维度。

from app.crypto_analysis.crypto_analysis_planner import create_analysis_plan  # 让 LLM 生成受限计划。
from app.crypto_analysis.crypto_metric_runtime import execute_dimension  # 执行维度内指标与聚合。
from app.crypto_analysis.crypto_dimension_catalog import DimensionCatalog, load_dimension_catalog  # 加载维度配置。
from app.observability import get_logger, log_event, timed_node  # 保留项目现有观测方式。
from app.config.config import settings  # 读取报告模型配置。
from app.tracing import estimate_tokens, trace_generation  # 记录模型调用与估算用量。


_LOG = get_logger("crypto_analysis.dynamic_analysis")  # 当前图使用的结构化日志器。


class DynamicAnalysisState(TypedDict, total=False):  # 维度分析的共享状态。
    messages: Annotated[list[Any], add_messages]  # 用户问题与最终报告。
    scope: dict[str, str]  # 整轮共用的标的、截止时间和追踪标识。
    catalog: DimensionCatalog  # 本轮读取的机器配置快照。
    plan: dict[str, Any]  # LLM 输出且已校验的计划。
    dimension_results: Annotated[list[dict[str, Any]], operator.add]  # 并行维度结果。
    findings: Annotated[list[str], operator.add]  # 前端可展开的指标事实。
    report: str  # 最终用户报告。


def build_dynamic_analysis_agent(smart_llm: Any, fast_llm: Any):  # 构建可复用的维度父图。
    @timed_node  # 记录 LLM 计划阶段的耗时和失败。
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:  # 读取配置并识别币种。
        catalog = load_dimension_catalog()  # 每轮只读取一次维度配置。
        question = next((message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)), None)  # 跳过 Supervisor 的工具交接消息。
        if not isinstance(question, str) or not question.strip():  # 规划只接受非空用户问题。
            raise ValueError("维度分析缺少有效用户问题")  # 不把工具交接信息当成币种问题。
        selection = await create_analysis_plan(fast_llm, question, catalog)  # LLM 选择工具和计算器。
        scope = {**state["scope"], "target_asset": selection["target_asset"]}  # 把目标币种锁定到共享范围。
        log_event(_LOG, "analysis.plan.ready", target_asset=selection["target_asset"], metrics=len(selection["metrics"]))  # 记录计划摘要。
        return {"catalog": catalog, "plan": selection, "scope": scope}  # 供维度节点统一使用。

    def send_dimensions(state: DynamicAnalysisState) -> list[Send]:  # 按配置动态分派六个维度。
        catalog = state["catalog"]  # 取本轮配置快照。
        plan_metrics = state["plan"]["metrics"]  # 取校验通过的全部指标计划。
        return [  # 每个启用维度对应一个独立任务。
            Send(  # 向同一个通用维度节点发送不同输入。
                "dimension_worker",  # 通用维度执行节点。
                {  # 节点只接收所属维度的任务。
                    "dimension": catalog.dimensions[dimension_id],  # 本维配置。
                    "selections": [item for item in plan_metrics if item["dimension_id"] == dimension_id],  # 本维计划。
                    "scope": state["scope"],  # 同一标的和截止时间。
                },  # 结束节点输入。
            ) for dimension_id in catalog.enabled_dimensions  # 保持配置规定的维度范围。
        ]  # 返回全部维度任务。

    @timed_node  # 记录维度执行耗时与异常。
    async def dimension_worker(state: dict[str, Any]) -> dict[str, Any]:  # 执行某个维度的可变数量指标。
        dimension = state["dimension"]  # 本节点负责的维度配置。
        result = await execute_dimension(dimension, state["selections"], state["scope"])  # 取数、计算并聚合。
        findings = [json.dumps(item, ensure_ascii=False) for item in result["indicators"]]  # 保留每项原始事实。
        from langgraph.config import get_stream_writer  # 按项目现有方式发送进度。
        get_stream_writer()({"event": "dimension_completed", "dimension_id": dimension.dimension_id})  # 通知页面完成的维度。
        log_event(_LOG, "analysis.dimension.done", dimension_id=dimension.dimension_id, status=result["status"])  # 记录维度状态。
        return {"dimension_results": [result], "findings": findings}  # 并行结果通过 reducer 合并。

    @timed_node  # 记录报告生成耗时与异常。
    async def writer(state: DynamicAnalysisState) -> dict[str, Any]:  # 基于结构化结果生成报告。
        catalog = state["catalog"]  # 读取维度顺序。
        by_id = {item["dimension_id"]: item for item in state["dimension_results"]}  # 整理并行结果。
        ordered = [by_id[dimension_id] for dimension_id in catalog.enabled_dimensions]  # 固定报告维度顺序。
        facts = json.dumps(ordered, ensure_ascii=False)  # 保留程序计算的事实和缺失范围。
        prompt = (  # 限制模型只能解释已提供的结果。
            f"你是加密货币研究报告撰写者。目标币种：{state['scope']['target_asset']}。"
            "依据下列维度结构化结果写中文报告。先说明只有哪些维度取得正式分；"
            "各待接入维度须写明数据缺失，分数写 N/A；不得编造实时数据、来源、分数或综合交易结论。"
            "有效的 0 分是实际分数，不等同缺失。报告只供研究参考，不构成投资建议。"
            f"\n维度结果：{facts}"  # 提供全部结构化维度事实。
        )  # 完成报告提示词。
        with trace_generation("crypto_analysis_writer", model=settings.smart_model) as generation:  # 记录报告模型调用。
            response = await smart_llm.ainvoke(prompt)  # 调用报告模型。
            if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
        if not isinstance(response.content, str) or not response.content.strip():  # 报告必须是非空文本。
            raise ValueError("报告模型未返回有效文本")  # 不输出空报告。
        report = response.content.strip()  # 去除模型输出首尾空白。
        return {"report": report, "messages": [AIMessage(content=report)]}  # 回写最终报告。

    builder = StateGraph(DynamicAnalysisState)  # 创建通用维度状态图。
    builder.add_node("plan", plan)  # 注册 LLM 计划节点。
    builder.add_node("dimension_worker", dimension_worker)  # 注册通用维度节点。
    builder.add_node("writer", writer)  # 注册报告节点。
    builder.add_edge(START, "plan")  # 从问题进入计划阶段。
    builder.add_conditional_edges("plan", send_dimensions)  # 计划通过后并行分派维度。
    builder.add_edge("dimension_worker", "writer")  # 等待维度结果合并后撰写报告。
    builder.add_edge("writer", END)  # 报告完成后结束。
    return builder.compile(name="crypto_analysis_agent")  # 返回编译后的通用图。
