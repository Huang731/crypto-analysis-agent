"""单次路由到研究专家，完成后统一整理报告。"""

import logging
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import TypedDict

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_agent import build_dynamic_analysis_agent
from app.crypto_analysis.crypto_analysis_state import HistorySummaryOutput
from app.crypto_analysis.skill_memory import merge_skill_memory
from app.observability import get_logger, timed_node
from app.orchestration.query_rewriter import rewrite_question
from app.rag.retriever import KBRetriever
from app.topic_research.topic_research_agent import build_topic_research_agent
from app.tracing import trace_generation_call, trace_span
from eval_agent.routing.route_decision import (
    RouteDecision,
    RouteTarget,
    append_routing_candidate,
    decide_route,
)
from prompts.crypto_analysis import history_summary_prompt

_LOG = get_logger("crypto_analysis.routing")


@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    before_sleep=before_sleep_log(_LOG, logging.WARNING),
    reraise=True,
)
@trace_generation_call(name="crypto_history_summary", model=settings.smart_model)
async def _invoke_history_summary(llm: BaseChatModel, prompt: str) -> str:
    """直接生成摘要正文，沿用非空、长度校验和有限重试。"""
    response = await llm.ainvoke(prompt, config={"tags": ["crypto_history_summary"]})
    output = HistorySummaryOutput(history_summary=response.content)
    return output.history_summary


class RoutingState(TypedDict, total=False):
    """外层只保留路由和报告，研究明细由专家子图持有。"""

    # 本轮请求范围：分析截止时间、会话和追踪标识、原始及补全问题等。
    scope: dict[str, str]
    # 本轮原话规范改写后的问题，用于路由和专家分析。
    resolved_question: str
    # 同会话各技能的指标表格与思想体系解读，按原共享范围和技能名保存。
    skill_memory: Annotated[dict[str, str], merge_skill_memory]
    # 父图更新加密分析的累计结论摘要，其余分支保持已有内容。
    history_summary: str
    # 承接加密分析计划，沿用无适用技能时不更新摘要的条件。
    plan: dict[str, Any]
    # 本轮路由目标：加密分析、主题研究或直接回复。
    route: RouteTarget
    # 直接回复分支的正文；进入专家分支时为空字符串。
    direct_reply: str
    # 本轮最终报告，由专家生成或由直接回复正文写入。
    report: str


def select_agent(state: RoutingState) -> RouteTarget:
    """根据已校验的路由结果选择唯一分支。"""
    return state["route"]


def finalize_report(state: RoutingState) -> dict[str, Any]:
    """统一承接已有报告，不重新调用模型或改写专家结论。"""
    report = state["direct_reply"] if state["route"] == "direct" else state["report"]
    if not isinstance(report, str) or not report.strip():
        raise ValueError("最终报告为空")
    return {"report": report}


def build_routing_graph(
    smart: BaseChatModel,
    fast: BaseChatModel,
    retriever: KBRetriever,
    *,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """保留现有构建入口，以显式图保证每轮只执行一个专家。"""
    crypto_agent = build_dynamic_analysis_agent(smart, fast, retriever)
    topic_agent = build_topic_research_agent(smart, fast)

    @timed_node
    async def rewrite(state: RoutingState) -> dict[str, Any]:
        """结合父图历史摘要补全本轮问题，保留用户原话。"""
        original = state["scope"]["original_question"]
        with trace_span("question_rewriter", question_chars=len(original)) as span:
            question = await rewrite_question(
                fast, original, state.get("history_summary", "")
            )
            span.metadata["rewritten_chars"] = len(question)
        return {
            "resolved_question": question,
            "scope": {**state["scope"], "resolved_question": question},
        }

    @timed_node
    async def router(state: RoutingState) -> dict[str, Any]:
        """路由调用、解析和类型校验整体最多尝试两次，耗尽后向上报告。"""
        decision: RouteDecision = await decide_route(smart, state["resolved_question"])
        if settings.routing_capture_enabled:
            append_routing_candidate(
                original_question=state["scope"]["original_question"],
                resolved_question=state["resolved_question"],
                decision=decision,
                path=settings.routing_capture_path,
            )
        return decision.model_dump()

    @timed_node
    async def summarize_history(state: RoutingState) -> dict[str, str]:
        """在父图报告整理完成后，沿用加密分析的历史摘要更新规则。"""
        if (
            state["route"] != "crypto_analysis_agent"
            or not state["plan"]["skill_names"]
        ):
            return {}
        prompt = history_summary_prompt(
            previous_summary=state.get("history_summary", ""),
            report=state["report"],
        )
        summary = await _invoke_history_summary(smart, prompt)
        return {"history_summary": summary}

    workflow = StateGraph(RoutingState)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("router", router)
    workflow.add_node("crypto_analysis_agent", crypto_agent)
    workflow.add_node("topic_research_agent", topic_agent)
    workflow.add_node("finalize", finalize_report)
    workflow.add_node("summarize_history", summarize_history)
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "router")
    workflow.add_conditional_edges(
        "router",
        select_agent,
        {
            "crypto_analysis_agent": "crypto_analysis_agent",
            "topic_research_agent": "topic_research_agent",
            "direct": "finalize",
        },
    )
    workflow.add_edge("crypto_analysis_agent", "finalize")
    workflow.add_edge("topic_research_agent", "finalize")
    workflow.add_edge("finalize", "summarize_history")
    workflow.add_edge("summarize_history", END)
    return workflow.compile(checkpointer=checkpointer, name="research_routing")
