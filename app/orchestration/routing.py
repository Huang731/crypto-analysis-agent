"""单次路由到研究专家，完成后统一整理报告。"""

from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, TypedDict

from app.candidate_capture import append_node_candidate
from app.config.config import settings
from app.crypto_analysis.crypto_analysis_parent_agent import (
    build_crypto_analysis_parent_agent,
)
from app.crypto_analysis.skill_memory import merge_skill_memory
from app.llm.model import ainvoke_structured
from app.observability import get_logger, timed_node
from app.orchestration.query_rewriter import rewrite_question
from app.rag.retriever import KBRetriever
from app.topic_research.topic_research_agent import build_topic_research_agent
from app.tracing import trace_generation_call, trace_span
from prompts.routing import routing_prompt

_LOG = get_logger("crypto_analysis.routing")
RouteTarget = Literal["crypto_analysis_agent", "topic_research_agent", "direct"]


class RouteDecision(BaseModel):
    """路由只选择一个分支，直接回复时必须提供正文。"""

    model_config = ConfigDict(extra="forbid")

    route: RouteTarget
    direct_reply: str = ""

    @model_validator(mode="after")
    def validate_reply(self) -> Self:
        """拒绝空直接回复，以及专家分支夹带的回复。"""
        if self.route == "direct":
            if not self.direct_reply.strip():
                raise ValueError("直接回复分支必须提供非空正文")
        elif self.direct_reply:
            raise ValueError("专家分支的 direct_reply 必须为空")
        return self


async def decide_route(
    fast: BaseChatModel,
    resolved_question: str,
) -> RouteDecision:
    """只执行路由，沿用模型工厂的超时和整体最多两次尝试。"""
    messages = [
        SystemMessage(content=routing_prompt()),
        HumanMessage(content=resolved_question),
    ]
    return await _invoke_route_decision(fast, messages)


@trace_generation_call(
    name="routing_decision",
    model=settings.bailian_fast_model,
    input_arg="messages",
    input_format="messages",
    output_format="json",
)
async def _invoke_route_decision(
    fast: BaseChatModel,
    messages: list[SystemMessage | HumanMessage],
) -> RouteDecision:
    """保留路由追踪，结构化调用与校验统一交给模型层。"""
    return await ainvoke_structured(fast, messages, schema=RouteDecision)


class RoutingState(TypedDict, total=False):
    """外层只保留路由和报告，研究明细由专家子图持有。"""

    # 本轮请求范围：分析截止时间、会话和追踪标识、原始及补全问题等。
    scope: dict[str, str]
    # 本轮原话规范改写后的问题，用于路由和专家分析。
    resolved_question: str
    # 最近一轮已完成回答的问题，保留补全后的主题供连续追问使用。
    previous_question: str
    # 同会话各技能的表格与思想原文，以 JSON 字符串按原共享范围和技能名保存。
    skill_memory: Annotated[dict[str, str], merge_skill_memory]
    # 承接加密分析子图的累计结论摘要，仅用于分析并跨轮保存。
    history_summary: str
    # 承接加密分析子图的计划。
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
    """统一承接已有报告，校验成功后保存本轮问题供下轮改写。"""
    report = state["direct_reply"] if state["route"] == "direct" else state["report"]
    if not isinstance(report, str) or not report.strip():
        raise ValueError("最终报告为空")
    update = {"report": report, "previous_question": state["resolved_question"]}
    if settings.finalize_capture_enabled and state["route"] != "topic_research_agent":
        append_node_candidate(
            node="finalize",
            scope=state["scope"],
            input={
                "route": state["route"],
                "resolved_question": state["resolved_question"],
                "report": report,
            },
            actual_output=update,
            path=settings.finalize_capture_path,
        )
    return update


def build_routing_graph(
    bailian_smart_llm: BaseChatModel,
    bailian_fast_llm: BaseChatModel,
    retriever: KBRetriever,
    *,
    checkpointer: BaseCheckpointSaver,
    glm_smart_llm: BaseChatModel,
    glm_4_flash_llm: BaseChatModel,
) -> CompiledStateGraph:
    """保留现有构建入口，以显式图保证每轮只执行一个专家。"""
    crypto_agent = build_crypto_analysis_parent_agent(
        glm_smart_llm=glm_smart_llm,
        bailian_fast_llm=bailian_fast_llm,
        retriever=retriever,
        glm_4_flash_llm=glm_4_flash_llm,
    )
    topic_agent = build_topic_research_agent(glm_smart_llm, glm_4_flash_llm)

    @timed_node
    async def rewrite(state: RoutingState) -> dict[str, Any]:
        """结合上一轮问题补全本轮问题，保留用户原话。"""
        original = state["scope"]["original_question"]
        with trace_span("question_rewriter", question_chars=len(original)) as span:
            question = await rewrite_question(
                glm_4_flash_llm, original, state.get("previous_question", "")
            )
            span.metadata["rewritten_chars"] = len(question)
        if settings.rewrite_capture_enabled:
            append_node_candidate(
                node="rewrite",
                scope=state["scope"],
                input={
                    "question": original,
                    "previous_question": state.get("previous_question", ""),
                },
                actual_output={"resolved_question": question},
                path=settings.rewrite_capture_path,
            )
        return {
            "resolved_question": question,
            "scope": {**state["scope"], "resolved_question": question},
        }

    @timed_node
    async def router(state: RoutingState) -> dict[str, Any]:
        """路由调用、解析和类型校验整体最多尝试两次，耗尽后向上报告。"""
        decision: RouteDecision = await decide_route(
            glm_4_flash_llm, state["resolved_question"]
        )
        if settings.routing_capture_enabled:
            append_node_candidate(
                node="router",
                scope=state["scope"],
                input={"resolved_question": state["resolved_question"]},
                actual_output=decision,
                path=settings.routing_capture_path,
            )
        return decision.model_dump()

    workflow = StateGraph(RoutingState)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("router", router)
    workflow.add_node("crypto_analysis_agent", crypto_agent)
    workflow.add_node("topic_research_agent", topic_agent)
    workflow.add_node("finalize", finalize_report)
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
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer, name="research_routing")
