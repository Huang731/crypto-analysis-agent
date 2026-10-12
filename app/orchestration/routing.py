"""单次路由到研究专家，完成后统一整理报告。"""

import logging
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import Self, TypedDict

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_agent import build_dynamic_analysis_agent
from app.crypto_analysis.skill_memory import merge_skill_memory
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


class RouteCallState(BaseModel):
    """本次调用显式持有的诊断信息，不进入业务图的会话状态。"""

    attempt_count: int = 0
    invalid_model_output: bool = False


@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    before_sleep=before_sleep_log(_LOG, logging.WARNING),
    reraise=True,
)
async def decide_route(
    smart: BaseChatModel,
    resolved_question: str,
    *,
    call_state: RouteCallState | None = None,
) -> RouteDecision:
    """只执行路由，沿用模型工厂的超时和整体最多两次尝试。"""
    if call_state is not None:
        call_state.attempt_count += 1
        call_state.invalid_model_output = False
    messages = [
        SystemMessage(content=routing_prompt()),
        HumanMessage(content=resolved_question),
    ]
    return await _invoke_route_decision(smart, messages, call_state)


@trace_generation_call(
    name="routing_decision",
    model=settings.smart_model,
    input_arg="messages",
    input_format="messages",
    output_format="json",
)
async def _invoke_route_decision(
    smart: BaseChatModel,
    messages: list[SystemMessage | HumanMessage],
    call_state: RouteCallState | None,
) -> RouteDecision:
    """保留原特殊返回分支及无效模型输出的诊断标记。"""
    decision = await smart.with_structured_output(
        RouteDecision, method="function_calling"
    ).ainvoke(messages)
    if not isinstance(decision, RouteDecision):
        if call_state is not None:
            call_state.invalid_model_output = True
        return RouteDecision(
            route="direct", direct_reply="暂时无法处理这次提问，请重试。"
        )
    return decision


class RoutingCandidateInput(BaseModel):
    """保留路由模型实际收到的问题。"""

    model_config = ConfigDict(extra="forbid")
    resolved_question: str


class RoutingCandidate(BaseModel):
    """实际输出仅用于候选记录，不自动生成预期答案。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    original_question: str
    input: RoutingCandidateInput
    actual_output: RouteDecision


def append_routing_candidate(
    *,
    original_question: str,
    resolved_question: str,
    decision: RouteDecision,
    path: str | Path,
) -> None:
    """整条序列化后一次追加，保留中文和已有候选记录。"""
    candidate = RoutingCandidate(
        original_question=original_question,
        input=RoutingCandidateInput(resolved_question=resolved_question),
        actual_output=decision,
    )
    line = candidate.model_dump_json(ensure_ascii=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line)


class RoutingState(TypedDict, total=False):
    """外层只保留路由和报告，研究明细由专家子图持有。"""

    # 本轮请求范围：分析截止时间、会话和追踪标识、原始及补全问题等。
    scope: dict[str, str]
    # 本轮原话规范改写后的问题，用于路由和专家分析。
    resolved_question: str
    # 最近一轮已完成回答的问题，保留补全后的主题供连续追问使用。
    previous_question: str
    # 同会话各技能的指标表格与思想体系解读，按原共享范围和技能名保存。
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
    return {"report": report, "previous_question": state["resolved_question"]}


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
        """结合上一轮问题补全本轮问题，保留用户原话。"""
        original = state["scope"]["original_question"]
        with trace_span("question_rewriter", question_chars=len(original)) as span:
            question = await rewrite_question(
                fast, original, state.get("previous_question", "")
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
