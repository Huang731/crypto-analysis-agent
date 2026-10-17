"""单次路由到研究专家，完成后统一整理报告。"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, messages_to_dict
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import TypedDict

from app.candidate_capture import append_node_candidate
from app.config.config import settings
from app.crypto_analysis.crypto_analysis_parent_agent import (
    build_crypto_analysis_parent_agent,
)
from app.llm.model import ainvoke_model
from app.observability import get_logger, timed_node
from app.orchestration.query_rewriter import rewrite_question
from app.rag.retriever import KBRetriever
from app.topic_research.topic_research_agent import build_topic_research_agent
from app.tracing import trace_generation_call
from prompts.routing import routing_messages

_LOG = get_logger("crypto_analysis.routing")


@trace_generation_call(
    name="routing_decision",
    model=settings.glm_4_flash_model,
    input_arg="messages",
    input_format="messages",
    output_format="text",
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_route_decision(
    llm: BaseChatModel,
    messages: list[BaseMessage],
) -> str:
    """保留路由追踪，对文本调用和 Agent 名称校验整体重试。"""
    response = await ainvoke_model(llm, messages, timeout=120)
    if not isinstance(response.content, str):
        raise ValueError("路由模型未返回文本")
    route = response.content.strip()
    if route not in ("crypto_analysis_agent", "topic_research_agent"):
        raise ValueError(f"路由模型返回了无效 Agent 名称：{route!r}")
    return route


class RoutingState(TypedDict, total=False):
    """外层只保留路由和报告，研究明细由专家子图持有。"""

    # 本轮请求范围：分析截止时间、会话和追踪标识、原始及补全问题等。
    scope: dict[str, str]
    # 本轮原话规范改写后的问题，用于路由和专家分析。
    resolved_question: str
    # 最近一轮已完成回答的问题，保留补全后的主题供连续追问使用。
    previous_question: str
    # 本轮路由目标：加密分析或主题研究。
    route: str
    # 本轮最终报告，由专家生成。
    report: str


def select_agent(state: RoutingState) -> str:
    """根据已校验的路由结果选择唯一分支。"""
    return state["route"]


def finalize_report(state: RoutingState) -> dict[str, Any]:
    """统一承接已有报告，校验成功后保存本轮问题供下轮改写。"""
    report = state["report"]
    if not isinstance(report, str) or not report.strip():
        raise ValueError("最终报告为空")
    update = {"report": report, "previous_question": state["resolved_question"]}
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
        question = await rewrite_question(
            glm_smart_llm, original, state.get("previous_question", "")
        )
        return {
            "resolved_question": question,
            "scope": {**state["scope"], "resolved_question": question},
        }

    @timed_node
    async def router(state: RoutingState) -> dict[str, Any]:
        """路由调用、解析和类型校验整体最多尝试两次，耗尽后向上报告。"""
        messages = routing_messages(state["resolved_question"])
        route = await _invoke_route_decision(
            glm_4_flash_llm,
            messages,
        )
        if settings.routing_capture_enabled:
            append_node_candidate(
                node="router",
                input=messages_to_dict(messages),
                actual_output=route,
                path=settings.routing_capture_path,
            )
        return {"route": route}

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
        },
    )
    workflow.add_edge("crypto_analysis_agent", "finalize")
    workflow.add_edge("topic_research_agent", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer, name="research_routing")
