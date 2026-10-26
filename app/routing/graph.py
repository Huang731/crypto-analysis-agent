"""单次路由到研究专家，完成后统一整理报告。"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    messages_to_dict,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import TypedDict

from app.core.candidate_capture import append_node_candidate
from app.core.config import settings
from app.core.observability import timed_node
from app.core.tracing import trace_generation_call
from app.crypto_analysis.crypto_analysis_parent_agent import (
    build_crypto_analysis_parent_agent,
)
from app.llm.model import ainvoke_model
from app.memory.memory_manager import MemoryManager
from app.rag.retriever import KBRetriever
from app.routing.query_rewriter import rewrite_question
from app.topic_research.topic_research_agent import build_topic_research_agent
from prompts.router import routing_messages


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
    response = await ainvoke_model(
        llm,
        messages,
    )
    if not isinstance(response.content, str):
        raise ValueError("路由模型未返回文本")
    route = response.content.strip()
    if route not in ("crypto_analysis_agent", "topic_research_agent"):
        raise ValueError(f"路由模型返回了无效 Agent 名称：{route!r}")
    return route


class RoutingState(TypedDict, total=False):
    """外层保存会话消息、路由和报告，研究明细由专家子图持有。"""

    # 节点返回完整消息列表，超过阈值后由摘要和近期原话覆盖旧历史。
    messages: list[AnyMessage]
    # 本轮请求范围：分析截止时间、会话和追踪标识、原始及补全问题等。
    scope: dict[str, str]
    # 本轮原话规范改写后的问题，用于路由和专家分析。
    resolved_question: str
    # 本轮路由目标：加密分析或主题研究。
    route: str
    # 本轮最终报告，由专家生成。
    report: str


def select_agent(state: RoutingState) -> str:
    """根据已校验的路由结果选择唯一分支。"""
    return state["route"]


def make_summarize_history(
    glm_4_flash_llm: BaseChatModel,
) -> Callable[[RoutingState], Awaitable[dict[str, list[AnyMessage]]]]:
    """追加本轮回答，超过阈值时压缩旧历史，再整体保存消息。"""
    memory_manager = MemoryManager(3, 2)

    @timed_node(label=lambda state: state["resolved_question"])
    @trace_generation_call(name="routing.summarize_history", kind="span")
    async def summarize_history(state: RoutingState) -> dict[str, list[AnyMessage]]:
        report = state["report"]
        if not isinstance(report, str) or not report.strip():
            raise ValueError("最终报告为空")
        messages = [*state["messages"], AIMessage(content=report)]
        messages = await memory_manager.manage(glm_4_flash_llm, messages)
        if settings.summarize_history_capture_enabled:
            append_node_candidate(
                node="summarize_history",
                input={
                    "messages": messages_to_dict(state["messages"]),
                    "report": report,
                },
                actual_output={"messages": messages_to_dict(messages)},
                path=settings.summarize_history_capture_path,
            )
        return {"messages": messages}

    return summarize_history


def build_routing_graph(
    glm_4_5_air_llm: BaseChatModel,
    retriever: KBRetriever,
    *,
    checkpointer: BaseCheckpointSaver,
    glm_4_flash_llm: BaseChatModel,
) -> CompiledStateGraph:
    """保留现有构建入口，以显式图保证每轮只执行一个专家。"""
    crypto_agent = build_crypto_analysis_parent_agent(
        retriever,
        glm_4_5_air_llm=glm_4_5_air_llm,
        glm_4_flash_llm=glm_4_flash_llm,
    )
    topic_agent = build_topic_research_agent(
        glm_4_flash_llm,
        glm_4_5_air_llm=glm_4_5_air_llm,
    )

    @timed_node(label=lambda state: state["scope"]["original_question"])
    async def rewrite(state: RoutingState) -> dict[str, Any]:
        """追加本轮原话后筛选全部用户消息改写，完整消息列表仍整体保存。"""
        messages = [
            *state.get("messages", []),
            HumanMessage(content=state["scope"]["original_question"]),
        ]
        user_messages = [
            message for message in messages if isinstance(message, HumanMessage)
        ]
        question = await rewrite_question(
            glm_4_flash_llm,
            user_messages,
        )
        return {
            "messages": messages,
            "resolved_question": question,
            "scope": {**state["scope"], "resolved_question": question},
        }

    @timed_node(label=lambda state: state["resolved_question"])
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
    workflow.add_node("summarize_history", make_summarize_history(glm_4_flash_llm))
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
    workflow.add_edge("crypto_analysis_agent", "summarize_history")
    workflow.add_edge("topic_research_agent", "summarize_history")
    workflow.add_edge("summarize_history", END)
    return workflow.compile(checkpointer=checkpointer, name="research_routing")
