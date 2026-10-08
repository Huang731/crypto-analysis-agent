"""单次路由到研究专家，完成后统一整理报告。"""

import asyncio
import json
from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self, TypedDict

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_agent import build_dynamic_analysis_agent
from app.rag.retriever import KBRetriever
from app.topic_research.topic_research_agent import build_topic_research_agent
from app.tracing import estimate_tokens, trace_generation
from prompts.routing import routing_prompt

RouteTarget = Literal["crypto_analysis_agent", "topic_research_agent", "direct"]
_ROUTE_TIMEOUT_SECONDS = 60


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


class RoutingState(TypedDict, total=False):
    """外层只保留路由和报告，研究明细由专家子图持有。"""

    messages: Annotated[list[Any], add_messages]
    scope: dict[str, str]
    route: RouteTarget
    direct_reply: str
    report: str


def select_agent(state: RoutingState) -> RouteTarget:
    """根据已校验的路由结果选择唯一分支。"""
    return state["route"]


def finalize_report(state: RoutingState) -> dict[str, Any]:
    """统一承接已有报告，不重新调用模型或改写专家结论。"""
    report = state["direct_reply"] if state["route"] == "direct" else state["report"]
    if not isinstance(report, str) or not report.strip():
        raise ValueError("最终报告为空")
    if state["route"] == "direct":
        return {"report": report, "messages": [AIMessage(content=report)]}
    # 专家已经写入报告消息，汇总时不重复追加。
    return {"report": report}


def build_routing_graph(
    smart: BaseChatModel, fast: BaseChatModel, retriever: KBRetriever
) -> CompiledStateGraph:
    """保留现有构建入口，以显式图保证每轮只执行一个专家。"""
    crypto_agent = build_dynamic_analysis_agent(smart, fast, retriever)
    topic_agent = build_topic_research_agent(smart, fast)
    # 结构化结果包含字段组合校验；失败重试仍属于同一次路由，不进入专家。
    route_model = smart.with_structured_output(
        RouteDecision, method="json_mode"
    ).with_retry(stop_after_attempt=2)

    async def router(state: RoutingState) -> dict[str, Any]:
        """在固定超时内识别意图，异常由图和现有请求日志向上报告。"""
        messages = [SystemMessage(content=routing_prompt()), *state["messages"]]
        with trace_generation(
            "routing_decision", model=settings.smart_model
        ) as generation:
            decision = await asyncio.wait_for(
                route_model.ainvoke(messages),
                timeout=_ROUTE_TIMEOUT_SECONDS,
            )
            if not isinstance(decision, RouteDecision):
                raise TypeError("路由模型未返回 RouteDecision 结构化结果")
            # 沿用项目估算口径，仅统计本次输入和最终有效的结构化结果。
            generation.usage = {
                "input": estimate_tokens(
                    json.dumps(
                        [message.model_dump(mode="json") for message in messages],
                        ensure_ascii=False,
                    )
                ),
                "output": estimate_tokens(decision.model_dump_json()),
            }
        return decision.model_dump()

    workflow = StateGraph(RoutingState)
    workflow.add_node("router", router)
    workflow.add_node("crypto_analysis_agent", crypto_agent)
    workflow.add_node("topic_research_agent", topic_agent)
    workflow.add_node("finalize", finalize_report)
    workflow.add_edge(START, "router")
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
    return workflow.compile(name="research_routing")
