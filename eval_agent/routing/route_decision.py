"""共用路由调用与候选采集，保留原有重试和返回契约。"""

import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import Self

from app.config.config import settings
from app.observability import get_logger
from app.tracing import trace_generation_call
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
    """保留原特殊返回分支，同时让评测区分无效输出与正确的直接回复。"""
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
