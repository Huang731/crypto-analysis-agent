"""加密货币分析：请求规划、技能执行、并行汇总及报告生成。"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, messages_to_dict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Overwrite, Send, StreamWriter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.candidate_capture import append_node_candidate
from app.config.config import settings
from app.crypto_analysis.crypto_analysis_state import (
    DynamicAnalysisState,
    SkillWorkerGraphState,
)
from app.crypto_analysis.crypto_analysis_sub_agent import (
    build_crypto_analysis_sub_agent,
    skill_worker_step_limit,
)
from app.crypto_analysis.skill_loader import get_skill_loader
from app.crypto_analysis.skill_memory import (
    read_skill_memory,
    resolve_memory_key,
)
from app.llm.model import (
    ainvoke_model,
    ainvoke_structured,
)
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import trace_generation_call
from prompts.crypto_analysis import (
    analysis_plan_messages,
    analysis_report_messages,
    analysis_review_messages,
    history_summary_messages,
)

_LOG = get_logger("crypto_analysis.dynamic_analysis")


@trace_generation_call(
    name="crypto_history_summary",
    model=settings.glm_4_flash_model,
    input_arg="messages",
    input_format="messages",
    record_io=True,
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_history_summary(
    llm: BaseChatModel, messages: list[BaseMessage]
) -> str:
    """直接生成摘要正文，保留正文非空校验与有限重试。"""
    # 历史摘要使用 GLM-4-Flash，调用与正文校验整体重试。
    response = await ainvoke_model(llm, messages)
    summary = response.content.strip()
    if not summary:
        raise ValueError("历史摘要不能为空")
    return summary


@trace_generation_call(
    name="crypto_analysis_writer",
    model=settings.glm_smart_model,
    input_arg="messages",
    input_format="messages",
    record_io=True,
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_report_text(
    llm: Any,
    messages: list[BaseMessage],
    stream_writer: StreamWriter,
) -> str:
    """生成正文前通知开始，完成后校验报告。"""
    stream_writer({"event": "report_start", "node": "writer"})
    response = await ainvoke_model(llm, messages)
    report = response.content.strip()
    if not report:
        raise ValueError("报告正文不能为空")
    return report


# 请求规划模型


class AnalysisPlan(BaseModel):
    """一轮分析的目标币种和所需技能，不指定具体工具。"""

    model_config = ConfigDict(extra="forbid")
    target_asset: str = Field(pattern=r"^[A-Z0-9]+$", strict=True)
    skill_names: list[str]
    refresh: bool = Field(
        strict=True,
        description="用户是否明确要求重新获取数据，跳过已有技能缓存",
    )

    @field_validator("skill_names")
    @classmethod
    def unique_skill_names(cls, names: list[str]) -> list[str]:
        """按首次出现顺序去重，避免并行派发同一技能多次。"""
        return list(dict.fromkeys(names))


class ReportReview(BaseModel):
    """审核返回证据核对记录、是否通过和修改意见，不修改技能证据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    review_process: str = Field(
        strict=True,
        min_length=1,
        description="逐项对照报告原文与证据原文或表格数据的具体核对记录",
    )
    passed: bool = Field(strict=True, description="报告是否符合本轮证据和用户问题")
    feedback: str = Field(strict=True, description="不通过时的具体问题、依据及修改要求")

    @model_validator(mode="after")
    def require_rework_feedback(self) -> "ReportReview":
        """拒绝没有修改目标的不通过结果，使解析错误进入调用重试。"""
        if not self.passed and not self.feedback:
            raise ValueError("报告审核不通过时必须提供修改意见")
        return self


@trace_generation_call(
    name="crypto_report_review",
    model=settings.glm_smart_model,
    input_arg="messages",
    input_format="messages",
    output_format="json",
    record_io=True,
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_report_review(llm: Any, messages: list[BaseMessage]) -> ReportReview:
    """记录审核用量，对结构化调用和校验整体重试。"""
    review = await ainvoke_structured(llm, messages, schema=ReportReview)
    if review is None:
        raise ValueError("审核模型返回了空结果")
    return review


@trace_generation_call(
    name="crypto_analysis_plan",
    model=settings.glm_4_flash_model,
    input_arg="messages",
    input_format="messages",
    output_format="json",
    record_io=True,
    kind="generation",
)
@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_analysis_plan(llm: Any, messages: list[BaseMessage]) -> AnalysisPlan:
    """记录规划用量，对结构化调用和校验整体重试。"""
    plan = await ainvoke_structured(llm, messages, schema=AnalysisPlan)
    if plan is None:
        raise ValueError("规划模型返回了空计划")
    return plan


# 请求规划


def create_request_scope(
    question: str, thread_id: str, trace_id: str
) -> dict[str, str]:
    """记录供前端展示的 UTC 分析时间和请求追踪标识。"""
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "trace_id": trace_id,
    }


async def create_analysis_plan(
    llm: Any,
    question: str,
    skill_descriptions: str,
) -> dict[str, Any]:
    """LLM 根据任务和技能描述选择资产、研究重点与所需 Skill。"""
    messages = analysis_plan_messages(skill_descriptions, question)
    plan = await _invoke_analysis_plan(llm, messages)
    return plan.model_dump()


def make_skill_worker(
    sub_graph: CompiledStateGraph,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """沿用父图调用边界，转交技能子图执行。"""

    @timed_node(label=lambda state: state["skill_name"])
    async def skill_worker(state: SkillWorkerGraphState) -> dict[str, Any]:
        name = state["skill_name"]
        skill_loader = get_skill_loader()
        skill = skill_loader.load_definition(name)
        # 按指标数量限制子图步数，避免缺失结果时无限循环。
        result = await sub_graph.ainvoke(
            state,
            config={
                "recursion_limit": skill_worker_step_limit(len(skill.scoring.metrics)),
            },
        )
        update = {"skill_results": [result["skill_result"]]}
        if "skill_memory" in result:
            update["skill_memory"] = result["skill_memory"]
        return update

    return skill_worker


# 分析父图：规划、并行技能、报告


def make_plan(
    glm_4_flash_llm: BaseChatModel,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, Any]]]:
    """创建本轮技能规划节点。"""

    @timed_node
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:
        skill_loader = get_skill_loader()
        question = state["scope"].get("resolved_question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("技能分析缺少有效用户问题")
        selection = await create_analysis_plan(
            glm_4_flash_llm,
            question,
            skill_loader.format_skill_descriptions(),
        )
        available_names = {skill.name for skill in skill_loader.list_skills()}
        selection["skill_names"] = [
            name for name in selection["skill_names"] if name in available_names
        ]
        if settings.planning_capture_enabled:
            append_node_candidate(
                node="plan",
                scope=state["scope"],
                input={"resolved_question": question},
                actual_output=selection,
                path=settings.planning_capture_path,
            )
        scope = {
            **state["scope"],
            "target_asset": selection["target_asset"],
        }
        return {
            "report": "",
            # 每轮请求独立计数，上一轮审核意见不能影响新报告。
            "review_decision": "pending",
            "review_count": 0,
            "review_feedback": "",
            "plan": selection,
            "scope": scope,
            # 缓存跨轮保留，追加型报告输入每轮从空列表开始。
            "skill_results": Overwrite([]),
        }

    return plan


@timed_node
def send_skills(state: DynamicAnalysisState) -> list[Send] | str:
    names = state["plan"]["skill_names"]
    if not names:
        return "writer"
    return [
        Send(
            "skill_worker",
            {
                "skill_name": name,
                "scope": state["scope"],
                "cached_memory": (
                    None
                    if state["plan"]["refresh"]
                    else read_skill_memory(
                        state.get("skill_memory", {}),
                        resolve_memory_key(state["scope"]["target_asset"], name),
                    )
                ),
            },
        )
        for name in names
    ]


def make_writer(
    glm_smart_llm: BaseChatModel,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """创建投研报告写作节点，保留原证据和重写逻辑。"""

    @timed_node
    async def writer(state: DynamicAnalysisState, runtime: Runtime) -> dict[str, Any]:
        """依据技能内容与历史摘要撰写结论，报告只包含结论正文。"""
        ordered = sorted(state["skill_results"], key=lambda item: item["skill_name"])
        # 只采用本轮技能内容，历史摘要不能替代本轮证据。
        messages = analysis_report_messages(
            target_asset=state["scope"]["target_asset"],
            original_question=state["scope"]["resolved_question"],
            content=[
                {
                    "skill_name": item["skill_name"],
                    "name": item["name"],
                    "content": item["content"],
                    "cache_hit": item["cache_hit"],
                }
                for item in ordered
            ],
            thought_context={
                item["skill_name"]: item["thought_context"] for item in ordered
            },
            history_summary=state.get("history_summary", ""),
            review_feedback=state["review_feedback"],
            previous_report=state["report"],
        )
        # 仅本节点关闭深度思考，不修改共享模型的配置。
        writer_llm = glm_smart_llm.bind(
            extra_body={
                "thinking": {"type": "disabled"},
            }
        )
        report = await _invoke_report_text(
            writer_llm,
            messages,
            runtime.stream_writer,
        )
        if settings.writer_capture_enabled:
            append_node_candidate(
                node="writer",
                scope=state["scope"],
                input={
                    "messages": messages_to_dict(messages),
                    "model": settings.glm_smart_model,
                    "review_count": state["review_count"],
                },
                actual_output={"report": report},
                path=settings.writer_capture_path,
            )
        return {
            "report": report,
        }

    return writer


def make_reviewer(
    glm_smart_llm: BaseChatModel,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, Any]]]:
    """创建报告审核节点，保留原有审核规则。"""

    @timed_node
    @trace_generation_call(name="crypto.reviewer", kind="span")
    async def reviewer(state: DynamicAnalysisState) -> dict[str, Any]:
        """审核本轮报告，沿用两轮审核上限和不通过时的修改意见。"""
        review_count = state["review_count"] + 1
        messages = analysis_review_messages(
            target_asset=state["scope"]["target_asset"],
            original_question=state["scope"]["resolved_question"],
            report=state["report"],
            skill_results=state["skill_results"],
            history_summary=state.get("history_summary", ""),
        )
        # 复用结构化调用的流标记，审核内容不进入报告正文。
        review = await _invoke_report_review(glm_smart_llm, messages)
        if not review.passed and review_count >= 2:
            update = {
                "review_decision": "pass",
                "review_count": review_count,
                "review_feedback": "已达重写2次上限，强制通过。",
            }
        else:
            update = {
                "review_decision": "pass" if review.passed else "rework",
                "review_count": review_count,
                "review_feedback": "" if review.passed else review.feedback,
            }
        if settings.reviewer_capture_enabled:
            append_node_candidate(
                node="reviewer",
                scope=state["scope"],
                input={
                    "messages": messages_to_dict(messages),
                    "model": settings.glm_smart_model,
                    "review_count": review_count,
                },
                actual_output={
                    "review": review.model_dump(),
                    "state_update": update,
                },
                path=settings.reviewer_capture_path,
            )
        return update

    return reviewer


def route_after_review(state: DynamicAnalysisState) -> str:
    """只有审核通过的报告进入历史摘要，失败时保留证据并重写。"""
    if state["review_decision"] == "pass":
        return "summarize_history"
    return "writer"


def make_summarize_history(
    glm_4_flash_llm: BaseChatModel,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, str]]]:
    """创建累计历史摘要节点。"""

    @timed_node
    async def summarize_history(state: DynamicAnalysisState) -> dict[str, str]:
        """根据本轮最终报告更新累计结论摘要。"""
        messages = history_summary_messages(
            previous_summary=state.get("history_summary", ""),
            report=state["report"],
        )
        summary = await _invoke_history_summary(glm_4_flash_llm, messages)
        update = {"history_summary": summary}
        if settings.summarize_history_capture_enabled:
            append_node_candidate(
                node="summarize_history",
                scope=state["scope"],
                input={
                    "plan": state["plan"],
                    "previous_summary": state.get("history_summary", ""),
                    "report": state["report"],
                    "messages": messages_to_dict(messages),
                },
                actual_output=update,
                path=settings.summarize_history_capture_path,
            )
        return update

    return summarize_history


def build_crypto_analysis_parent_agent(
    glm_smart_llm: BaseChatModel,
    bailian_fast_llm: Any,
    retriever: KBRetriever,
    *,
    glm_4_flash_llm: BaseChatModel,
) -> CompiledStateGraph:
    """以节点工厂组装规划、技能子图和报告父图。"""
    sub_graph = build_crypto_analysis_sub_agent(bailian_fast_llm, retriever)
    builder = StateGraph(DynamicAnalysisState)
    builder.add_node("plan", make_plan(glm_4_flash_llm))
    builder.add_node("skill_worker", make_skill_worker(sub_graph))
    builder.add_node("writer", make_writer(glm_smart_llm))
    builder.add_node("reviewer", make_reviewer(glm_smart_llm))
    builder.add_node("summarize_history", make_summarize_history(glm_4_flash_llm))
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", send_skills)
    builder.add_edge("skill_worker", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"summarize_history": "summarize_history", "writer": "writer"},
    )
    builder.add_edge("summarize_history", END)
    return builder.compile(name="crypto_analysis_agent")
