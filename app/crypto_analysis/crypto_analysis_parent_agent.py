"""加密货币分析：请求规划、技能执行、并行汇总及报告生成。"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
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

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_state import (
    DynamicAnalysisState,
    SkillWorkerGraphState,
)
from app.crypto_analysis.crypto_analysis_sub_agent import (
    build_crypto_analysis_sub_agent,
    skill_worker_step_limit,
)
from app.crypto_analysis.skill_loader import (
    SkillLoader,
    get_skill_loader,
)
from app.crypto_analysis.skill_memory import (
    read_skill_memory,
    resolve_memory_key,
)
from app.llm.model import (
    ainvoke_structured,
    ainvoke_text,
    with_thinking,
)
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import trace_generation_call, trace_span
from prompts.crypto_analysis import (
    analysis_plan_prompt,
    analysis_report_prompt,
    analysis_review_prompt,
    history_summary_prompt,
)

_LOG = get_logger("crypto_analysis.dynamic_analysis")


@trace_generation_call(name="crypto_history_summary", model=settings.fast_model)
async def _invoke_history_summary(llm: BaseChatModel, prompt: str) -> str:
    """直接生成摘要正文，沿用非空、长度校验和有限重试。"""
    # 历史摘要使用执行模型，沿用模型工厂的思考配置。
    summary = await ainvoke_text(llm, prompt)
    if not summary:
        raise ValueError("历史摘要不能为空")
    return summary


async def _invoke_structured_output(
    llm: Any,
    schema: type[BaseModel],
    prompt: str,
    trace_name: str,
    model_name: str,
) -> BaseModel:
    """函数调用、解析和类型校验整体重试，保留流标记和用量追踪。"""
    return await _invoke_structured_model(llm, schema, prompt, trace_name, model_name)


@trace_generation_call(
    name=lambda arguments: arguments.arguments["trace_name"],
    model=lambda arguments: arguments.arguments["model_name"],
    output_format="json",
)
async def _invoke_structured_model(
    model: Any,
    schema: type[BaseModel],
    prompt: str,
    trace_name: str,
    model_name: str,
) -> BaseModel:
    """保留模型结果的类型校验，由装饰器记录本次调用用量。"""
    response = await ainvoke_structured(model, prompt, schema=schema)
    if not isinstance(response, schema):
        raise TypeError(f"模型未返回 {schema.__name__} 结构化结果")
    return response


@trace_generation_call(name="crypto_analysis_writer", model=settings.smart_model)
async def _invoke_report_text(
    llm: Any,
    prompt: str,
    stream_writer: StreamWriter,
) -> str:
    """流式生成正文，重试时重新开始展示，完成后校验报告。"""
    report = await ainvoke_text(
        llm,
        prompt,
        on_attempt=lambda: stream_writer({"event": "report_start", "node": "writer"}),
    )
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
        """拒绝重复技能，避免并行派发同一技能多次。"""
        if len(names) != len(set(names)):
            raise ValueError("分析计划中的技能名称不能重复")
        return names


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


async def _invoke_report_review(llm: Any, prompt: str) -> ReportReview:
    """单次审核限定超时，调用、解析或超时异常整体最多尝试两次。"""
    result = await _invoke_structured_model(
        llm, ReportReview, prompt, "crypto_report_review", settings.smart_model
    )
    return ReportReview.model_validate(result)


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
    prompt = analysis_plan_prompt(skill_descriptions, question)
    plan = await _invoke_structured_output(
        llm,
        AnalysisPlan,
        prompt,
        "crypto_analysis_plan",
        settings.smart_model,
    )
    return plan.model_dump()


class PlanningCandidateInput(BaseModel):
    """保留规划模型实际收到的问题。"""

    model_config = ConfigDict(extra="forbid")
    resolved_question: str


class PlanningCandidate(BaseModel):
    """实际计划已经由生产规划函数校验，仅作为候选输出保留。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    original_question: str
    input: PlanningCandidateInput
    actual_output: dict[str, Any]


def append_planning_candidate(
    *,
    original_question: str,
    resolved_question: str,
    decision: dict[str, Any],
    path: str | Path,
) -> None:
    """一次追加完整 JSONL 记录，保留中文和已有候选。"""
    candidate = PlanningCandidate(
        original_question=original_question,
        input=PlanningCandidateInput(resolved_question=resolved_question),
        actual_output=decision,
    )
    line = candidate.model_dump_json(ensure_ascii=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line)


def make_skill_worker(
    sub_graph: CompiledStateGraph,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """沿用父图调用边界和整体追踪，转交技能子图执行。"""

    @timed_node(label=lambda state: state["skill_name"])
    async def skill_worker(
        state: SkillWorkerGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        with trace_span("crypto.skill_worker", skill_name=state["skill_name"]) as span:
            try:
                name = state["skill_name"]
                skill = SkillLoader.from_snapshot(
                    {name: state["skill_source"]}
                ).load_definition(name)
                # 按指标数量限制子图步数，避免缺失结果时无限循环。
                result = await sub_graph.ainvoke(
                    state,
                    config={
                        **config,
                        "recursion_limit": skill_worker_step_limit(
                            len(skill.scoring.metrics)
                        ),
                    },
                )
                span.metadata["cache_hit"] = result["skill_result"]["cache_hit"]
                update = {"skill_results": [result["skill_result"]]}
                if "skill_memory" in result:
                    update["skill_memory"] = result["skill_memory"]
                return update
            except Exception:
                span.metadata["status"] = "error"
                raise

    return skill_worker


# 分析父图：规划、并行技能、报告


def make_plan(
    fast_llm: Any,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, Any]]]:
    """创建本轮技能规划节点。"""

    @timed_node
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:
        with trace_span("crypto.plan") as node_span:
            try:
                skill_loader = get_skill_loader()
                question = state["scope"].get("resolved_question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("技能分析缺少有效用户问题")
                selection = await create_analysis_plan(
                    fast_llm,
                    question,
                    skill_loader.format_skill_descriptions(),
                )
                if settings.planning_capture_enabled:
                    append_planning_candidate(
                        original_question=state["scope"]["original_question"],
                        resolved_question=question,
                        decision=selection,
                        path=settings.planning_capture_path,
                    )
                scope = {
                    **state["scope"],
                    "target_asset": selection["target_asset"],
                }
                history_summary = state.get("history_summary", "")
                if not isinstance(history_summary, str):
                    raise TypeError("历史摘要必须是字符串")
                return {
                    "history_summary": history_summary,
                    "report": "",
                    # 每轮请求独立计数，上一轮审核意见不能影响新报告。
                    "review_decision": "pending",
                    "review_count": 0,
                    "review_feedback": "",
                    "skill_sources": skill_loader.snapshot(),
                    "plan": selection,
                    "scope": scope,
                    # 缓存跨轮保留，追加型报告输入每轮从空列表开始。
                    "skill_results": Overwrite([]),
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

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
                "skill_source": state["skill_sources"][name],
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


def make_writer(smart_llm: Any) -> Callable[..., Awaitable[dict[str, Any]]]:
    """创建投研报告写作节点，保留原证据和重写逻辑。"""

    @timed_node
    async def writer(state: DynamicAnalysisState, runtime: Runtime) -> dict[str, Any]:
        """依据技能内容与历史摘要撰写结论，报告只包含结论正文。"""
        with trace_span("crypto.writer") as node_span:
            try:
                if not state["plan"]["skill_names"]:
                    report = "当前任务没有适用的分析技能；请说明希望分析的资产或问题。"
                    return {
                        "report": report,
                    }
                ordered = sorted(
                    state["skill_results"], key=lambda item: item["skill_name"]
                )
                # 只采用本轮技能内容，历史摘要不能替代本轮证据。
                prompt = analysis_report_prompt(
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
                    history_summary=state["history_summary"],
                    review_feedback=state["review_feedback"],
                    previous_report=state["report"],
                )
                # 仅本节点关闭深度思考，不修改共享模型的配置。
                writer_llm = with_thinking(smart_llm, False)
                report = await _invoke_report_text(
                    writer_llm,
                    prompt,
                    runtime.stream_writer,
                )
                return {
                    "report": report,
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return writer


def make_reviewer(
    smart_llm: Any,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, Any]]]:
    """创建报告审核节点，保留原有审核规则。"""

    @timed_node
    async def reviewer(state: DynamicAnalysisState) -> dict[str, Any]:
        """审核本轮报告，最多审核五轮，未通过时带意见重写。"""
        with trace_span("crypto.reviewer") as node_span:
            try:
                # 无适用技能时沿用原有说明回复，无需审核投研内容。
                if not state["plan"]["skill_names"]:
                    return {"review_decision": "pass"}
                review_count = state["review_count"] + 1
                prompt = analysis_review_prompt(
                    target_asset=state["scope"]["target_asset"],
                    original_question=state["scope"]["resolved_question"],
                    report=state["report"],
                    skill_results=sorted(
                        state["skill_results"], key=lambda item: item["skill_name"]
                    ),
                    history_summary=state["history_summary"],
                )
                # 复用结构化调用的流标记，审核内容不进入报告正文。
                review = await _invoke_report_review(smart_llm, prompt)
                node_span.output = {
                    "review_process": review.review_process,
                    "passed": review.passed,
                    "review_count": review_count,
                    "feedback": review.feedback,
                }
                if not review.passed and review_count >= 2:
                    return {
                        "review_decision": "pass",
                        "review_count": review_count,
                        "review_feedback": "已达重写2次上限，强制通过。",
                    }
                return {
                    "review_decision": "pass" if review.passed else "rework",
                    "review_count": review_count,
                    "review_feedback": "" if review.passed else review.feedback,
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return reviewer


def route_after_review(state: DynamicAnalysisState) -> str:
    """只有审核通过的报告进入历史摘要，失败时保留证据并重写。"""
    if state["review_decision"] == "pass":
        return "summarize_history"
    return "writer"


def make_summarize_history(
    fast_llm: Any,
) -> Callable[[DynamicAnalysisState], Awaitable[dict[str, str]]]:
    """创建累计历史摘要节点。"""

    @timed_node
    async def summarize_history(state: DynamicAnalysisState) -> dict[str, str]:
        """在报告生成后更新累计结论摘要，无适用技能时保留已有摘要。"""
        if not state["plan"]["skill_names"]:
            return {}
        prompt = history_summary_prompt(
            previous_summary=state.get("history_summary", ""),
            report=state["report"],
        )
        summary = await _invoke_history_summary(fast_llm, prompt)
        return {"history_summary": summary}

    return summarize_history


def build_crypto_analysis_parent_agent(
    smart_llm: Any, fast_llm: Any, retriever: KBRetriever
) -> CompiledStateGraph:
    """以节点工厂组装规划、技能子图和报告父图。"""
    sub_graph = build_crypto_analysis_sub_agent(fast_llm, retriever)
    builder = StateGraph(DynamicAnalysisState)
    builder.add_node("plan", make_plan(fast_llm))
    builder.add_node("skill_worker", make_skill_worker(sub_graph))
    builder.add_node("writer", make_writer(smart_llm))
    builder.add_node("reviewer", make_reviewer(smart_llm))
    builder.add_node("summarize_history", make_summarize_history(fast_llm))
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
