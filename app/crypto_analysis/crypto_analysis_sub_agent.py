"""加密资产技能执行子图：缓存、指标循环、内容生成及结果收尾。"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from math import fsum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_state import (
    SkillWorkerGraphState,
)
from app.crypto_analysis.metric_service import metric_field_labels
from app.crypto_analysis.skill_loader import LoadedSkill, SkillLoader
from app.crypto_analysis.skill_memory import (
    SkillMemoryContent,
    SkillMemoryKey,
    resolve_memory_key,
    validate_skill_memory,
)
from app.crypto_analysis.tools.registry import get_tool_registry
from app.llm.model import ainvoke_model, ainvoke_text, with_tools
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import trace_generation_call, trace_span
from prompts.crypto_analysis import (
    metric_agent_system_prompt,
    metric_execution_prompt,
    skill_output_prompt,
    thought_retrieval_prompt,
)

_LOG = get_logger("crypto_analysis.skill_worker")


# 技能结果聚合、内容生成与缓存


class SkillExecutionCandidate(BaseModel):
    """保存执行输入、原始响应和实际结果，预期结果由人工标注。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    input: dict[str, Any]
    replay: dict[str, Any]
    actual_output: dict[str, Any]
    duration_ms: float | None = Field(default=None, ge=0)


class SkillGenerationCandidate(BaseModel):
    """保存可回放的正文生成输入，保留请求范围及实际模型。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    input: dict[str, Any]
    scope: dict[str, str]
    generator_model: str
    actual_output: str


def _append_skill_candidate(candidate: BaseModel, path: str | Path) -> None:
    """整条序列化后一次追加，保留中文和已有候选。"""
    line = candidate.model_dump_json(ensure_ascii=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line)


def append_skill_execution_candidate(
    *,
    model_name: str,
    skill: LoadedSkill,
    scope: dict[str, str],
    evidence: dict[str, dict[str, Any]],
    result: dict[str, Any],
    duration_ms: float | None,
    path: str | Path,
) -> None:
    """保存取数响应与原始参数，沿用执行评测的数据源名称。"""
    source_responses = []
    for record in evidence.values():
        data_kind = record["data_kind"]
        source = "fred_observations" if data_kind in {"m2", "walcl"} else data_kind
        response = {"source": source, "arguments": record["arguments"]}
        if record["status"] == "ok":
            response["raw"] = record["raw"]
        else:
            response["error"] = record["reason"]
        source_responses.append(response)
    candidate = SkillExecutionCandidate(
        input={
            "metric_agent": {"model": model_name},
            "skill": skill.model_dump(mode="json"),
            "scope": dict(scope),
        },
        replay={"source_responses": source_responses},
        actual_output=result,
        duration_ms=duration_ms,
    )
    _append_skill_candidate(candidate, path)


def append_skill_generation_candidate(
    *,
    model_name: str,
    skill_source: str,
    skill: LoadedSkill,
    scope: dict[str, str],
    analysis_subject: str,
    result: dict[str, Any],
    content: str,
    duration_ms: float,
    path: str | Path,
) -> None:
    """保存固定指标、材料和实际正文，直接对应正文回放评测输入。"""
    candidate = SkillGenerationCandidate(
        input={
            "mode": "response_replay",
            "skill_source": skill_source,
            "skill_name": skill.name,
            "analysis_subject": analysis_subject,
            "user_input": scope["resolved_question"],
            "result": {
                k: v for k, v in result.items() if k not in {"thought_context", "scope"}
            },
            "thought_context": result["thought_context"],
            "fixed_response": content,
            "source_generation_duration_ms": duration_ms,
        },
        scope=dict(scope),
        generator_model=model_name,
        actual_output=content,
    )
    _append_skill_candidate(candidate, path)


def aggregate_skill_results(
    skill: LoadedSkill, indicators: list[dict[str, Any]]
) -> dict[str, Any]:
    """根据 Skill 权重聚合有效指标，不设置最低覆盖率。"""
    scoring = skill.scoring
    ids = [item["indicator_id"] for item in indicators]
    if len(ids) != len(set(ids)) or set(ids) != {
        metric.id for metric in scoring.metrics
    }:
        raise ValueError("技能指标记录不完整或重复")
    valid = {
        item["indicator_id"]: item for item in indicators if item["status"] == "ok"
    }
    coverage = fsum(metric.weight for metric in scoring.metrics if metric.id in valid)
    if coverage > 0:
        score = (
            sum(
                metric.weight * valid[metric.id]["score"]
                for metric in scoring.metrics
                if metric.id in valid
            )
            / coverage
        )
        status = "ok" if len(valid) == len(scoring.metrics) else "partial"
        reason = None
    else:
        score = None
        reason = "；".join(
            item["reason"]
            for item in indicators
            if item["status"] != "ok" and item.get("reason")
        )
        if not reason:
            reason = "没有可用的有效指标"
        if all(item["status"] == "not_applicable" for item in indicators):
            status = "not_applicable"
        elif any(item["status"] == "calculation_error" for item in indicators):
            status = "calculation_error"
        else:
            status = "data_missing"
    return {
        "skill_name": skill.name,
        "name": scoring.name,
        "status": status,
        "reason": reason,
        "score": score,
        "indicators": indicators,
    }


async def _retrieve_thought_context(
    retriever: KBRetriever, skill: LoadedSkill, target_asset: str
) -> dict[str, Any]:
    """检索技能理论材料，检索失败仍保留指标并记录明确的失败状态。"""
    scoring = skill.scoring
    query = thought_retrieval_prompt(
        target_asset,
        scoring.name,
        [metric.label for metric in scoring.metrics],
    )
    with trace_span("analysis.skill.thought_retrieval", skill_name=skill.name) as span:
        try:
            documents = await retriever.aretrieve(query)
        except Exception:  # noqa: BLE001 - 沿用原检索错误处理，不替换已计算指标。
            span.metadata["status"] = "error"
            _LOG.exception("思想材料检索失败：%s", skill.name)
            return {"query": query, "status": "error", "documents": []}
        # 空检索结果明确保留，不生成思想材料或替换已有指标结果。
        span.output = {"document_count": len(documents)}
        return {
            "query": query,
            "status": "ok" if documents else "no_results",
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
        }


async def generate_skill(
    llm: Any,
    skill_loader: SkillLoader,
    skill: LoadedSkill,
    result: dict[str, Any],
    analysis_subject: str,
) -> str:
    """按通用格式生成指标表格，思想原文保留给最终 writer 解读。"""
    started = time.perf_counter()
    indicators = [
        {**item, "field_labels": metric_field_labels(item["indicator_id"])}
        for item in result["indicators"]
        if item["status"] == "ok"
    ]
    # 没有成功指标时不生成表格，思想原文仍由 writer 使用。
    if not indicators:
        content = ""
    else:
        prompt = skill_output_prompt(
            analysis_subject=analysis_subject,
            skill_name=skill.name,
            indicators=indicators,
        )
        # 技能内容本身是 Markdown，直接接收正文，避免外层 JSON 的引号转义问题。
        content = await _invoke_skill_output(llm, prompt)
    if settings.skill_generation_capture_enabled:
        append_skill_generation_candidate(
            model_name=llm.model_name,
            skill_source=skill_loader.load_skill(skill.name),
            skill=skill,
            scope=result["scope"],
            analysis_subject=analysis_subject,
            result=result,
            content=content,
            duration_ms=(time.perf_counter() - started) * 1000,
            path=settings.skill_generation_capture_path,
        )
    return content


@trace_generation_call(name="crypto_skill_output", model=settings.fast_model)
async def _invoke_skill_output(llm: Any, prompt: str) -> str:
    """沿用技能文本校验与重试，用量统一由装饰器记录。"""
    # 表格生成使用执行模型，沿用模型工厂的思考配置。
    content = await ainvoke_text(llm, prompt)
    if not content:
        raise ValueError("技能内容不能为空")
    return content


def _build_skill_update(
    skill: LoadedSkill,
    key: SkillMemoryKey,
    result: dict[str, Any],
    content: str,
) -> SkillWorkerGraphState:
    """回传展示内容，沿用指标完整成功且检索正常的缓存写入条件。"""
    output: dict[str, Any] = {
        "skill_name": skill.name,
        "name": skill.scoring.name,
        "content": content,
        "thought_context": result["thought_context"],
        "cache_hit": False,
    }
    update: SkillWorkerGraphState = {"skill_result": output}
    if (
        not content.strip()
        or result["status"] != "ok"
        or result["thought_context"]["status"] not in {"ok", "no_results"}
    ):
        return update
    cached_content = SkillMemoryContent(
        content=validate_skill_memory(content, key),
        thought_context=result["thought_context"],
    )
    update["skill_memory"] = {
        key.cache_key: cached_content.model_dump_json(ensure_ascii=False)
    }
    return update


def skill_worker_step_limit(metric_count: int) -> int:
    """每个指标预留两轮计算与检查，另含缓存、生成、收尾及纠错预算。"""
    return metric_count * 4 + 7


# 技能执行子图


@trace_generation_call(
    name="crypto_metric_decision",
    model=settings.fast_model,
    input_arg="messages",
    input_format="messages",
    output_format="json",
)
async def _invoke_metric_decision(llm: Any, messages: list[BaseMessage]) -> AIMessage:
    """直接调用指标决策模型，仅对模型调用和返回类型校验进行重试。"""
    response = await ainvoke_model(llm, messages)
    if not isinstance(response, AIMessage):
        raise TypeError("指标决策模型未返回 AIMessage")
    if response.invalid_tool_calls:
        raise ValueError("指标决策模型返回了无法解析的工具调用")
    return response


def _select_thought_context(context: dict[str, Any]) -> dict[str, Any]:
    """保留重排分数最高的一条原文，同分时沿用第一条。"""
    documents = context["documents"]
    if len(documents) <= 1:
        return context
    return {
        **context,
        "documents": [max(documents, key=lambda doc: doc["metadata"]["rerank_score"])],
    }


def make_read_cache() -> Callable[[SkillWorkerGraphState], Awaitable[dict[str, Any]]]:
    """创建技能加载及结构化缓存读取节点。"""

    @timed_node
    async def read_cache(state: SkillWorkerGraphState) -> dict[str, Any]:
        with trace_span("crypto.skill.read_cache") as node_span:
            try:
                name = state["skill_name"]
                loader = SkillLoader.from_snapshot({name: state["skill_source"]})
                skill = loader.load_definition(name)
                key = resolve_memory_key(state["scope"]["target_asset"], name)
                cached = state["cached_memory"]
                node_span.metadata["cache_hit"] = cached is not None
                if cached is not None:
                    try:
                        content = SkillMemoryContent.model_validate_json(
                            validate_skill_memory(cached, key)
                        )
                    except ValidationError as exc:
                        raise ValueError(
                            "技能缓存缺少结构化的表格和思想原文，"
                            "请明确要求不用缓存重新分析"
                        ) from exc
                    return {
                        "skill": skill,
                        "skill_result": {
                            "skill_name": skill.name,
                            "name": skill.scoring.name,
                            "content": content.content,
                            "thought_context": _select_thought_context(
                                content.thought_context
                            ),
                            "cache_hit": True,
                        },
                    }
                return {
                    "skill": skill,
                    "messages": [
                        HumanMessage(
                            content=metric_execution_prompt(skill.model_dump())
                        )
                    ],
                    "evidence": {},
                    "indicator_results": {},
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return read_cache


def route_cache(state: SkillWorkerGraphState) -> Literal["hit", "miss"]:
    """缓存有效时直接收尾，否则进入指标执行。"""
    return "hit" if state["cached_memory"] is not None else "miss"


def make_calculate_metrics(fast_llm: Any) -> Callable[..., Awaitable[list[Any]]]:
    """创建单轮模型决策及工具执行节点，循环交由子图控制。"""

    @timed_node
    async def calculate_metrics(
        state: SkillWorkerGraphState, config: RunnableConfig, runtime: Runtime
    ) -> list[Any]:
        with trace_span("crypto.skill.calculate_metrics") as node_span:
            try:
                registry = get_tool_registry()
                tools = [registry[name] for name in state["skill"].tools]
                response = await _invoke_metric_decision(
                    with_tools(fast_llm, tools),
                    [
                        SystemMessage(content=metric_agent_system_prompt()),
                        *state["messages"],
                    ],
                )
                updates: list[Any] = [Command(update={"messages": [response]})]
                if response.tool_calls:
                    # 工具读取完整状态；当前技能上下文由程序注入，不交给模型生成。
                    tool_state = {
                        **state,
                        "messages": add_messages(state["messages"], [response]),
                    }
                    tool_runtime = replace(
                        runtime,
                        context={"skill": state["skill"], "scope": state["scope"]},
                    )
                    tool_updates = await ToolNode(tools).ainvoke(
                        tool_state, config=config, runtime=tool_runtime
                    )
                    # 手动调用工具执行器后，必须向图返回更新以应用原 reducer。
                    if isinstance(tool_updates, dict):
                        updates.append(tool_updates)
                    else:
                        updates.extend(tool_updates)
                return updates
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return calculate_metrics


def make_check_metrics() -> Callable[
    [SkillWorkerGraphState], Awaitable[dict[str, Any]]
]:
    """创建指标完整性检查节点，只依据工具实际写入的结果。"""

    @timed_node
    async def check_metrics(state: SkillWorkerGraphState) -> dict[str, Any]:
        with trace_span("crypto.skill.check_metrics") as node_span:
            skill = state["skill"]
            records = state["indicator_results"]
            missing = [
                metric.id
                for metric in skill.scoring.metrics
                if metric.id not in records
            ]
            node_span.output = {"missing_metric_ids": missing}
            if missing:
                # 复用原任务模板，仅将任务清单限定为缺失项，不新增提示词。
                pending = skill.model_dump()
                pending["scoring"]["metrics"] = [
                    metric
                    for metric in pending["scoring"]["metrics"]
                    if metric["id"] in missing
                ]
                return {
                    "messages": [
                        HumanMessage(content=metric_execution_prompt(pending))
                    ],
                }
            result = aggregate_skill_results(
                skill, [records[metric.id] for metric in skill.scoring.metrics]
            )
            if settings.skill_execution_capture_enabled:
                # 拆图后仅保留节点耗时，不为执行候选补造整段指标耗时。
                append_skill_execution_candidate(
                    model_name=settings.fast_model,
                    skill=skill,
                    scope=state["scope"],
                    evidence=state["evidence"],
                    result=result,
                    duration_ms=None,
                    path=settings.skill_execution_capture_path,
                )
            return {"metric_result": result}

    return check_metrics


def route_metrics(
    state: SkillWorkerGraphState,
) -> Literal["incomplete", "complete"]:
    """缺少指标记录时继续执行，失败记录仍属于已处理结果。"""
    complete = all(
        metric.id in state["indicator_results"]
        for metric in state["skill"].scoring.metrics
    )
    return "complete" if complete else "incomplete"


def make_generate_content(
    fast_llm: Any, retriever: KBRetriever
) -> Callable[[SkillWorkerGraphState], Awaitable[dict[str, Any]]]:
    """创建思想检索及指标表格生成节点。"""

    @timed_node
    async def generate_content(state: SkillWorkerGraphState) -> dict[str, Any]:
        with trace_span("crypto.skill.generate_content") as node_span:
            try:
                skill = state["skill"]
                key = resolve_memory_key(state["scope"]["target_asset"], skill.name)
                analysis_subject = (
                    "加密市场"
                    if key.asset_scope == "GLOBAL"
                    else state["scope"]["target_asset"]
                )
                thought_context = _select_thought_context(
                    await _retrieve_thought_context(retriever, skill, analysis_subject)
                )
                result = {
                    **state["metric_result"],
                    "thought_context": thought_context,
                    "scope": dict(state["scope"]),
                }
                loader = SkillLoader.from_snapshot(
                    {state["skill_name"]: state["skill_source"]}
                )
                content = await generate_skill(
                    fast_llm, loader, skill, result, analysis_subject
                )
                return {
                    "skill_result": {
                        "skill_name": skill.name,
                        "name": skill.scoring.name,
                        "content": content,
                        "thought_context": thought_context,
                        "cache_hit": False,
                    }
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return generate_content


def make_finalize_skill() -> Callable[
    [SkillWorkerGraphState], Awaitable[dict[str, Any]]
]:
    """创建技能收尾节点，只回传当前技能的结果及缓存增量。"""

    @timed_node
    async def finalize_skill(state: SkillWorkerGraphState) -> dict[str, Any]:
        skill = state["skill"]
        if state["cached_memory"] is not None:
            return {"skill_result": state["skill_result"]}
        output = state["skill_result"]
        result = {
            **state["metric_result"],
            "thought_context": output["thought_context"],
        }
        key = resolve_memory_key(state["scope"]["target_asset"], skill.name)
        return _build_skill_update(skill, key, result, output["content"])

    return finalize_skill


def build_crypto_analysis_sub_agent(
    fast_llm: Any, retriever: KBRetriever
) -> CompiledStateGraph:
    """组装缓存分支、指标执行循环和技能内容生成子图。"""
    graph = StateGraph(SkillWorkerGraphState)
    graph.add_node("read_cache", make_read_cache())
    graph.add_node("calculate_metrics", make_calculate_metrics(fast_llm))
    graph.add_node("check_metrics", make_check_metrics())
    graph.add_node("generate_content", make_generate_content(fast_llm, retriever))
    graph.add_node("finalize_skill", make_finalize_skill())
    graph.add_edge(START, "read_cache")
    graph.add_conditional_edges(
        "read_cache",
        route_cache,
        {"hit": "finalize_skill", "miss": "calculate_metrics"},
    )
    graph.add_edge("calculate_metrics", "check_metrics")
    # 反应式Agent的工具调用循环
    graph.add_conditional_edges(
        "check_metrics",
        route_metrics,
        {"incomplete": "calculate_metrics", "complete": "generate_content"},
    )
    graph.add_edge("generate_content", "finalize_skill")
    graph.add_edge("finalize_skill", END)
    return graph.compile(name="crypto_skill_subgraph")
