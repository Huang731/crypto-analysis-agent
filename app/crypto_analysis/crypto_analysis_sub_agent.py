"""加密资产技能执行子图：缓存、指标执行、内容生成及结果收尾。"""

from collections.abc import Awaitable, Callable
from traceback import format_exc
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import (
    BaseMessage,
    messages_to_dict,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.candidate_capture import append_node_candidate
from app.core.config import settings
from app.core.observability import get_logger, log_event, timed_node
from app.core.tracing import trace_model_call
from app.crypto_analysis.crypto_analysis_state import (
    CryptoAnalysisSubGraphState,
    MetricAgentContext,
    MetricAgentState,
)
from app.crypto_analysis.metric_renderer import render_metric_tables
from app.crypto_analysis.skill_loader import LoadedSkill, get_skill_loader
from app.crypto_analysis.skill_memory import (
    SkillMemoryContent,
    resolve_memory_key,
    validate_skill_memory,
)
from app.crypto_analysis.tools.registry import get_tool_registry
from app.llm.model import ainvoke_model, make_glm_4_flash_llm
from app.rag.retriever import KBRetriever
from prompts.crypto_analysis_calculate_metrics import (
    calculate_metrics_messages,
)
from prompts.crypto_analysis_skill_content import skill_retrieval_messages

# 技能结果聚合、内容生成与缓存


def aggregate_skill_results(
    skill: LoadedSkill, indicators: list[dict[str, Any]]
) -> dict[str, Any]:
    """汇总已有指标及执行状态，不要求指标齐全或最低覆盖率。"""
    scoring = skill.scoring
    ids = [item["indicator_id"] for item in indicators]
    if len(ids) != len(set(ids)) or not set(ids).issubset(
        {metric.id for metric in scoring.metrics}
    ):
        raise ValueError("技能指标记录重复或不属于当前技能")
    valid = {
        item["indicator_id"]: item for item in indicators if item["status"] == "ok"
    }
    if valid:
        status = "ok" if len(valid) == len(scoring.metrics) else "partial"
        reason = None
    else:
        reason = "；".join(
            item["reason"]
            for item in indicators
            if item["status"] != "ok" and item.get("reason")
        )
        if not reason:
            reason = "没有可用的有效指标"
        if indicators and all(
            item["status"] == "not_applicable" for item in indicators
        ):
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
        "indicators": indicators,
    }


@trace_model_call(name="analysis.skill.thought_retrieval")
async def _retrieve_thought_context(
    retriever: KBRetriever, query: str, *, trace: Any
) -> dict[str, Any]:
    """用生成的检索问题查找理论材料，检索失败仍保留指标与失败状态。"""
    trace.metadata.update(query=query)
    try:
        documents = await retriever.aretrieve(query)
    except Exception:  # noqa: BLE001 - 沿用原检索错误处理，不替换已计算指标。
        trace.metadata["status"] = "error"
        log_event(
            get_logger("crypto_analysis.skill_worker"),
            "思想材料检索失败",
            exception=format_exc(),
            query=query,
        )
        result = {"query": query, "status": "error", "documents": []}
    else:
        # 空检索结果明确保留，不生成思想材料或替换已有指标结果。
        trace.output = {"document_count": len(documents)}
        result = {
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
    if settings.thought_retrieval_capture_enabled:
        append_node_candidate(
            node="thought_retrieval",
            input=query,
            actual_output=result,
            path=settings.thought_retrieval_capture_path,
        )
    return result


@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_skill_retrieval_query(
    llm: Any,
    indicators: list[dict[str, Any]],
    *,
    analysis_subject: str,
    skill_name: str,
) -> str:
    """在调用内部组装消息，只生成检索问题并保留有限重试。"""
    messages = skill_retrieval_messages(
        analysis_subject=analysis_subject,
        skill_name=skill_name,
        indicators=indicators,
    )

    @trace_model_call(
        name="crypto_skill_retrieval_query", model=settings.glm_4_flash_model
    )
    async def invoke(messages: list[BaseMessage]) -> str:
        """追踪实际发送的完整消息，避免遗漏提示词用量。"""
        response = await ainvoke_model(
            llm,
            messages,
        )
        query = response.content.strip()
        if not query:
            raise ValueError("检索问题不能为空")
        return query

    return await invoke(messages)


# 技能执行子图


def _select_thought_context(context: dict[str, Any]) -> dict[str, Any]:
    """保留重排分数最高的一条原文，同分时沿用第一条。"""
    documents = context["documents"]
    if len(documents) <= 1:
        return context
    return {
        **context,
        "documents": [max(documents, key=lambda doc: doc["metadata"]["rerank_score"])],
    }


def make_read_cache() -> Callable[
    [CryptoAnalysisSubGraphState], Awaitable[dict[str, Any]]
]:
    """创建技能加载及结构化缓存读取节点。"""

    async def read_cache(state: CryptoAnalysisSubGraphState) -> dict[str, Any]:
        name = state["skill_name"]
        loader = get_skill_loader()
        skill = loader.load_definition(name)
        key = resolve_memory_key(state["scope"]["target_asset"], name)
        cached = state["cached_memory"]
        if cached is not None:
            content = SkillMemoryContent.model_validate_json(
                validate_skill_memory(cached, key)
            )
            update = {
                "skill": skill,
                "skill_result": {
                    "skill_name": skill.name,
                    "name": skill.scoring.name,
                    "content": content.content,
                    "thought_context": _select_thought_context(content.thought_context),
                    "cache_hit": True,
                },
            }
        else:
            update = {
                "skill": skill,
                "messages": [],
                "evidence": {},
                "indicator_results": {},
            }
        return update

    return read_cache


def route_cache(state: CryptoAnalysisSubGraphState) -> Literal["hit", "miss"]:
    """缓存有效时直接收尾，否则进入指标执行。"""
    return "hit" if state["cached_memory"] is not None else "miss"


@trace_model_call(name="crypto.metric_agent")
async def _invoke_metric_agent(
    agent: CompiledStateGraph, state: CryptoAnalysisSubGraphState
) -> MetricAgentState:
    """追踪指标 Agent 整体执行，模型用量由内部中间件统计。"""
    pending = state["skill"].model_dump()
    # 复用原任务模板，仅将任务清单限定为缺失项，不新增提示词。
    pending["scoring"]["metrics"] = [
        metric
        for metric in pending["scoring"]["metrics"]
        if metric["id"] not in state["indicator_results"]
    ]
    system_message, task_message = calculate_metrics_messages(pending)
    messages = list(state["messages"])
    if not messages:
        messages.append(system_message)
    messages.append(task_message)
    return await agent.ainvoke(
        {
            "messages": messages,
            "evidence": state["evidence"],
            "indicator_results": state["indicator_results"],
        },
        context={"skill": state["skill"], "scope": state["scope"]},
    )


def make_calculate_metrics() -> Callable[..., Awaitable[dict[str, Any]]]:
    """使用指标 Agent 执行工具，再将已有结果交给内容节点。"""
    glm_4_flash_llm = make_glm_4_flash_llm()

    @timed_node(label=lambda state: state["skill_name"])
    async def calculate_metrics(state: CryptoAnalysisSubGraphState) -> dict[str, Any]:
        registry = get_tool_registry()
        # 工具执行后直接退出内部 Agent，由内容节点处理已有指标。
        tools = [
            registry[name].model_copy(update={"return_direct": True})
            for name in state["skill"].tools
        ]
        agent = create_agent(
            model=glm_4_flash_llm,
            tools=tools,
            state_schema=MetricAgentState,
            context_schema=MetricAgentContext,
            middleware=[
                ToolCallLimitMiddleware(
                    run_limit=settings.max_tool_calls_per_skill,
                    exit_behavior="continue",
                )
            ],
            name="crypto_metric_agent",
        )
        result = await _invoke_metric_agent(agent, state)
        if settings.calculate_metrics_capture_enabled:
            append_node_candidate(
                node="calculate_metrics",
                input={
                    "skill": state["skill"].model_dump(),
                    "scope": state["scope"],
                    "messages": messages_to_dict(state["messages"]),
                    "evidence": state["evidence"],
                    "indicator_results": state["indicator_results"],
                },
                actual_output={
                    "messages": messages_to_dict(result["messages"]),
                    "evidence": result["evidence"],
                    "indicator_results": result["indicator_results"],
                },
                path=settings.calculate_metrics_capture_path,
            )
        # 外层状态已有输入历史，只回传本轮新增消息，避免重复合并旧消息。
        new_messages = result["messages"][len(state["messages"]) :]
        return {
            "messages": new_messages,
            "evidence": result["evidence"],
            "indicator_results": result["indicator_results"],
        }

    return calculate_metrics


def make_skill_content(
    retriever: KBRetriever,
) -> Callable[[CryptoAnalysisSubGraphState], Awaitable[dict[str, Any]]]:
    """使用已有指标渲染表格，再根据成功指标检索思想材料。"""
    glm_4_flash_llm = make_glm_4_flash_llm()

    @timed_node(label=lambda state: state["skill_name"])
    async def skill_content(state: CryptoAnalysisSubGraphState) -> dict[str, Any]:
        skill = state["skill"]
        records = state["indicator_results"]
        metric_result = aggregate_skill_results(
            skill,
            [
                records[metric.id]
                for metric in skill.scoring.metrics
                if metric.id in records
            ],
        )
        key = resolve_memory_key(state["scope"]["target_asset"], skill.name)
        analysis_subject = (
            "加密市场"
            if key.asset_scope == "GLOBAL"
            else state["scope"]["target_asset"]
        )
        indicators = metric_result["indicators"]
        content = render_metric_tables(skill, indicators)
        successful_indicators = [item for item in indicators if item["status"] == "ok"]
        retrieval_query = ""
        if successful_indicators:
            retrieval_query = await _invoke_skill_retrieval_query(
                glm_4_flash_llm,
                successful_indicators,
                analysis_subject=analysis_subject,
                skill_name=skill.name,
            )
            thought_context = _select_thought_context(
                await _retrieve_thought_context(retriever, retrieval_query)
            )
        else:
            thought_context = {"query": "", "status": "no_results", "documents": []}
        update = {
            "metric_result": metric_result,
            "skill_result": {
                "skill_name": skill.name,
                "name": skill.scoring.name,
                "content": content,
                "thought_context": thought_context,
                "cache_hit": False,
            },
        }
        if settings.skill_content_capture_enabled:
            append_node_candidate(
                node="skill_content",
                input={
                    "analysis_subject": analysis_subject,
                    "skill_name": skill.name,
                    "indicators": successful_indicators,
                },
                actual_output={"content": content, "retrieval_query": retrieval_query},
                path=settings.skill_content_capture_path,
            )
        return update

    return skill_content


def make_finalize_skill() -> Callable[
    [CryptoAnalysisSubGraphState], Awaitable[dict[str, Any]]
]:
    """创建技能收尾节点，只回传当前技能的结果及缓存增量。"""

    async def finalize_skill(state: CryptoAnalysisSubGraphState) -> dict[str, Any]:
        output = state["skill_result"]
        update: dict[str, Any] = {"skill_result": output}
        if state["cached_memory"] is not None:
            return update
        thought_context = output["thought_context"]
        if state["metric_result"]["status"] != "ok" or thought_context[
            "status"
        ] not in {"ok", "no_results"}:
            return update
        key = resolve_memory_key(state["scope"]["target_asset"], state["skill"].name)
        cached_content = SkillMemoryContent(
            content=output["content"],
            thought_context=thought_context,
        )
        update["skill_memory"] = {
            key.cache_key: cached_content.model_dump_json(ensure_ascii=False)
        }
        return update

    return finalize_skill


def build_crypto_analysis_sub_agent(
    retriever: KBRetriever,
) -> CompiledStateGraph:
    """组装缓存分支、指标执行和技能内容生成子图。"""
    graph = StateGraph(CryptoAnalysisSubGraphState)
    graph.add_node("read_cache", make_read_cache())
    graph.add_node("calculate_metrics", make_calculate_metrics())
    graph.add_node("skill_content", make_skill_content(retriever))
    graph.add_node("finalize_skill", make_finalize_skill())
    graph.add_edge(START, "read_cache")
    graph.add_conditional_edges(
        "read_cache",
        route_cache,
        {"hit": "finalize_skill", "miss": "calculate_metrics"},
    )
    graph.add_edge("calculate_metrics", "skill_content")
    graph.add_edge("skill_content", "finalize_skill")
    graph.add_edge("finalize_skill", END)
    return graph.compile(name="crypto_skill_subgraph")
