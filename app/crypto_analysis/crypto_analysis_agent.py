"""加密货币分析：请求规划、技能执行、并行汇总及报告生成。"""

import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from math import fsum
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite, Send
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_state import (
    AnalysisReport,
    DynamicAnalysisState,
    HistorySummaryOutput,
    MetricAgentState,
    MetricToolContext,
    SkillOutput,
    SkillResult,
    SkillWorkerState,
)
from app.crypto_analysis.metric_service import missing_metric
from app.crypto_analysis.skill_loader import (
    LoadedSkill,
    SkillLoader,
    get_skill_loader,
)
from app.crypto_analysis.skill_memory import (
    SkillMemoryKey,
    read_skill_memory,
    resolve_memory_key,
    validate_skill_memory,
)
from app.crypto_analysis.tools.registry import get_tool_registry
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import estimate_tokens, trace_generation, trace_span
from prompts.crypto_analysis import (
    analysis_plan_prompt,
    analysis_report_prompt,
    history_summary_prompt,
    metric_agent_system_prompt,
    metric_execution_prompt,
    skill_output_prompt,
    thought_retrieval_prompt,
)

_LOG = get_logger("crypto_analysis.dynamic_analysis")


async def _invoke_structured_output(
    llm: Any,
    schema: type[BaseModel],
    prompt: str,
    trace_name: str,
    model_name: str,
) -> BaseModel:
    """通过函数调用提交结构化结果，保留类型校验、流过滤标记和用量追踪。"""
    model = llm.with_structured_output(schema, method="function_calling")
    with trace_generation(trace_name, model=model_name) as generation:
        response = await model.ainvoke(
            prompt, config={"tags": ["crypto_structured_output"]}
        )
        if not isinstance(response, schema):
            raise TypeError(f"模型未返回 {schema.__name__} 结构化结果")
        generation.usage = {
            "input": estimate_tokens(prompt),
            "output": estimate_tokens(response.model_dump_json()),
        }
        return response


# 请求规划模型


class AnalysisPlan(BaseModel):
    """一轮分析的目标币种和所需技能，不指定具体工具。"""

    model_config = ConfigDict(extra="forbid")
    target_asset: str = Field(pattern=r"^[A-Z0-9]+$", strict=True)
    skill_names: list[str]

    @field_validator("skill_names")
    @classmethod
    def unique_skill_names(cls, names: list[str]) -> list[str]:
        """拒绝重复技能，避免并行派发同一技能多次。"""
        if len(names) != len(set(names)):
            raise ValueError("分析计划中的技能名称不能重复")
        return names


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
        settings.fast_model,
    )
    return plan.model_dump()


# 指标工具 Agent


class MetricCompletionMiddleware(AgentMiddleware):
    """所有指标已有可核验工具结果时结束，避免模型重复调用工具。"""

    async def awrap_model_call(
        self,
        request: ModelRequest[MetricToolContext],
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        skill = request.runtime.context["skill"]
        records = request.state["indicator_results"]
        if all(metric.id in records for metric in skill.scoring.metrics):
            # 空消息不生成报告、不修改指标，只让工具图进入正常结束节点。
            return ModelResponse(result=[AIMessage(content="")])
        return await handler(request)


class MetricModelTrace(AgentMiddleware):
    """把每轮工具决策模型调用接入项目已有的上下文追踪。"""

    def __init__(self, model_name: str = settings.fast_model) -> None:
        self.model_name = model_name

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        with trace_generation(
            "crypto_metric_decision", model=self.model_name
        ) as generation:
            response = await handler(request)
            messages = [*request.messages]
            if request.system_message is not None:
                messages.insert(0, request.system_message)
            generation.usage = {
                "input": estimate_tokens(
                    json.dumps(
                        [message.model_dump(mode="json") for message in messages],
                        ensure_ascii=False,
                    )
                ),
                "output": estimate_tokens(
                    json.dumps(
                        [
                            message.model_dump(mode="json")
                            for message in response.result
                        ],
                        ensure_ascii=False,
                    )
                ),
            }
            return response


def build_metric_agent(llm: Any, skill: LoadedSkill) -> CompiledStateGraph:
    """只注册当前 Skill 配置声明的工具。"""
    tool_registry = get_tool_registry()
    tools = [tool_registry[name] for name in skill.tools]
    return create_agent(
        model=llm,
        tools=tools,
        # 声明 evidence 与 indicator_results，工具才能写入原始数据和指标结果。
        state_schema=MetricAgentState,
        # 声明 skill 与 scope，工具才能读取币种，且模型不能修改。
        context_schema=MetricToolContext,
        system_prompt=metric_agent_system_prompt(),
        middleware=[MetricCompletionMiddleware(), MetricModelTrace()],
    )


# 技能聚合与执行


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


def metric_agent_step_limit(metric_count: int) -> int:
    """按指标数量给出工具图步数上限。

    图的每一步是一次模型决策或一次工具执行。每个指标按取数、计算
    各占一轮，且一轮只调用一个工具；全部完成后还需要一步模型收尾，
    并额外保留一轮用于纠正一次错误调用，以及一步供图确认结束。
    """
    return metric_count * 4 + 4


async def execute_skill(
    metric_agent: Any,
    skill: LoadedSkill,
    scope: dict[str, str],
) -> dict[str, Any]:
    """让技能 Agent 自主调用工具，只接收工具写入的可核验指标结果。"""
    scoring = skill.scoring
    with trace_span("analysis.skill.execute", skill_name=skill.name) as node_span:
        try:
            with trace_span("analysis.metric_agent.graph", skill_name=skill.name):
                state = await metric_agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=metric_execution_prompt(skill.model_dump())
                            )
                        ],
                        "evidence": {},
                        "indicator_results": {},
                    },
                    context={"skill": skill, "scope": dict(scope)},
                    config={
                        "recursion_limit": metric_agent_step_limit(len(scoring.metrics))
                    },
                )
            # 模型最终生成的文本不作为指标分数，遗漏项明确记为未取得证据。
            records = state["indicator_results"]
            indicators = [
                records[metric.id]
                if metric.id in records
                else missing_metric(
                    skill.name,
                    metric.id,
                    "data_missing",
                    "技能 Agent 未取得该指标的可核验工具结果",
                )
                for metric in scoring.metrics
            ]
            result = aggregate_skill_results(skill, indicators)
            return result
        except Exception:
            node_span.metadata["status"] = "error"
            raise


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


async def _generate_skill_content(
    llm: Any,
    skill_loader: SkillLoader,
    skill: LoadedSkill,
    result: dict[str, Any],
    analysis_subject: str,
) -> str:
    """按技能模板生成表格与思想解读，只使用成功指标和实际检索材料。"""
    indicators = [item for item in result["indicators"] if item["status"] == "ok"]
    thought_context = result["thought_context"]
    # 全部缺失时返回空展示，不调用模型；只有思想材料时仍可解读已有原则。
    if not indicators and not thought_context["documents"]:
        return ""
    prompt = skill_output_prompt(
        analysis_subject=analysis_subject,
        skill_name=skill.name,
        indicators=indicators,
        thought_context=thought_context,
        output_template=skill_loader.load_analysis_output(skill.name),
    )
    # 技能内容本身是 Markdown，直接接收正文，避免外层 JSON 的引号转义问题。
    with trace_generation(
        "crypto_skill_output", model=settings.smart_model
    ) as generation:
        response = await llm.ainvoke(prompt, config={"tags": ["crypto_skill_output"]})
        output = SkillOutput(content=response.content)
        generation.usage = {
            "input": estimate_tokens(prompt),
            "output": estimate_tokens(output.content),
        }
        return output.content


def _build_skill_update(
    skill: LoadedSkill,
    key: SkillMemoryKey,
    result: dict[str, Any],
    content: str,
) -> DynamicAnalysisState:
    """回传展示内容，沿用指标完整成功且检索正常的缓存写入条件。"""
    output: SkillResult = {
        "skill_name": skill.name,
        "name": skill.scoring.name,
        "content": content,
        "cache_hit": False,
    }
    update: DynamicAnalysisState = {"skill_results": [output]}
    if (
        not content.strip()
        or result["status"] != "ok"
        or result["thought_context"]["status"] not in {"ok", "no_results"}
    ):
        return update
    update["skill_memory"] = {key.cache_key: validate_skill_memory(content, key)}
    return update


# 分析主图：规划、并行技能、报告


def build_dynamic_analysis_agent(
    smart_llm: Any, fast_llm: Any, retriever: KBRetriever
) -> CompiledStateGraph:
    @timed_node
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:
        with trace_span("crypto.plan") as node_span:
            try:
                skill_loader = get_skill_loader()
                question = state["scope"].get("resolved_question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("技能分析缺少有效用户问题")
                selection = await create_analysis_plan(
                    smart_llm,
                    question,
                    skill_loader.format_skill_descriptions(),
                )
                scope = {
                    **state["scope"],
                    "target_asset": selection["target_asset"],
                }
                history_summary = state.get("history_summary", "")
                if not isinstance(history_summary, str) or len(history_summary) > 2000:
                    raise ValueError("历史摘要必须是不超过 2000 字符的字符串")
                return {
                    "history_summary": history_summary,
                    "report": "",
                    "skill_sources": skill_loader.snapshot(),
                    "plan": selection,
                    "scope": scope,
                    # 缓存跨轮保留，追加型报告输入每轮从空列表开始。
                    "skill_results": Overwrite([]),
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

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
                    "cached_memory": read_skill_memory(
                        state.get("skill_memory", {}),
                        resolve_memory_key(state["scope"]["target_asset"], name),
                    ),
                },
            )
            for name in names
        ]

    @timed_node(label=lambda state: state["skill_name"])
    async def skill_worker(state: SkillWorkerState) -> DynamicAnalysisState:
        """命中时直接返回完整技能内容，未命中时执行指标、检索并生成内容。"""
        with trace_span(
            "crypto.skill_worker", skill_name=state["skill_name"]
        ) as node_span:
            try:
                # 根据本轮派发的快照加载技能，保持定义与规划时一致。
                skill_loader = SkillLoader.from_snapshot(
                    {state["skill_name"]: state["skill_source"]}
                )
                skill = skill_loader.load_definition(state["skill_name"])
                # 全局技能共用 GLOBAL 范围，其他技能按本轮币种区分缓存。
                key = resolve_memory_key(
                    state["scope"]["target_asset"], state["skill_name"]
                )
                cached = state["cached_memory"]
                # 只记录是否找到缓存；缓存内容是否有效由后续校验判断。
                node_span.metadata["cache_hit"] = cached is not None
                if cached is not None:
                    content = validate_skill_memory(cached, key)
                    return {
                        "skill_results": [
                            {
                                "skill_name": skill.name,
                                "name": skill.scoring.name,
                                "content": content,
                                "cache_hit": True,
                            }
                        ]
                    }
                analysis_subject = (
                    "加密市场"
                    if key.asset_scope == "GLOBAL"
                    else state["scope"]["target_asset"]
                )
                # 未命中时沿用原指标 Agent，执行取数、计算和结果聚合。
                metric_agent = build_metric_agent(fast_llm, skill)
                result = await execute_skill(metric_agent, skill, state["scope"])
                # 检索与内容生成使用同一对象，避免全局缓存混入特定币种判断。
                result["thought_context"] = await _retrieve_thought_context(
                    retriever, skill, analysis_subject
                )
                content = await _generate_skill_content(
                    smart_llm, skill_loader, skill, result, analysis_subject
                )
                # 只回传当前技能的结果与缓存增量，由主图 reducer 合并并行更新。
                return _build_skill_update(skill, key, result, content)
            except Exception:
                # 标记当前节点失败并继续抛出，不将异常当作缓存未命中。
                node_span.metadata["status"] = "error"
                raise

    @timed_node
    async def writer(state: DynamicAnalysisState) -> dict[str, Any]:
        """依据技能内容与历史摘要撰写结论，报告只包含结论正文。"""
        with trace_span("crypto.writer") as node_span:
            try:
                if not state["plan"]["skill_names"]:
                    report = "当前任务没有适用的分析技能；请说明希望分析的资产或问题。"
                    return {
                        "report": report,
                        "messages": [AIMessage(content=report)],
                    }
                ordered = sorted(
                    state["skill_results"], key=lambda item: item["skill_name"]
                )
                # 只采用本轮技能内容，历史摘要不能替代本轮证据。
                prompt = analysis_report_prompt(
                    target_asset=state["scope"]["target_asset"],
                    original_question=state["scope"]["resolved_question"],
                    skill_results=ordered,
                    history_summary=state["history_summary"],
                )
                output = await _invoke_structured_output(
                    smart_llm,
                    AnalysisReport,
                    prompt,
                    "crypto_analysis_writer",
                    settings.smart_model,
                )
                report = output.report
                return {
                    "report": report,
                    "messages": [AIMessage(content=report)],
                }
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    @timed_node
    async def summarize_history(state: DynamicAnalysisState) -> dict[str, str]:
        """在结论完成后更新摘要，不将全部表格重复写入历史。"""
        # 无适用技能时报告为提示语，不纳入历史分析摘要。
        if not state["plan"]["skill_names"]:
            return {}
        prompt = history_summary_prompt(
            previous_summary=state["history_summary"],
            target_asset=state["scope"]["target_asset"],
            original_question=state["scope"]["resolved_question"],
            analysis_time=state["scope"]["as_of"],
            report=state["report"],
        )
        output = await _invoke_structured_output(
            fast_llm,
            HistorySummaryOutput,
            prompt,
            "crypto_history_summary",
            settings.fast_model,
        )
        return {"history_summary": output.history_summary}

    builder = StateGraph(DynamicAnalysisState)
    builder.add_node("plan", plan)
    builder.add_node("skill_worker", skill_worker)
    builder.add_node("writer", writer)
    builder.add_node("summarize_history", summarize_history)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", send_skills)
    builder.add_edge("skill_worker", "writer")
    builder.add_edge("writer", "summarize_history")
    builder.add_edge("summarize_history", END)
    return builder.compile(name="crypto_analysis_agent")
