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
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.config import settings
from app.crypto_analysis.crypto_analysis_state import (
    DynamicAnalysisState,
    MetricAgentState,
    MetricToolContext,
    SkillWorkerState,
)
from app.crypto_analysis.metric_service import missing_metric
from app.crypto_analysis.skill_loader import (
    LoadedSkill,
    SkillLoader,
    get_skill_loader,
)
from app.crypto_analysis.tools.registry import get_tool_registry
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import estimate_tokens, trace_generation, trace_span
from prompts.crypto_analysis import (
    analysis_plan_prompt,
    analysis_report_prompt,
    metric_agent_system_prompt,
    metric_execution_prompt,
    thought_retrieval_prompt,
)

_LOG = get_logger("crypto_analysis.dynamic_analysis")


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
    """在请求入口固定 UTC 截止时间和追踪标识。"""
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
    prompt = analysis_plan_prompt(
        json.dumps(AnalysisPlan.model_json_schema(), ensure_ascii=False),
        skill_descriptions,
        question,
    )
    # 规划只提取字段，使用 JSON 模式，避免把普通文本输出误当作工具调用结果。
    structured_llm = llm.with_structured_output(AnalysisPlan, method="json_mode")
    with trace_generation(
        "crypto_analysis_plan", model=settings.fast_model
    ) as generation:
        plan = await structured_llm.ainvoke(prompt)
        if isinstance(plan, AnalysisPlan):
            generation.usage = {
                "input": estimate_tokens(prompt),
                "output": estimate_tokens(plan.model_dump_json()),
            }
    if not isinstance(plan, AnalysisPlan):
        raise TypeError("规划模型未返回 AnalysisPlan 结构化结果")
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
        # 声明 skill 与 scope，工具才能读取币种、截止时间，且模型不能修改。
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


# 分析主图：规划、并行技能、报告


def build_dynamic_analysis_agent(
    smart_llm: Any, fast_llm: Any, retriever: KBRetriever
) -> CompiledStateGraph:
    @timed_node
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:
        with trace_span("crypto.plan") as node_span:
            try:
                skill_loader = get_skill_loader()
                question = state["scope"].get("original_question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("技能分析缺少有效用户问题")
                selection = await create_analysis_plan(
                    fast_llm,
                    question,
                    skill_loader.format_skill_descriptions(),
                )
                unknown_names = set(selection["skill_names"]) - {
                    meta.name for meta in skill_loader.list_skills()
                }
                if unknown_names:
                    raise ValueError(
                        f"分析计划包含未知技能：{'、'.join(sorted(unknown_names))}"
                    )
                scope = {
                    **state["scope"],
                    "target_asset": selection["target_asset"],
                    "original_question": question,
                }
                return {
                    "skill_sources": skill_loader.snapshot(),
                    "plan": selection,
                    "scope": scope,
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
                },
            )
            for name in names
        ]

    @timed_node(label=lambda state: state["skill_name"])
    async def skill_worker(state: SkillWorkerState) -> DynamicAnalysisState:
        """执行当前技能并附加理论材料，指标只保存在技能结果中。"""
        with trace_span(
            "crypto.skill_worker", skill_name=state["skill_name"]
        ) as node_span:
            try:
                skill_loader = SkillLoader.from_snapshot(
                    {state["skill_name"]: state["skill_source"]}
                )
                skill = skill_loader.load_definition(state["skill_name"])
                scoring = skill.scoring
                metric_agent = build_metric_agent(fast_llm, skill)
                result = await execute_skill(metric_agent, skill, state["scope"])
                query = thought_retrieval_prompt(
                    state["scope"]["target_asset"],
                    scoring.name,
                    [metric.label for metric in scoring.metrics],
                )
                with trace_span(
                    "analysis.skill.thought_retrieval",
                    skill_name=skill.name,
                ) as span:
                    try:
                        documents = await retriever.aretrieve(query)
                    except Exception:  # noqa: BLE001 - 检索失败保留已计算指标，明确记录失败状态。
                        span.metadata["status"] = "error"
                        _LOG.exception("思想材料检索失败：%s", skill.name)
                        result["thought_context"] = {
                            "query": query,
                            "status": "error",
                            "documents": [],
                        }
                    else:
                        # 空检索结果明确保留，不生成思想材料或替换已有指标结果。
                        result["thought_context"] = {
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
                        span.output = {"document_count": len(documents)}
                return {"skill_results": [result]}
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    @timed_node
    async def writer(state: DynamicAnalysisState) -> dict[str, Any]:
        with trace_span("crypto.writer") as node_span:
            try:
                names = state["plan"]["skill_names"]
                if not names:
                    report = "当前任务没有适用的分析技能；请说明希望分析的资产或问题。"
                    return {"report": report, "messages": [AIMessage(content=report)]}
                ordered = sorted(
                    state["skill_results"], key=lambda item: item["skill_name"]
                )
                facts = json.dumps(ordered, ensure_ascii=False)
                print(f"facts: {facts}", flush=True)
                skill_loader = SkillLoader.from_snapshot(
                    {name: state["skill_sources"][name] for name in names}
                )
                output_templates = "\n\n".join(
                    f"技能 {item['skill_name']}：\n"
                    f"{skill_loader.load_analysis_output(item['skill_name'])}"
                    for item in ordered
                )
                prompt = analysis_report_prompt(
                    state["scope"]["target_asset"],
                    state["scope"]["original_question"],
                    facts,
                    output_templates,
                )
                with trace_generation(
                    "crypto_analysis_writer", model=settings.smart_model
                ) as generation:
                    response = await smart_llm.ainvoke(prompt)
                    if isinstance(response.content, str):
                        generation.usage = {
                            "input": estimate_tokens(prompt),
                            "output": estimate_tokens(response.content),
                        }
                if (
                    not isinstance(response.content, str)
                    or not response.content.strip()
                ):
                    raise ValueError("报告模型未返回有效文本")
                report = response.content.strip()
                return {"report": report, "messages": [AIMessage(content=report)]}
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    builder = StateGraph(DynamicAnalysisState)
    builder.add_node("plan", plan)
    builder.add_node("skill_worker", skill_worker)
    builder.add_node("writer", writer)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", send_skills)
    builder.add_edge("skill_worker", "writer")
    builder.add_edge("writer", END)
    return builder.compile(name="crypto_analysis_agent")
