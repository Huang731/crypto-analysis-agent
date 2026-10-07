"""加密货币分析：请求规划、证据与指标校验、维度执行、并行汇总及报告生成。"""

import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone
from math import fsum, isfinite
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict

from app.config.config import settings
from app.crypto_analysis.calculators.dxy_calculator import calculate_dxy_metrics
from app.crypto_analysis.calculators.etf_flow_calculator import calculate_etf_flows
from app.crypto_analysis.calculators.fear_greed_calculator import (
    calculate_fear_and_greed,
)
from app.crypto_analysis.calculators.fred_calculator import (
    calculate_m2_yoy_change,
    calculate_walcl_ma4,
)
from app.crypto_analysis.calculators.stablecoin_calculator import _stablecoin_metrics
from app.crypto_analysis.calculators.trend_calculator import calculate_trend
from app.crypto_analysis.crypto_analysis_state import (
    DimensionWorkerState,
    DynamicAnalysisState,
    MetricAgentState,
    MetricContext,
)
from app.crypto_analysis.skill_loader import (
    LoadedSkill,
    SkillLoader,
    SkillScoring,
    get_skill_loader,
)
from app.observability import get_logger, timed_node
from app.rag.retriever import KBRetriever
from app.tracing import estimate_tokens, trace_generation, trace_span
from prompts.crypto_analysis import (
    analysis_plan_prompt,
    analysis_report_prompt,
    metric_agent_system_prompt,
    thought_retrieval_prompt,
)

_LOG = get_logger("crypto_analysis.dynamic_analysis")


# 请求规划模型


class AnalysisPlan(BaseModel):
    """一轮分析的目标币种和所需技能，不指定具体工具。"""

    model_config = ConfigDict(extra="forbid")
    target_asset: str
    skill_names: list[str]


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


# 指标结果校验与计算


_METRIC_FIELDS = {
    "Fed_Rate_Change_3M": (
        (
            "current_date",
            "baseline_date",
            "comparison_date",
            "current_lower_pct",
            "current_upper_pct",
            "baseline_lower_pct",
            "baseline_upper_pct",
        ),
        ("current_midpoint_pct", "baseline_midpoint_pct", "change_bp", "policy_change"),
    ),
    "Fear_Greed": (
        ("value", "value_classification", "record_timestamp"),
        ("signal_reference",),
    ),
    "Trend": (
        ("symbol", "kline_count", "first_open_time", "last_close_time", "close"),
        (
            "ma50",
            "ma200",
            "ma50_20_days_ago",
            "ma200_20_days_ago",
            "adx14",
            "plus_di",
            "minus_di",
            "admission",
            "timing",
            "confirmation",
            "intermediate_values",
            "calculation_method",
        ),
    ),
    "Funding": (
        (
            "symbol",
            "funding_times",
            "funding_rates",
            "funding_timestamp",
            "price_source",
            "funding_source",
        ),
        ("funding_average_pct", "funding_price_change_pct"),
    ),
    "OI": (
        (
            "symbol",
            "oi_timestamp",
            "latest_oi",
            "baseline_oi",
            "latest_price",
            "baseline_price",
            "price_source",
            "funding_source",
        ),
        (
            "oi_change_pct",
            "price_change_pct",
            "funding_average_pct",
            "funding_score",
            "market_state",
        ),
    ),
    "ETF_Flows": (
        (
            "trading_dates",
            "daily_flows_usd_m",
            "latest_flow_usd_m",
            "previous_five_day_flow_usd_m",
        ),
        (
            "five_day_flow_usd_m",
            "twenty_day_flow_usd_m",
            "five_day_score",
            "twenty_day_score",
            "conflict_flag",
        ),
    ),
    "DXY_Trend": (
        ("dxy", "ma50", "ma200"),
        ("ma50_deviation_pct",),
    ),
    "Stable_MCap": (
        (
            "usdt_market_cap",
            "usdc_market_cap",
            "total_market_cap",
            "ma30_market_cap",
            "seven_day_market_cap",
            "thirty_day_market_cap",
            "quote_timestamps",
        ),
        (
            "ma30_change_pct",
            "seven_day_change_pct",
            "thirty_day_change_pct",
            "conflict_flag",
        ),
    ),
    "M2_YoY": (
        (
            "latest_value",
            "previous_value",
            "latest_year_ago_value",
            "previous_year_ago_value",
            "comparison_months",
        ),
        ("latest_yoy_pct", "previous_yoy_pct", "yoy_change_pp"),
    ),
    "Fed_Balance": (
        ("latest_value", "ma4_value", "comparison_dates", "weekly_values"),
        ("ma4_change_pct",),
    ),
}


def missing_metric(
    dimension_id: str, metric_id: str, status: str, reason: str
) -> dict[str, Any]:
    return {
        "dimension_id": dimension_id,
        "indicator_id": metric_id,
        "status": status,
        "reason": reason,
        "raw_values": {},
        "calculated_values": {},
        "unit": None,
        "observed_at": None,
        "source": None,
        "score": None,
    }


def _normalize_result(
    dimension_id: str, metric_id: str, result: dict[str, Any], source_action: str
) -> dict[str, Any]:
    if result.get("status") != "ok":
        status = result.get("status")
        if status not in {"data_missing", "calculation_error", "not_applicable"}:
            raise ValueError(f"{dimension_id}.{metric_id} 返回无效状态")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{dimension_id}.{metric_id} 缺少原因")
        return missing_metric(dimension_id, metric_id, status, reason)
    raw_fields, calculated_fields = _METRIC_FIELDS[metric_id]
    required = [
        *raw_fields,
        *calculated_fields,
        "unit",
        "observed_at",
        "source",
        "score",
    ]
    absent = [field for field in required if result.get(field) is None]
    if absent:
        raise ValueError(f"{dimension_id}.{metric_id} 缺少字段：{', '.join(absent)}")
    score = result["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(score)
        or not 0 <= score <= 10
    ):
        raise ValueError(f"{dimension_id}.{metric_id} 分数无效")
    if not all(
        isinstance(result[field], str) and result[field]
        for field in ("unit", "observed_at", "source")
    ):
        raise ValueError(f"{dimension_id}.{metric_id} 的单位、日期或来源无效")
    try:
        date.fromisoformat(result["observed_at"])
    except ValueError as exc:
        raise ValueError(f"{dimension_id}.{metric_id} 的观测日期格式无效") from exc
    return {
        "dimension_id": dimension_id,
        "indicator_id": metric_id,
        "status": "ok",
        "reason": None,
        "raw_values": {field: result[field] for field in raw_fields},
        "calculated_values": {field: result[field] for field in calculated_fields}
        | (
            {
                field: result[field]
                for field in ("oi_24h_change_pct", "price_24h_change_pct")
            }
            if metric_id == "OI"
            else {}
        ),
        "unit": result["unit"],
        "observed_at": result["observed_at"],
        "source": {"action": source_action, "url": result["source"]},
        "score": score,
    }


def _calculate_dxy(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)
    result = calculate_dxy_metrics(raw["rows"], cutoff.date())
    return result


def _calculate_stable(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)
    result = _stablecoin_metrics(raw["payload"], cutoff)
    return result


def _calculate_m2(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)
    result = calculate_m2_yoy_change(raw["payload"], cutoff.date())
    return result


def _calculate_walcl(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)
    result = calculate_walcl_ma4(raw["payload"], cutoff.date())
    return result


def _calculate_etf(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按已确认的ETF金额分档计算，保留完整性缺失状态。"""
    return calculate_etf_flows(raw, scope)


def _calculate_trend(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按维度05文档公式计算均线和趋势，保留原始数据缺失状态。"""
    return calculate_trend(raw, scope)


def _calculate_fear_greed(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按维度06分档计算市场情绪，保留缺失状态。"""
    return calculate_fear_and_greed(raw, scope)


# 证据处理与维度工具Agent


def _tool_message(
    runtime: ToolRuntime[MetricContext], content: dict[str, Any]
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
    )


def _tool_error(runtime: ToolRuntime[MetricContext], reason: str) -> ToolMessage:
    """把可纠正的调用参数错误交还模型，不产生指标数据或分数。"""
    return ToolMessage(
        content=json.dumps({"status": "error", "reason": reason}, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
        status="error",
    )


async def _fetch_evidence(
    runtime: ToolRuntime[MetricContext],
    data_kind: str,
    source_action: str,
    source_function: Callable[..., Awaitable[dict[str, Any]]],
    arguments: dict[str, Any],
) -> Command:
    """保存真实原始响应，只把证据标识与概要交给模型。"""
    evidence_id = runtime.tool_call_id
    with trace_span(
        "analysis.metric.fetch", data_kind=data_kind, source_action=source_action
    ) as span:
        try:
            raw = await source_function(**arguments)
        except Exception as exc:  # noqa: BLE001 - 取数失败类型不固定，作为缺失证据交还 Agent。
            # 取数失败作为真实缺失证据交还 Agent，不返回其他来源或缓存数据。
            reason = (
                str(exc)
                if isinstance(exc, (RuntimeError, TypeError, ValueError))
                else type(exc).__name__
            )
            record = {
                "data_kind": data_kind,
                "source_action": source_action,
                "status": "data_missing",
                "reason": reason,
            }
            summary = {"evidence_id": evidence_id, **record}
            span.metadata["status"] = "error"
        else:
            record = {
                "data_kind": data_kind,
                "source_action": source_action,
                "status": "ok",
                "raw": raw,
            }
            summary = {
                "evidence_id": evidence_id,
                "data_kind": data_kind,
                "status": "ok",
                "source": raw["source"],
            }
            span.output = summary
    return Command(
        update={
            "evidence": {evidence_id: record},
            "messages": [_tool_message(runtime, summary)],
        }
    )


def _compute_metric(
    runtime: ToolRuntime[MetricContext],
    evidence_id: str,
    metric_id: str,
    data_kind: str,
    calculator: Callable[[dict[str, Any], dict[str, str]], dict[str, Any]],
) -> Command | ToolMessage:
    """计算器直接读取工具状态中的原始数据，模型不传入或修改数值。"""
    scoring = runtime.context["skill"]
    if metric_id not in {metric.id for metric in scoring.metrics}:
        return _tool_error(
            runtime, f"指标 {metric_id} 不属于当前维度，请选择本维度适用的工具"
        )
    if evidence_id not in runtime.state["evidence"]:
        return _tool_error(
            runtime, "证据 ID 不存在，请先取得真实数据并使用工具返回的 evidence_id"
        )
    evidence = runtime.state["evidence"][evidence_id]
    if evidence["data_kind"] != data_kind:
        return _tool_error(
            runtime, f"证据类型不适用于指标 {metric_id}，请核对取数工具返回的 data_kind"
        )
    with trace_span(
        "analysis.metric.calculate", metric_id=metric_id, evidence_id=evidence_id
    ) as span:
        if evidence["status"] != "ok":
            result = missing_metric(
                scoring.dimension_id, metric_id, "data_missing", evidence["reason"]
            )
        else:
            try:
                calculated = calculator(evidence["raw"], runtime.context["scope"])
                result = _normalize_result(
                    scoring.dimension_id,
                    metric_id,
                    calculated,
                    evidence["source_action"],
                )
            except (TypeError, ValueError) as exc:
                result = missing_metric(
                    scoring.dimension_id, metric_id, "data_missing", str(exc)
                )
            except Exception as exc:  # noqa: BLE001 - 计算器未预期错误记为 calculation_error，不中断维度。
                result = missing_metric(
                    scoring.dimension_id, metric_id, "calculation_error", str(exc)
                )
        span.output = {"status": result["status"], "score": result["score"]}
    return Command(
        update={
            "indicator_results": {metric_id: result},
            "messages": [_tool_message(runtime, result)],
        }
    )


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


def build_metric_agent(llm: Any) -> CompiledStateGraph:
    """构建维度工具 Agent，由模型按当前技能调用取数和计算工具。"""
    # 各维度工具依赖本模块的共享辅助函数，统一在构建 Agent 时导入以避免循环导入。
    from app.crypto_analysis.tools.derivatives_tools import build_derivatives_tools
    from app.crypto_analysis.tools.etf_tools import build_etf_tools
    from app.crypto_analysis.tools.fed_policy_tools import build_fed_policy_tools
    from app.crypto_analysis.tools.liquidity_tools import build_liquidity_tools
    from app.crypto_analysis.tools.sentiment_tools import build_sentiment_tools
    from app.crypto_analysis.tools.trend_tools import build_trend_tools

    tools = [
        *build_liquidity_tools(),
        *build_fed_policy_tools(),
        *build_etf_tools(),
        *build_derivatives_tools(),
        *build_trend_tools(),
        *build_sentiment_tools(),
    ]
    return create_agent(
        model=llm,
        tools=tools,
        # 声明 evidence 与 indicator_results，工具才能写入原始数据和指标结果。
        state_schema=MetricAgentState,
        # 声明 skill 与 scope，工具才能读取币种、截止时间，且模型不能修改。
        context_schema=MetricContext,
        system_prompt=metric_agent_system_prompt(),
        middleware=[MetricModelTrace()],
    )


# 维度聚合与执行


def aggregate_skill_results(
    scoring: SkillScoring, indicators: list[dict[str, Any]]
) -> dict[str, Any]:
    """根据 Skill 权重聚合有效指标，不设置最低覆盖率。"""
    ids = [item["indicator_id"] for item in indicators]
    if len(ids) != len(set(ids)) or set(ids) != {
        metric.id for metric in scoring.metrics
    }:
        raise ValueError("维度指标记录不完整或重复")
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
        "dimension_id": scoring.dimension_id,
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
    并额外保留一轮用于纠正一次错误调用。
    """
    return metric_count * 4 + 3


async def execute_skill(
    metric_agent: Any,
    skill: LoadedSkill,
    scope: dict[str, str],
) -> dict[str, Any]:
    """让维度 Agent 自主调用工具，只接收工具写入的可核验指标结果。"""
    scoring = skill.scoring
    with trace_span(
        "analysis.dimension.execute", dimension_id=scoring.dimension_id
    ) as node_span:
        try:
            with trace_span(
                "analysis.metric_agent.graph", dimension_id=scoring.dimension_id
            ):
                state = await metric_agent.ainvoke(
                    {
                        "messages": [HumanMessage(content=skill.body)],
                        "evidence": {},
                        "indicator_results": {},
                        "research_evidence": [],
                    },
                    context={"skill": scoring, "scope": dict(scope)},
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
                    scoring.dimension_id,
                    metric.id,
                    "data_missing",
                    "维度 Agent 未取得该指标的可核验工具结果",
                )
                for metric in scoring.metrics
            ]
            result = aggregate_skill_results(scoring, indicators)
            result["research_evidence"] = state["research_evidence"]
            return result
        except Exception:
            node_span.metadata["status"] = "error"
            raise


# 分析主图：规划、并行维度、报告


def build_dynamic_analysis_agent(
    smart_llm: Any, fast_llm: Any, retriever: KBRetriever
) -> CompiledStateGraph:
    metric_agent = build_metric_agent(fast_llm)

    @timed_node
    async def plan(state: DynamicAnalysisState) -> dict[str, Any]:
        with trace_span("crypto.plan") as node_span:
            try:
                skill_loader = get_skill_loader()
                question = state["scope"].get("original_question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("维度分析缺少有效用户问题")
                selection = await create_analysis_plan(
                    fast_llm,
                    question,
                    skill_loader.format_skill_descriptions(),
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
    def send_dimensions(state: DynamicAnalysisState) -> list[Send] | str:
        names = state["plan"]["skill_names"]
        if not names:
            return "writer"
        return [
            Send(
                "dimension_worker",
                {
                    "skill_name": name,
                    "skill_source": state["skill_sources"][name],
                    "scope": state["scope"],
                },
            )
            for name in names
        ]

    @timed_node
    async def dimension_worker(state: DimensionWorkerState) -> DynamicAnalysisState:
        """执行当前技能并附加理论材料，指标只保存在维度结果中。"""
        with trace_span(
            "crypto.dimension_worker", skill_name=state["skill_name"]
        ) as node_span:
            try:
                skill_loader = SkillLoader.from_snapshot(
                    {state["skill_name"]: state["skill_source"]}
                )
                skill = skill_loader.load_definition(state["skill_name"])
                scoring = skill.scoring
                result = await execute_skill(metric_agent, skill, state["scope"])
                query = thought_retrieval_prompt(
                    state["scope"]["target_asset"],
                    scoring.name,
                    [metric.label for metric in scoring.metrics],
                )
                with trace_span(
                    "analysis.dimension.thought_retrieval",
                    dimension_id=scoring.dimension_id,
                ) as span:
                    documents = await retriever.aretrieve(query)
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
                return {"dimension_results": [result]}
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
                    state["dimension_results"], key=lambda item: item["dimension_id"]
                )
                facts = json.dumps(ordered, ensure_ascii=False)
                prompt = analysis_report_prompt(
                    state["scope"]["target_asset"],
                    state["scope"]["original_question"],
                    facts,
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
    builder.add_node("dimension_worker", dimension_worker)
    builder.add_node("writer", writer)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", send_dimensions)
    builder.add_edge("dimension_worker", "writer")
    builder.add_edge("writer", END)
    return builder.compile(name="crypto_analysis_agent")
