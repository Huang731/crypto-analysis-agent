"""指标工具的取数、证据处理、计算调用与结果标准化。"""

import json
from collections.abc import Awaitable, Callable
from datetime import date
from math import isfinite
from typing import Any
from urllib.parse import urlsplit

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.tracing import trace_span

# 指标结果校验与计算

_SOURCE_WEBSITES = {
    "eodhd.com": "https://eodhd.com/",
    "pro-api.coinmarketcap.com": "https://coinmarketcap.com/",
    "api.stlouisfed.org": "https://fred.stlouisfed.org/",
    "api.cryptoetf.today": "https://cryptoetf.today/",
    "api.binance.com": "https://www.binance.com/",
    "fapi.binance.com": "https://www.binance.com/",
}


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
            "daily_flows_usd_100m",
            "latest_flow_usd_100m",
            "previous_five_day_flow_usd_100m",
        ),
        (
            "five_day_flow_usd_100m",
            "twenty_day_flow_usd_100m",
            "five_day_score",
            "twenty_day_score",
        ),
    ),
    "DXY_Trend": (
        ("dxy", "ma50", "ma200"),
        ("ma50_deviation_pct", "ma50_trend", "ma200_trend"),
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
    skill_name: str, metric_id: str, status: str, reason: str
) -> dict[str, Any]:
    return {
        "skill_name": skill_name,
        "indicator_id": metric_id,
        "status": status,
        "reason": reason,
        "raw_values": {},
        "calculated_values": {},
        "unit": None,
        "observed_at": None,
        "source": None,
        "url": None,
        "score": None,
    }


def _normalize_result(
    skill_name: str, metric_id: str, result: dict[str, Any]
) -> dict[str, Any]:
    if result.get("status") != "ok":
        status = result.get("status")
        if status not in {"data_missing", "calculation_error", "not_applicable"}:
            raise ValueError(f"{skill_name}.{metric_id} 返回无效状态")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{skill_name}.{metric_id} 缺少原因")
        return missing_metric(skill_name, metric_id, status, reason)
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
        raise ValueError(f"{skill_name}.{metric_id} 缺少字段：{', '.join(absent)}")
    score = result["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not isfinite(score)
        or not 0 <= score <= 10
    ):
        raise ValueError(f"{skill_name}.{metric_id} 分数无效")
    if not all(
        isinstance(result[field], str) and result[field]
        for field in ("unit", "observed_at", "source")
    ):
        raise ValueError(f"{skill_name}.{metric_id} 的单位、日期或来源无效")
    try:
        date.fromisoformat(result["observed_at"])
    except ValueError as exc:
        raise ValueError(f"{skill_name}.{metric_id} 的观测日期格式无效") from exc
    calculated_values = {field: result[field] for field in calculated_fields}
    if metric_id == "OI":
        calculated_values.update(
            oi_24h_change_pct=result["oi_24h_change_pct"],
            price_24h_change_pct=result["price_24h_change_pct"],
        )
    # 报告展示数据提供方官网，取数状态中的真实 API 地址保持原样。
    source_host = urlsplit(result["source"]).hostname
    if source_host not in _SOURCE_WEBSITES:
        raise ValueError(f"指标来源未配置官网：{source_host}")
    return {
        "skill_name": skill_name,
        "indicator_id": metric_id,
        "status": "ok",
        "reason": None,
        "raw_values": {field: result[field] for field in raw_fields},
        "calculated_values": calculated_values,
        "unit": result["unit"],
        "observed_at": result["observed_at"],
        "source": result["source"],
        "url": _SOURCE_WEBSITES[source_host],
        "score": score,
    }


# 工具证据处理


def _tool_message(
    runtime: ToolRuntime[MetricToolContext], content: dict[str, Any]
) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
    )


def _tool_error(runtime: ToolRuntime[MetricToolContext], reason: str) -> ToolMessage:
    """把可纠正的调用参数错误交还模型，不产生指标数据或分数。"""
    return ToolMessage(
        content=json.dumps({"status": "error", "reason": reason}, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id,
        status="error",
    )


async def _fetch_evidence(
    runtime: ToolRuntime[MetricToolContext],
    data_kind: str,
    source_action: str,
    source_function: Callable[..., Awaitable[dict[str, Any]]],
    arguments: dict[str, Any],
) -> Command:
    """保存真实原始响应，只把证据标识与概要交给模型。"""
    evidence_id = runtime.tool_call_id
    record = {"data_kind": data_kind, "source_action": source_action}
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
            record.update(status="data_missing", reason=reason)
            summary = {"evidence_id": evidence_id, **record}
            span.metadata["status"] = "error"
        else:
            record.update(status="ok", raw=raw)
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
    runtime: ToolRuntime[MetricToolContext],
    evidence_id: str,
    metric_id: str,
    data_kind: str,
    calculator: Callable[[dict[str, Any]], dict[str, Any]],
) -> Command | ToolMessage:
    """计算器直接读取工具状态中的原始数据，模型不传入或修改数值。"""
    skill = runtime.context["skill"]
    scoring = skill.scoring
    if metric_id not in {metric.id for metric in scoring.metrics}:
        return _tool_error(
            runtime, f"指标 {metric_id} 不属于当前技能，请选择本技能适用的工具"
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
                skill.name, metric_id, "data_missing", evidence["reason"]
            )
        else:
            try:
                calculated = calculator(evidence["raw"])
                result = _normalize_result(
                    skill.name,
                    metric_id,
                    calculated,
                )
            except (TypeError, ValueError) as exc:
                result = missing_metric(skill.name, metric_id, "data_missing", str(exc))
            except Exception as exc:  # noqa: BLE001 - 计算器未预期错误记为 calculation_error，不中断技能。
                result = missing_metric(
                    skill.name, metric_id, "calculation_error", str(exc)
                )
        span.output = {"status": result["status"], "score": result["score"]}
    return Command(
        update={
            "indicator_results": {metric_id: result},
            "messages": [_tool_message(runtime, result)],
        }
    )
