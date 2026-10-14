"""指标工具的取数、证据处理、计算调用与结果标准化。"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from traceback import format_tb
from typing import Any
from urllib.parse import urlsplit

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.crypto_analysis.calculators.derivatives_calculator import (
    FUNDING_FIELD_LABELS,
    OI_FIELD_LABELS,
)
from app.crypto_analysis.calculators.dxy_calculator import DXY_FIELD_LABELS
from app.crypto_analysis.calculators.etf_flow_calculator import ETF_FLOW_FIELD_LABELS
from app.crypto_analysis.calculators.fear_greed_calculator import (
    FEAR_GREED_FIELD_LABELS,
)
from app.crypto_analysis.calculators.fed_policy_calculator import FED_RATE_FIELD_LABELS
from app.crypto_analysis.calculators.fred_calculator import (
    M2_FIELD_LABELS,
    WALCL_FIELD_LABELS,
)
from app.crypto_analysis.calculators.stablecoin_calculator import (
    STABLECOIN_FIELD_LABELS,
)
from app.crypto_analysis.calculators.trend_calculator import TREND_FIELD_LABELS
from app.crypto_analysis.crypto_analysis_state import MetricToolContext
from app.observability import get_logger
from app.tracing import trace_span

_LOG = get_logger("crypto_analysis.metrics")


@dataclass(frozen=True)
class MetricJob:
    """一次取数及其共用原始数据的计算器，由代码定义。"""

    data_kind: str
    source_action: str
    source_function: Callable[..., Awaitable[dict[str, Any]]]
    arguments: dict[str, Any]
    calculators: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]


def _log_reason(reason: str) -> str:
    """隐藏日志原因中的完整请求地址，避免认证信息或参数携带密钥。"""
    return re.sub(
        r"https?://[^\s'\"<>]+",
        "[请求地址已隐藏]",
        reason,
        flags=re.IGNORECASE,
    )


# 指标结果校验与计算

_SOURCE_WEBSITES = {
    "eodhd.com": "https://eodhd.com/",
    "pro-api.coinmarketcap.com": "https://coinmarketcap.com/",
    "api.stlouisfed.org": "https://fred.stlouisfed.org/",
    "api.cryptoetf.today": "https://cryptoetf.today/",
    "api.binance.com": "https://www.binance.com/",
    "fapi.binance.com": "https://www.binance.com/",
}


# 各计算器依次声明原始字段、必填计算字段、允许为空的计算字段。
_METRIC_FIELDS = {
    "Fed_Rate_Change_3M": FED_RATE_FIELD_LABELS,
    "Fear_Greed": FEAR_GREED_FIELD_LABELS,
    "Trend": TREND_FIELD_LABELS,
    "Funding": FUNDING_FIELD_LABELS,
    "OI": OI_FIELD_LABELS,
    "ETF_Flows": ETF_FLOW_FIELD_LABELS,
    "DXY_Trend": DXY_FIELD_LABELS,
    "Stable_MCap": STABLECOIN_FIELD_LABELS,
    "M2_YoY": M2_FIELD_LABELS,
    "Fed_Balance": WALCL_FIELD_LABELS,
}


def metric_field_labels(metric_id: str) -> dict[str, str]:
    """校验计算器声明的字段中文名称，原始值和计算值共用一张指标表。"""
    groups = _METRIC_FIELDS[metric_id]
    labels = {field: label for group in groups for field, label in group.items()}
    if len(labels) != sum(len(group) for group in groups):
        raise ValueError(f"{metric_id} 字段定义重复")
    if any(
        not isinstance(label, str)
        or not label.strip()
        or label != label.strip()
        or not any("\u4e00" <= char <= "\u9fff" for char in label)
        for label in labels.values()
    ):
        raise ValueError(f"{metric_id} 字段缺少有效中文名称")
    names = list(labels.values())
    fixed_names = {
        "字段",
        "指标标识",
        "指标状态",
        "指标分数",
        "单位",
        "观测日期",
        "来源链接",
    }
    if len(names) != len(set(names)) or fixed_names & set(names):
        raise ValueError(f"{metric_id} 中文字段名称重复")
    return labels


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
    raw_fields, calculated_fields, nullable_fields = _METRIC_FIELDS[metric_id]
    metric_field_labels(metric_id)
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
    calculated_values.update({field: result[field] for field in nullable_fields})
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


async def _fetch_evidence(
    runtime: ToolRuntime[MetricToolContext],
    evidence_id: str,
    data_kind: str,
    source_action: str,
    source_function: Callable[..., Awaitable[dict[str, Any]]],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """取得原始响应或真实失败记录，供同一次调用直接计算。"""
    started = time.perf_counter()
    record = {
        "data_kind": data_kind,
        "source_action": source_action,
        "arguments": dict(arguments),
    }
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
            span.output = {"evidence_id": evidence_id, **record}
            span.metadata["status"] = "error"
            _LOG.warning(
                "metric.fetch.failed",
                extra={
                    "event": "metric.fetch.failed",
                    "skill_name": runtime.context["skill"].name,
                    "source_action": source_action,
                    "data_kind": data_kind,
                    "evidence_id": evidence_id,
                    "status": "data_missing",
                    "error_type": type(exc).__name__,
                    "error_message": _log_reason(str(exc)),
                    "reason": _log_reason(reason),
                    "duration_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        else:
            record.update(status="ok", raw=raw)
            span.output = {
                "evidence_id": evidence_id,
                "data_kind": data_kind,
                "status": "ok",
                "source": raw["source"],
            }
    return record


def _compute_metric(
    runtime: ToolRuntime[MetricToolContext],
    evidence_id: str,
    metric_id: str,
    evidence: dict[str, Any],
    calculator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """直接计算本次取得的原始数据，并按 Skill 字段标准化。"""
    skill = runtime.context["skill"]
    started = time.perf_counter()
    stage = "fetch"
    error_type: str | None = None
    traceback: str | None = None
    with trace_span(
        "analysis.metric.calculate", metric_id=metric_id, evidence_id=evidence_id
    ) as span:
        if evidence["status"] != "ok":
            result = missing_metric(
                skill.name, metric_id, "data_missing", evidence["reason"]
            )
        else:
            try:
                stage = "calculate"
                calculated = calculator(evidence["raw"])
                stage = "normalize"
                result = _normalize_result(
                    skill.name,
                    metric_id,
                    calculated,
                )
                if result["status"] != "ok":
                    stage = "calculate"
            except (TypeError, ValueError) as exc:
                error_type = type(exc).__name__
                result = missing_metric(skill.name, metric_id, "data_missing", str(exc))
            except Exception as exc:  # noqa: BLE001 - 计算器未预期错误记为 calculation_error，不中断技能。
                error_type = type(exc).__name__
                # 只记录栈帧；异常消息单独处理，避免完整请求地址进入日志。
                traceback = _log_reason("".join(format_tb(exc.__traceback__)))
                result = missing_metric(
                    skill.name, metric_id, "calculation_error", str(exc)
                )
        span.output = {"status": result["status"], "score": result["score"]}
        if result["status"] != "ok":
            fields = {
                "event": "metric.result.unavailable",
                "skill_name": skill.name,
                "indicator_id": metric_id,
                "evidence_id": evidence_id,
                "source_action": evidence["source_action"],
                "stage": stage,
                "status": result["status"],
                "reason": _log_reason(result["reason"]),
                "error_type": error_type,
                "duration_ms": round((time.perf_counter() - started) * 1000),
            }
            if traceback is not None:
                _LOG.error(
                    "metric.result.unavailable",
                    extra={**fields, "traceback": traceback},
                )
            else:
                _LOG.warning("metric.result.unavailable", extra=fields)
    return result


async def _execute_metrics(
    runtime: ToolRuntime[MetricToolContext], jobs: Sequence[MetricJob]
) -> Command:
    """一次工具调用完成取数和计算，只向模型返回各指标状态。"""
    metric_ids = [metric_id for job in jobs for metric_id in job.calculators]
    configured_ids = {metric.id for metric in runtime.context["skill"].scoring.metrics}
    if len(metric_ids) != len(set(metric_ids)) or set(metric_ids) != configured_ids:
        raise ValueError("组合工具的指标与当前技能配置不一致")
    if len({job.data_kind for job in jobs}) != len(jobs):
        raise ValueError("组合工具的取数类型不能重复")

    # 同一工具调用可能取多类数据，按类型区分证据 ID，避免状态合并时覆盖。
    fetched = await asyncio.gather(
        *(
            _fetch_evidence(
                runtime,
                f"{runtime.tool_call_id}:{job.data_kind}",
                job.data_kind,
                job.source_action,
                job.source_function,
                job.arguments,
            )
            for job in jobs
        )
    )
    evidence: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    indicator_evidence_ids: dict[str, str] = {}
    for job, record in zip(jobs, fetched, strict=True):
        evidence_id = f"{runtime.tool_call_id}:{job.data_kind}"
        evidence[evidence_id] = record
        for metric_id, calculator in job.calculators.items():
            results[metric_id] = _compute_metric(
                runtime,
                evidence_id,
                metric_id,
                record,
                calculator,
            )
            indicator_evidence_ids[metric_id] = evidence_id

    summaries = [
        {
            "indicator_id": metric_id,
            "status": result["status"],
            **({"reason": result["reason"]} if result["status"] != "ok" else {}),
        }
        for metric_id, result in results.items()
    ]
    return Command(
        update={
            "evidence": evidence,
            "indicator_results": results,
            "messages": [
                ToolMessage(
                    content=json.dumps({"indicators": summaries}, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                    # 证据关联供程序校验和追踪，不进入模型消息内容。
                    artifact={"indicator_evidence_ids": indicator_evidence_ids},
                )
            ],
        }
    )
