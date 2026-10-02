"""校验工具产生的指标结果，并按业务权重聚合维度。"""

import json  # 把维度业务任务传给工具决策模型。
from datetime import date, datetime, timezone  # 统一解析请求的 UTC 截止时间和观测日期。
from math import isfinite  # 校验分数是否为有限数值。
from typing import Any  # 表示各工具不同的原始响应。

from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

from app.crypto_analysis.calculators.dxy_calculator import (
    calculate_dxy_metrics,  # DXY 程序计算器。
)
from app.crypto_analysis.calculators.trend_calculator import calculate_trend
from app.crypto_analysis.calculators.etf_flow_calculator import calculate_etf_flows
from app.crypto_analysis.calculators.fred_calculator import (  # FRED 程序计算器。
    calculate_m2_yoy_change,
    calculate_walcl_ma4,
)
from app.crypto_analysis.calculators.stablecoin_calculator import (
    _stablecoin_metrics,  # 稳定币程序计算器。
)
from app.crypto_analysis.crypto_dimension_catalog import (
    DimensionConfig,  # 读取维度配置类型。
)
from app.tracing import trace_span


class MetricContext(TypedDict):
    """由程序注入的维度与请求范围，模型不能修改。"""

    dimension: DimensionConfig
    scope: dict[str, str]


# 结果字段属于计算器的代码契约，不再由 JSON 决定。
_METRIC_FIELDS = {
    "Trend": (
        ("symbol", "kline_count", "first_open_time", "last_close_time", "close"),
        ("ma50", "ma200", "ma50_20_days_ago", "ma200_20_days_ago", "adx14", "plus_di", "minus_di",
         "admission", "timing", "confirmation", "intermediate_values", "calculation_method"),
    ),
    "Funding": (
        ("symbol", "funding_times", "funding_rates", "funding_timestamp", "price_source", "funding_source"),
        ("funding_average_pct", "funding_price_change_pct"),
    ),
    "OI": (
        ("symbol", "oi_timestamp", "latest_oi", "baseline_oi", "latest_price", "baseline_price", "price_source", "funding_source"),
        ("oi_change_pct", "price_change_pct", "funding_average_pct", "funding_score", "market_state"),
    ),
    "ETF_Flows": (
        ("trading_dates", "daily_flows_usd_m", "latest_flow_usd_m", "previous_five_day_flow_usd_m"),
        ("five_day_flow_usd_m", "twenty_day_flow_usd_m", "five_day_score", "twenty_day_score", "conflict_flag"),
    ),
    "DXY_Trend": (
        ("dxy", "ma50", "ma200"),
        ("ma50_deviation_pct",),
    ),
    "Stable_MCap": (
        ("usdt_market_cap", "usdc_market_cap", "total_market_cap", "ma30_market_cap",
         "seven_day_market_cap", "thirty_day_market_cap", "quote_timestamps"),
        ("ma30_change_pct", "seven_day_change_pct", "thirty_day_change_pct", "conflict_flag"),
    ),
    "M2_YoY": (
        ("latest_value", "previous_value", "latest_year_ago_value", "previous_year_ago_value", "comparison_months"),
        ("latest_yoy_pct", "previous_yoy_pct", "yoy_change_pp"),
    ),
    "Fed_Balance": (
        ("latest_value", "ma4_value", "comparison_dates", "weekly_values"),
        ("ma4_change_pct",),
    ),
}


def _calculate_trend(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按维度05文档公式计算均线和趋势，保留原始数据缺失状态。"""
    return calculate_trend(raw, scope)


def _calculate_etf(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:
    """按已确认的ETF金额分档计算，保留完整性缺失状态。"""
    return calculate_etf_flows(raw, scope)


def _calculate_dxy(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:  # 用程序计算 DXY 指标。
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)  # 取得统一截止时间。
    result = calculate_dxy_metrics(raw["rows"], cutoff.date())  # 运行原有均线及评分逻辑。
    return {**result, "confidence": 1}  # 直接来源且数据完整时标记证据档位。


def _calculate_stable(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:  # 用程序计算稳定币市值指标。
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)  # 取得统一截止时间。
    result = _stablecoin_metrics(raw["payload"], cutoff)  # 运行原有市值及评分逻辑。
    confidence = 0.75 if result.get("conflict_flag") else 1  # 子项冲突时降低结论明确度。
    return {**result, "confidence": confidence}  # 保留程序计算结果。


def _calculate_m2(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:  # 用程序计算 M2 同比变化。
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)  # 取得统一截止时间。
    result = calculate_m2_yoy_change(raw["payload"], cutoff.date())  # 运行原有月度计算。
    return {**result, "confidence": 1}  # 完整官方数据的证据档位。


def _calculate_walcl(raw: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:  # 用程序计算 WALCL 周度指标。
    cutoff = datetime.fromisoformat(scope["as_of"]).astimezone(timezone.utc)  # 取得统一截止时间。
    result = calculate_walcl_ma4(raw["payload"], cutoff.date())  # 运行原有 MA4 计算。
    return {**result, "confidence": 1}  # 完整官方数据的证据档位。


def missing_metric(dimension_id: str, metric_id: str, status: str, reason: str) -> dict[str, Any]:  # 统一表示无效指标。
    return {  # 保留指标位置，供聚合与页面展示。
        "dimension_id": dimension_id,  # 所属维度。
        "indicator_id": metric_id,  # 指标标识。
        "status": status,  # 数据缺失、计算失败或不适用。
        "reason": reason,  # 明确说明原因。
        "raw_values": {},  # 缺失时没有可核验的原始值。
        "calculated_values": {},  # 缺失时没有计算值。
        "unit": None,  # 缺失时不编造单位。
        "observed_at": None,  # 缺失时不编造观测日期。
        "source": None,  # 未成功调用时不标记来源。
        "score": None,  # 缺失与有效零分严格区分。
        "confidence": 0,  # 无有效证据时信心档位为零。
    }


def _normalize_result(dimension_id: str, metric_id: str, result: dict[str, Any], source_action: str) -> dict[str, Any]:  # 形成统一指标记录。
    if result.get("status") != "ok":  # 先处理计算器明确报告的数据缺失。
        status = result.get("status")  # 读取计算器状态。
        if status not in {"data_missing", "calculation_error", "not_applicable"}:  # 状态必须属于约定枚举。
            raise ValueError(f"{dimension_id}.{metric_id} 返回无效状态")  # 拒绝无法解释的结果。
        reason = result.get("reason")  # 读取具体原因。
        if not isinstance(reason, str) or not reason:  # 无效状态必须说明原因。
            raise ValueError(f"{dimension_id}.{metric_id} 缺少原因")  # 不生成空白原因。
        return missing_metric(dimension_id, metric_id, status, reason)  # 记录缺失状态。
    raw_fields, calculated_fields = _METRIC_FIELDS[metric_id]
    required = [*raw_fields, *calculated_fields, "unit", "observed_at", "source", "score", "confidence"]  # 必要字段。
    absent = [field for field in required if result.get(field) is None]  # 查找缺失值。
    if absent:  # 必要字段不全不能给正式分。
        raise ValueError(f"{dimension_id}.{metric_id} 缺少字段：{', '.join(absent)}")  # 说明具体字段。
    score = result["score"]  # 取程序计算的分数。
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score) or not 0 <= score <= 10:  # 校验 0–10。
        raise ValueError(f"{dimension_id}.{metric_id} 分数无效")  # 不让错误分数进入聚合。
    confidence = result["confidence"]  # 取证据质量档位。
    if confidence not in (0, 0.5, 0.75, 1):  # 档位必须符合总规则。
        raise ValueError(f"{dimension_id}.{metric_id} confidence 无效")  # 阻止任意数值。
    if not all(isinstance(result[field], str) and result[field] for field in ("unit", "observed_at", "source")):  # 沿用原有的文本字段校验。
        raise ValueError(f"{dimension_id}.{metric_id} 的单位、日期或来源无效")  # 缺少可核验字段不能给分。
    try:  # 沿用原有的观测日期格式约束。
        date.fromisoformat(result["observed_at"])  # 确认是标准日期。
    except ValueError as exc:  # 日期格式不正确时保留原因。
        raise ValueError(f"{dimension_id}.{metric_id} 的观测日期格式无效") from exc  # 不接受任意文本。
    return {  # 返回统一的指标事实记录。
        "dimension_id": dimension_id,  # 所属维度。
        "indicator_id": metric_id,  # 指标标识。
        "status": "ok",  # 已取得并校验数据。
        "reason": None,  # 有效结果无缺失原因。
        "raw_values": {field: result[field] for field in raw_fields},  # 保留必要原始值。
        "calculated_values": {field: result[field] for field in calculated_fields} | ({field: result[field] for field in ("oi_24h_change_pct", "price_24h_change_pct")} if metric_id == "OI" else {}),  # 保留必要计算值。
        "unit": result["unit"],  # 使用工具返回单位。
        "observed_at": result["observed_at"],  # 使用实际观测日期。
        "source": {"action": source_action, "url": result["source"]},  # 记录实际成功的数据源。
        "score": score,  # 保留有效零分。
        "confidence": confidence,  # 保留证据质量档位。
    }


async def execute_dimension(
    metric_agent: Any,
    dimension: DimensionConfig,
    scope: dict[str, str],
    analysis_focus: str,
) -> dict[str, Any]:
    """让维度 Agent 自主调用工具，只接收工具写入的可核验指标结果。"""
    with trace_span("analysis.dimension.execute", dimension_id=dimension.dimension_id) as node_span:
        try:
            metrics = [{"id": metric.id, "label": metric.label, "kind": metric.kind} for metric in dimension.metrics]
            prompt = (
                f"目标币种：{scope['target_asset']}；数据截止时间：{scope['as_of']}。"
                f"\n用户原始问题：{scope['original_question']}"
                f"\n研究重点：{analysis_focus}"
                f"\n本维度：{dimension.dimension_id} {dimension.name}"
                f"\n需取得证据的指标：{json.dumps(metrics, ensure_ascii=False)}"
                "\n根据各工具的能力说明自行选择取数和计算步骤。"
                "先查询真实数据，再用返回的 evidence_id 调用适用的计算工具。"
                "各项指标都应取得工具结果或明确缺失原因。完成后简要说明证据是否齐全。"
            )
            with trace_span("analysis.metric_agent.graph", dimension_id=dimension.dimension_id):
                state = await metric_agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)], "evidence": {}, "indicator_results": {}, "research_evidence": []},
                    context={"dimension": dimension, "scope": dict(scope)},
                    config={"recursion_limit": 60},
                )
            # 模型最终生成的文本不作为指标分数，遗漏项明确记为未取得证据。
            records = state["indicator_results"]
            indicators = [
                records[metric.id] if metric.id in records else missing_metric(
                    dimension.dimension_id, metric.id, "data_missing", "维度 Agent 未取得该指标的可核验工具结果"
                )
                for metric in dimension.metrics
            ]
            result = aggregate_dimension(dimension, indicators)
            result["research_evidence"] = state["research_evidence"]
            return result
        except Exception:
            node_span.metadata["status"] = "error"
            raise


def aggregate_dimension(dimension: DimensionConfig, indicators: list[dict[str, Any]]) -> dict[str, Any]:
    """保留既有覆盖率、必需指标、评分与置信度聚合规则。"""
    ids = [item["indicator_id"] for item in indicators]
    if len(ids) != len(set(ids)) or set(ids) != {metric.id for metric in dimension.metrics}:
        raise ValueError("维度指标记录不完整或重复")
    valid = {item["indicator_id"]: item for item in indicators if item["status"] == "ok"}  # 找出有效指标。
    coverage = round(sum(metric.weight for metric in dimension.metrics if metric.id in valid), 10)  # 计算原权重覆盖率。
    required_valid = set(dimension.required_metrics) <= set(valid)  # 检查维度专属门槛。
    formal = required_valid and coverage >= dimension.min_coverage - 1e-9  # 判断是否有正式分。
    if formal:  # 按有效指标权重归一化计算维度分。
        score = sum(metric.weight * valid[metric.id]["score"] for metric in dimension.metrics if metric.id in valid) / coverage  # 保留未舍入值。
        raw_confidence = sum(metric.weight * valid[metric.id]["confidence"] for metric in dimension.metrics if metric.id in valid)  # 按全部原权重计算。
        confidence = next(level for level in (1, 0.75, 0.5, 0) if raw_confidence >= level - 1e-9)  # 向下归档。
        status = "ok" if len(valid) == len(dimension.metrics) else "partial"  # 区分全部与部分有效。
        reason = None  # 正式有效时不写缺失原因。
    else:  # 没有正式分时保留具体状态。
        score = None  # 不把缺失计算为零分。
        confidence = 0  # 无正式维度分时归零。
        reason = "；".join(item["reason"] for item in indicators if item["status"] != "ok" and item.get("reason"))
        if not reason:
            reason = "有效指标覆盖率或必需指标未达到正式评分条件"
        if all(item["status"] == "not_applicable" for item in indicators):  # 整维不适用时保留独立状态。
            status = "not_applicable"  # 沿用既有维度状态约定。
        elif any(item["status"] == "calculation_error" for item in indicators):  # 检查是否发生计算失败。
            status = "calculation_error"  # 计算异常不写成普通缺数。
        else:  # 剩余情况均属于数据或门槛不足。
            status = "data_missing"  # 返回真实缺失状态。
    return {  # 把维度及其指标一并交给父图。
        "dimension_id": dimension.dimension_id,  # 维度编号。
        "name": dimension.name,  # 维度名称。
        "status": status,  # 维度状态。
        "reason": reason,  # 维度缺失原因。
        "score": score,  # 正式维度分或空值。
        "confidence": confidence,  # 证据质量档位。
        "coverage": coverage,  # 有效原权重覆盖率。
        "valid_indicators": len(valid),  # 有效指标数量。
        "total_indicators": len(dimension.metrics),  # 本维指标总数。
        "indicators": indicators,  # 完整指标事实记录。
    }
