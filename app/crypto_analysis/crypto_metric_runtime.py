"""按配置执行指标工具、程序计算器和维度聚合。"""

import asyncio  # 并行执行同一维度内已接入的独立指标。
from datetime import date, datetime, timezone  # 统一解析请求的 UTC 截止时间和观测日期。
from math import isfinite  # 校验分数是否为有限数值。
from typing import Any  # 表示各工具不同的原始响应。

from app.crypto_analysis.crypto_dimension_catalog import DimensionConfig, MetricConfig  # 读取维度配置类型。
from app.crypto_analysis.calculators.dxy_calculator import calculate_dxy_metrics  # DXY 程序计算器。
from app.crypto_analysis.calculators.fred_calculator import calculate_m2_yoy_change, calculate_walcl_ma4  # FRED 程序计算器。
from app.crypto_analysis.calculators.stablecoin_calculator import _stablecoin_metrics  # 稳定币程序计算器。
from app.crypto_analysis.sources.coinmarketcap_source import get_stablecoin_historical_market_cap  # 稳定币原始数据工具。
from app.crypto_analysis.sources.eodhd_source import query_dxy_from_official_mcp  # DXY 原始数据工具。
from app.crypto_analysis.sources.fred_source import get_m2_yoy_change  # FRED 原始数据工具。


async def _fetch_dxy(scope: dict[str, str]) -> dict[str, Any]:  # 查询 DXY 原始日线。
    return await query_dxy_from_official_mcp.ainvoke({"as_of": scope["as_of"]})  # 请求 MCP 原始日线。


async def _fetch_stable(scope: dict[str, str]) -> dict[str, Any]:  # 查询稳定币原始报价。
    return await get_stablecoin_historical_market_cap.ainvoke({"as_of": scope["as_of"]})  # 请求 CMC 原始报价。


async def _fetch_m2(scope: dict[str, str]) -> dict[str, Any]:  # 查询 M2SL 原始观测。
    return await get_m2_yoy_change.ainvoke({"series_id": "M2SL", "as_of": scope["as_of"]})  # 请求 FRED 原始月值。


async def _fetch_walcl(scope: dict[str, str]) -> dict[str, Any]:  # 查询 WALCL 原始观测。
    return await get_m2_yoy_change.ainvoke({"series_id": "WALCL", "as_of": scope["as_of"]})  # 请求 FRED 原始周值。


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


TOOL_REGISTRY = {  # 当前已经接入的原始数据工具。
    "eodhd_dxy_raw": _fetch_dxy,  # DXY 的 EODHD MCP 请求。
    "cmc_stable_raw": _fetch_stable,  # 稳定币的 CMC 请求。
    "fred_m2sl_raw": _fetch_m2,  # M2SL 的 FRED 请求。
    "fred_walcl_raw": _fetch_walcl,  # WALCL 的 FRED 请求。
}
CALCULATOR_REGISTRY = {  # 当前已经接入的程序计算器。
    "dxy_trend_v1": _calculate_dxy,  # DXY 均线计算器。
    "stable_mcap_v1": _calculate_stable,  # 稳定币市值计算器。
    "m2_yoy_v1": _calculate_m2,  # M2 同比计算器。
    "fed_balance_v1": _calculate_walcl,  # WALCL MA4 计算器。
}


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


def _normalize_result(dimension_id: str, metric: MetricConfig, result: dict[str, Any]) -> dict[str, Any]:  # 形成统一指标记录。
    if result.get("status") != "ok":  # 先处理计算器明确报告的数据缺失。
        status = result.get("status")  # 读取计算器状态。
        if status not in {"data_missing", "calculation_error", "not_applicable"}:  # 状态必须属于约定枚举。
            raise ValueError(f"{dimension_id}.{metric.id} 返回无效状态")  # 拒绝无法解释的结果。
        reason = result.get("reason")  # 读取具体原因。
        if not isinstance(reason, str) or not reason:  # 无效状态必须说明原因。
            raise ValueError(f"{dimension_id}.{metric.id} 缺少原因")  # 不生成空白原因。
        return missing_metric(dimension_id, metric.id, status, reason)  # 记录缺失状态。
    required = [*metric.raw_fields, *metric.calculated_fields, "unit", "observed_at", "source", "score", "confidence"]  # 必要字段。
    absent = [field for field in required if result.get(field) is None]  # 查找缺失值。
    if absent:  # 必要字段不全不能给正式分。
        raise ValueError(f"{dimension_id}.{metric.id} 缺少字段：{', '.join(absent)}")  # 说明具体字段。
    score = result["score"]  # 取程序计算的分数。
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score) or not 0 <= score <= 10:  # 校验 0–10。
        raise ValueError(f"{dimension_id}.{metric.id} 分数无效")  # 不让错误分数进入聚合。
    confidence = result["confidence"]  # 取证据质量档位。
    if confidence not in (0, 0.5, 0.75, 1):  # 档位必须符合总规则。
        raise ValueError(f"{dimension_id}.{metric.id} confidence 无效")  # 阻止任意数值。
    if not all(isinstance(result[field], str) and result[field] for field in ("unit", "observed_at", "source")):  # 沿用原有的文本字段校验。
        raise ValueError(f"{dimension_id}.{metric.id} 的单位、日期或来源无效")  # 缺少可核验字段不能给分。
    try:  # 沿用原有的观测日期格式约束。
        date.fromisoformat(result["observed_at"])  # 确认是标准日期。
    except ValueError as exc:  # 日期格式不正确时保留原因。
        raise ValueError(f"{dimension_id}.{metric.id} 的观测日期格式无效") from exc  # 不接受任意文本。
    return {  # 返回统一的指标事实记录。
        "dimension_id": dimension_id,  # 所属维度。
        "indicator_id": metric.id,  # 指标标识。
        "status": "ok",  # 已取得并校验数据。
        "reason": None,  # 有效结果无缺失原因。
        "raw_values": {field: result[field] for field in metric.raw_fields},  # 保留必要原始值。
        "calculated_values": {field: result[field] for field in metric.calculated_fields},  # 保留必要计算值。
        "unit": result["unit"],  # 使用工具返回单位。
        "observed_at": result["observed_at"],  # 使用实际观测日期。
        "source": {"action": metric.source_action, "url": result["source"]},  # 记录实际成功的数据源。
        "score": score,  # 保留有效零分。
        "confidence": confidence,  # 保留证据质量档位。
    }


async def execute_metric(dimension_id: str, metric: MetricConfig, selection: dict[str, Any], scope: dict[str, str]) -> dict[str, Any]:  # 执行一项指标。
    if metric.status == "pending":  # 未接入的维度保持明确缺失。
        return missing_metric(dimension_id, metric.id, "data_missing", "指标工具尚未接入")  # 不伪造实时数据。
    tool_id = selection["tool_id"]  # 读取已校验的工具选择。
    calculator_id = selection["calculator_id"]  # 读取已校验的计算器选择。
    try:  # 工具失败属于取数缺失。
        raw = await TOOL_REGISTRY[tool_id](scope)  # 请求原始数据。
    except Exception as exc:  # 记录真实取数失败。
        return missing_metric(dimension_id, metric.id, "data_missing", f"数据工具失败：{exc}")  # 不用其他来源补数。
    try:  # 计算器只处理这次工具取得的原始数据。
        result = CALCULATOR_REGISTRY[calculator_id](raw, scope)  # 执行程序公式。
        return _normalize_result(dimension_id, metric, result)  # 校验并标准化结果。
    except ValueError as exc:  # 原始记录不足或字段无效归为数据缺失。
        return missing_metric(dimension_id, metric.id, "data_missing", str(exc))  # 明确数据原因。
    except Exception as exc:  # 其他程序错误归为计算失败。
        return missing_metric(dimension_id, metric.id, "calculation_error", str(exc))  # 保留失败原因。


async def execute_dimension(dimension: DimensionConfig, selections: list[dict[str, Any]], scope: dict[str, str]) -> dict[str, Any]:  # 执行一个维度。
    selected = {item["metric_id"]: item for item in selections}  # 按指标标识索引计划。
    indicators = await asyncio.gather(*(execute_metric(dimension.dimension_id, metric, selected[metric.id], scope) for metric in dimension.metrics))  # 并行执行本维指标。
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
        if all(item["status"] == "not_applicable" for item in indicators):  # 整维不适用时保留独立状态。
            status = "not_applicable"  # 沿用既有维度状态约定。
        elif any(item["status"] == "calculation_error" for item in indicators):  # 检查是否发生计算失败。
            status = "calculation_error"  # 计算异常不写成普通缺数。
        else:  # 剩余情况均属于数据或门槛不足。
            status = "data_missing"  # 返回真实缺失状态。
        if dimension.dimension_id == "01" and "Stable_MCap" not in valid:  # 保留维度 01 的专属门槛说明。
            reason = "Stable_MCap 无效，Macro_Only；正式维度分不可用"  # 不把宏观参考分当正式分。
        else:  # 其他维度沿用配置中的必需项和覆盖率门槛。
            reason = "必需指标无效" if not required_valid else "有效指标权重覆盖率不足"  # 说明正式分缺失原因。
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
