"""按维度06规则校验CMC市场情绪记录并计算固定分档。"""

from typing import Any

from app.crypto_analysis.tools.source_time_utils import utc_time


def calculate_fear_and_greed(
    raw: dict[str, Any], scope: dict[str, str]
) -> dict[str, Any]:
    """按指数固定分档评分，记录时间只用于展示。"""
    if raw.get("status") != "ok":
        return {
            "status": "data_missing",
            "reason": raw.get("reason") or "CMC未返回有效情绪记录",
        }
    try:
        if raw["mode"] != "latest":
            raise ValueError("维度06只使用latest情绪记录")
        record = raw["payload"]
        value = record["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 100
        ):
            raise ValueError("恐惧贪婪指数必须为0至100的整数")
        classification = record["value_classification"]
        if not isinstance(classification, str) or not classification.strip():
            raise ValueError("恐惧贪婪指数分类缺失")
        observed = utc_time(record["observed_at"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "data_missing", "reason": str(exc)}
    if value <= 19:
        score, signal = 5, "买入区域"
    elif value <= 39:
        score, signal = 4, "偏买入/等待确认"
    elif value <= 59:
        score, signal = 2.5, "等待时机"
    elif value <= 79:
        score, signal = 1, "持有/准备减仓"
    else:
        score, signal = 0, "减仓区域"
    return {
        "status": "ok",
        "value": value,
        "value_classification": classification,
        "record_timestamp": observed.isoformat(),
        "signal_reference": signal,
        "score": score,
        "unit": "index (0-100)",
        "observed_at": observed.date().isoformat(),
        "source": raw["source"],
    }
