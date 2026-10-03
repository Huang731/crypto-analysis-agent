"""CoinMarketCap 稳定币市值与恐惧贪婪指数工具。"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.source_time_utils import (
    utc_time,  # 只解析恐惧贪婪接口的时间字段。
)

_ROOT = Path(__file__).resolve().parents[3]  # 数据源目录比项目根目录多一层。
_BASE_URL = "https://pro-api.coinmarketcap.com"


class CMCSettings(BaseSettings):
    """读取 CoinMarketCap Pro API 密钥。"""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    cmc_pro_api_key: str = ""


async def _get_cmc(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """调用指定接口，并检查 HTTP 与 CMC 业务状态。"""
    api_key = CMCSettings().cmc_pro_api_key
    if not api_key:
        raise RuntimeError("未配置 CMC_PRO_API_KEY")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{_BASE_URL}{path}",
            params=params,
            headers={"X-CMC_PRO_API_KEY": api_key},
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), dict):
        raise TypeError("CMC 响应缺少 status")
    status = payload["status"]
    if status.get("error_code") not in (0, "0"):
        raise RuntimeError(
            f"CMC API 错误 {status.get('error_code')}: {status.get('error_message')}"
        )
    return payload


async def get_stablecoin_historical_market_cap(
    as_of: str,
) -> dict[str, Any]:  # 获取稳定币历史原始报价。
    """查询 CMC 的 USDT 和 USDC 日频历史市值，返回原始报价与来源。"""
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    payload = await _get_cmc(
        "/v3/cryptocurrency/quotes/historical",
        {"id": "825,3408", "count": 100, "interval": "daily", "convert": "USD"},
    )
    return {
        "payload": payload,
        "source": f"{_BASE_URL}/v3/cryptocurrency/quotes/historical",
    }  # 原样交给程序计算器。


async def get_fear_and_greed(
    mode: Literal["latest", "historical"] = "latest", start: int = 1, limit: int = 30
) -> dict[str, Any]:
    """查询 CMC 恐惧贪婪指数；mode 选择最新值或历史记录。"""
    if start < 1 or not 1 <= limit <= 500:
        raise ValueError("start 必须大于 0，limit 必须在 1 到 500 之间")
    path = f"/v3/fear-and-greed/{mode}"
    params = {"start": start, "limit": limit} if mode == "historical" else {}
    payload = await _get_cmc(path, params)
    data = payload.get("data")
    records = [data] if mode == "latest" else data
    if not isinstance(records, list) or not records:
        return {"status": "data_missing", "reason": "CMC 未返回恐惧贪婪指数记录"}
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("CMC 恐惧贪婪指数记录格式无效")
        value = record.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 100
        ):
            raise ValueError("CMC 恐惧贪婪指数必须是 0 到 100 的整数")
        classification = record.get("value_classification")
        if not isinstance(classification, str) or not classification:
            raise ValueError("CMC 恐惧贪婪指数缺少分类")
        timestamp = (
            record.get("update_time") if mode == "latest" else record.get("timestamp")
        )
        observed_at = utc_time(timestamp)  # 将数据源时间转为 UTC。
        normalized.append(
            {
                "value": value,
                "value_classification": classification,
                "observed_at": observed_at.isoformat(),
            }
        )
    return {
        "status": "ok",
        "source": f"{_BASE_URL}{path}",
        "mode": mode,
        "data": normalized[0] if mode == "latest" else normalized,
    }
