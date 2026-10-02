"""FRED 宏观数据工具。"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.tools import tool
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.source_time_utils import (
    shift_month,  # 只用于构造 FRED 月度请求窗口。
)

_ROOT = Path(__file__).resolve().parents[3]  # 数据源目录比项目根目录多一层。
_FRED_SOURCE = "https://api.stlouisfed.org/fred/series/observations"


class FREDSettings(BaseSettings):
    """读取 FRED API 密钥。"""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    fred_api_key: str = ""


@tool
async def get_m2_yoy_change(series_id: Literal["M2SL", "WALCL"], as_of: str) -> dict[str, Any]:  # 获取固定序列的原始观测。
    """查询 FRED 的 M2SL 或 WALCL 序列，返回原始观测与来源。"""
    api_key = FREDSettings().fred_api_key
    if not api_key:
        raise RuntimeError("未配置 FRED_API_KEY")
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    today = cutoff.astimezone(timezone.utc).date()
    if series_id == "M2SL":
        first_day = date(today.year, today.month, 1)
        start = shift_month(first_day, -18)  # 保留既有的 18 个月请求范围。
    else:
        start = today - timedelta(days=90)
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": today.isoformat(),
        "units": "lin",
        "sort_order": "asc",
        "limit": 1000,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(_FRED_SOURCE, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"FRED API HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("FRED 响应不是 JSON 对象")
    return {"payload": payload, "source": _FRED_SOURCE}  # 交由程序计算器校验并评分。
