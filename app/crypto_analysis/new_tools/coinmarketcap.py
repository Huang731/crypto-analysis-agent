"""CoinMarketCap 稳定币市值与恐惧贪婪指数，按接口返回中文 Markdown。"""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytz
import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.new_tools.errors import (
    DataSourceNotConfiguredError,
    DataSourceResponseError,
    DataSourceUnavailableError,
    DataSourceValidationError,
)
from app.crypto_analysis.tools.source_time_utils import utc_time

CMC_API_BASE = "https://pro-api.coinmarketcap.com"
STABLECOIN_MARKET_CAP_URL = f"{CMC_API_BASE}/v3/cryptocurrency/quotes/historical"
FEAR_AND_GREED_LATEST_URL = f"{CMC_API_BASE}/v3/fear-and-greed/latest"
FEAR_AND_GREED_HISTORICAL_URL = f"{CMC_API_BASE}/v3/fear-and-greed/historical"
CMC_TZ = pytz.timezone("UTC")
REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40
STABLECOIN_IDS = {"USDT": "825", "USDC": "3408"}
FEAR_AND_GREED_CLASSIFICATIONS = {
    "extreme fear": "极度恐惧",
    "fear": "恐惧",
    "neutral": "中性",
    "greed": "贪婪",
    "extreme greed": "极度贪婪",
}


class CMCNotConfiguredError(DataSourceNotConfiguredError):
    """选用 CoinMarketCap 但未配置 API 密钥时抛出的异常。"""


class CMCSettings(BaseSettings):
    """读取 CoinMarketCap 配置，进程环境变量优先于项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cmc_pro_api_key: str = ""


def get_api_key() -> str:
    """读取 CoinMarketCap Pro API 密钥，配置缺失时明确报错。"""
    api_key = CMCSettings().cmc_pro_api_key
    if not api_key:
        raise CMCNotConfiguredError(
            "未配置 CMC_PRO_API_KEY，请在环境变量或 .env 中设置。"
        )
    return api_key


def _request(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """使用同步请求传递 CMC 认证请求头，检查 HTTP 与业务状态。"""
    api_key = get_api_key()
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"X-CMC_PRO_API_KEY": api_key},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceUnavailableError(f"CMC 请求失败：{exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise DataSourceResponseError("CMC 响应不是有效 JSON") from exc
    status = payload["status"]
    if status.get("error_code") not in (0, "0"):
        raise DataSourceUnavailableError(
            f"CMC API 错误 {status.get('error_code')}: {status.get('error_message')}"
        )
    return payload


def _format_value(value: Any) -> str:
    """市值保留两位小数，指数和分类保留原值。"""
    return f"{value:.2f}" if isinstance(value, Decimal) else str(value)


def _render_report(
    title: str,
    metadata: str,
    columns: tuple[str, ...],
    points: list[tuple[Any, ...]],
    source: str,
) -> str:
    """使用完整记录计算变化摘要，表格仅展示最近若干条。"""
    header = f"## CoinMarketCap：{title}\n{metadata}\n- 来源：{source}\n"
    if not points:
        return header + "\n查询窗口内没有可用记录。\n"

    latest = "；".join(
        f"{column}：{_format_value(value)}"
        for column, value in zip(columns, points[-1], strict=True)
    )
    summary = ""
    # 最新接口只有一条观测，不构造没有历史对照的变化值。
    if len(points) > 1:
        first_time, last_time = points[0][0], points[-1][0]
        first_value = Decimal(str(points[0][1]))
        last_value = Decimal(str(points[-1][1]))
        delta = last_value - first_value
        pct = f" ({delta / first_value * 100:+.2f}%)" if first_value != 0 else ""
        summary = (
            f"\n**{first_time} 至 {last_time} 的{columns[1]}变化：** "
            f"{delta:+.2f}{pct}，起始值为 {_format_value(points[0][1])}\n"
        )
    note = (
        f"\n共获取 {len(points)} 条记录，展示最近 {min(len(points), MAX_ROWS)} 条。\n"
    )
    table = "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *(
                "| " + " | ".join(_format_value(value) for value in point) + " |"
                for point in points[-MAX_ROWS:]
            ),
        ]
    )
    return f"{header}\n**最新记录：** {latest}\n{summary}{note}\n{table}\n"


def get_stablecoin_historical_market_cap(target_asset: str) -> str:
    """查询 USDT 或 USDC 最近 365 条日度市值，统一以亿美元展示。"""
    symbol = target_asset.upper()
    if symbol not in STABLECOIN_IDS:
        raise DataSourceValidationError("稳定币代码仅支持 USDT 或 USDC")
    coin_id = STABLECOIN_IDS[symbol]
    end = datetime.now(CMC_TZ).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(seconds=1)
    payload = _request(
        STABLECOIN_MARKET_CAP_URL,
        {
            "id": coin_id,
            "count": DEFAULT_LOOKBACK_DAYS,
            "interval": "daily",
            "time_end": end.isoformat(),
            "convert": "USD",
        },
    )
    data = payload["data"]
    points: list[tuple[Any, ...]] = []
    coin = data.get(coin_id)
    if coin is not None:
        for row in coin["quotes"]:
            usd = row["quote"]["USD"]
            market_cap = usd["market_cap"]
            if market_cap is None:
                continue
            value = Decimal(str(market_cap))
            observed_at = utc_time(usd["timestamp"])
            # 先换算单位再计算摘要，展示时才保留两位小数。
            points.append(
                (observed_at.strftime("%Y-%m-%d %H:%M:%S"), value / 100_000_000)
            )
    points.sort(key=lambda point: point[0])
    return _render_report(
        title=f"{symbol} 历史市值",
        metadata=(
            "- 频率：日度（UTC，每日首笔报价）\n"
            f"- 回溯窗口：最近 {DEFAULT_LOOKBACK_DAYS} 条日度记录\n"
            f"- 截止时点：{end:%Y-%m-%d %H:%M:%S} UTC\n"
            "- 单位：亿美元"
        ),
        columns=("观测时间（UTC）", "市值（亿美元）"),
        points=points,
        source=STABLECOIN_MARKET_CAP_URL,
    )


def _fear_and_greed_points(
    records: list[Any], timestamp_field: str
) -> list[tuple[Any, ...]]:
    """提取指数、中文情绪分类与 UTC 观测时间。"""
    points: list[tuple[Any, ...]] = []
    for record in records:
        value = record["value"]
        classification_key = record["value_classification"].strip().casefold()
        observed_at = utc_time(record[timestamp_field])
        points.append(
            (
                observed_at.strftime("%Y-%m-%d %H:%M:%S"),
                value,
                FEAR_AND_GREED_CLASSIFICATIONS[classification_key],
            )
        )
    return sorted(points, key=lambda point: point[0])


def get_fear_and_greed_latest() -> str:
    """查询最新的全市场恐惧贪婪指数，无需传入币种或时间参数。"""
    data = _request(FEAR_AND_GREED_LATEST_URL, {})["data"]
    points = _fear_and_greed_points([data], "update_time") if data else []
    return _render_report(
        title="最新恐惧贪婪指数",
        metadata="- 指标范围：全市场\n- 指数范围：0 至 100",
        columns=("观测时间（UTC）", "恐惧贪婪指数", "情绪分类"),
        points=points,
        source=FEAR_AND_GREED_LATEST_URL,
    )


def get_fear_and_greed_historical() -> str:
    """查询最近 365 条日度恐惧贪婪指数，无需传入币种或时间参数。"""
    data = _request(
        FEAR_AND_GREED_HISTORICAL_URL,
        {"start": 1, "limit": DEFAULT_LOOKBACK_DAYS},
    )["data"]
    return _render_report(
        title="历史恐惧贪婪指数",
        metadata=(
            "- 指标范围：全市场\n- 频率：日度（UTC）\n"
            f"- 回溯窗口：最近 {DEFAULT_LOOKBACK_DAYS} 条日度记录\n"
            "- 指数范围：0 至 100"
        ),
        columns=("观测时间（UTC）", "恐惧贪婪指数", "情绪分类"),
        points=_fear_and_greed_points(data, "timestamp"),
        source=FEAR_AND_GREED_HISTORICAL_URL,
    )
