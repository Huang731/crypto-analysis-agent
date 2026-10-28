"""通过 CryptoETF REST 接口查询日度 ETF 净流量，返回中文 Markdown。"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytz
import requests
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.new_tools.errors import (
    DataSourceNotConfiguredError,
    DataSourceResponseError,
    DataSourceUnavailableError,
    DataSourceValidationError,
)

CRYPTO_ETF_API_BASE = "https://api.cryptoetf.today/api/v1"
CRYPTO_ETF_FLOWS_URL = f"{CRYPTO_ETF_API_BASE}/flows/{{asset}}"
CRYPTO_ETF_TZ = pytz.timezone("UTC")
REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40


class CryptoETFNotConfiguredError(DataSourceNotConfiguredError):
    """选用 CryptoETF 但未配置 API 密钥时抛出的异常。"""


class CryptoETFSettings(BaseSettings):
    """沿用已有密钥配置，进程环境变量优先于项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    crypto_etf_mcp_api_key: SecretStr = SecretStr("")


def get_api_key() -> str:
    """从已有 CRYPTO_ETF_MCP_API_KEY 配置读取服务方 API 密钥。"""
    api_key = CryptoETFSettings().crypto_etf_mcp_api_key.get_secret_value()
    if not api_key:
        raise CryptoETFNotConfiguredError(
            "未配置 CRYPTO_ETF_MCP_API_KEY，请在环境变量或 .env 中设置。"
        )
    return api_key


def _request(url: str) -> dict[str, Any]:
    """同步请求 CryptoETF 接口，通过请求头认证并检查 HTTP 状态。"""
    api_key = get_api_key()
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceUnavailableError(f"CryptoETF 请求失败：{exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise DataSourceResponseError("CryptoETF 响应不是有效 JSON") from exc
    return payload


def get_crypto_etf_flows(
    target_asset: str, look_back_days: int = DEFAULT_LOOKBACK_DAYS
) -> str:
    """查询目标资产的日度 ETF 净流量，金额统一为亿美元，默认回溯 365 天。"""
    asset = target_asset.lower()
    if look_back_days <= 0:
        raise DataSourceValidationError("look_back_days 必须大于 0")
    today = datetime.now(CRYPTO_ETF_TZ).date()
    start_date = today - timedelta(days=look_back_days)
    end_date = today - timedelta(days=1)
    url = CRYPTO_ETF_FLOWS_URL.format(asset=asset)
    payload = _request(url)
    days = payload["days"]
    window_days = payload["windowDays"]

    points: list[tuple[date, Decimal]] = []
    for point in days:
        day = date.fromisoformat(point["date"])
        amount = Decimal(str(point["netFlowUsdM"]))
        if start_date <= day < today:
            # 接口金额为百万美元，除以 100 后换算为亿美元，不改写零值或负值。
            points.append((day, amount / 100))
    points.sort(key=lambda point: point[0])

    header = (
        f"## CryptoETF：{asset.upper()} 现货 ETF 资金流\n"
        "- 频率：日度（UTC）\n"
        f"- 回溯窗口：{look_back_days} 天\n"
        f"- 查询日期：{start_date} 至 {end_date}\n"
        f"- 接口历史范围：最近 {window_days} 天，实际窗口以返回记录为准\n"
        "- 单位：亿美元\n"
        "- 正值表示净流入，负值表示净流出\n"
        f"- 来源：{url}\n"
    )
    if not points:
        return header + "\n查询窗口内没有可用记录。\n"

    first_day, first_value = points[0]
    last_day, last_value = points[-1]
    latest = (
        f"\n**最新记录：** 日期（UTC）：{last_day}；"
        f"净流量（亿美元）：{last_value:+.2f}\n"
    )
    summary = ""
    if len(points) > 1:
        # 净流量可能为负或为零，首末日仅比较金额差，不计算相对涨跌幅。
        delta = last_value - first_value
        summary = (
            f"\n**{first_day} 至 {last_day} 的单日净流量变化：** "
            f"{delta:+.2f} 亿美元，起始值为 {first_value:+.2f} 亿美元\n"
        )
    note = (
        f"\n共获取 {len(points)} 条记录，展示最近 {min(len(points), MAX_ROWS)} 条。\n"
    )
    table = "\n| 日期（UTC） | 净流量（亿美元） |\n| --- | --- |\n"
    table += "\n".join(
        f"| {day} | {amount:+.2f} |" for day, amount in points[-MAX_ROWS:]
    )
    return f"{header}{latest}{summary}{note}{table}\n"
