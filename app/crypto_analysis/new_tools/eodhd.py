"""EODHD 历史日线数据源。

通过 EODHD 日线 API 获取指定标的的历史收盘价，返回中文 Markdown 报告。
从环境变量或项目根目录的 .env 读取 EODHD_API_KEY。
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.new_tools.errors import (
    DataSourceNotConfiguredError,
    DataSourceResponseError,
    DataSourceUnavailableError,
    DataSourceValidationError,
)
from app.crypto_analysis.new_tools.net import get_with_redacted_errors

EODHD_API_BASE = "https://eodhd.com/api"

# 与 FRED 数据源保持一致的请求超时、默认回溯窗口和报告展示行数。
REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40

# 常用指标别名到 EODHD 标的代码的映射，其他输入直接作为标的代码使用。
MARKET_SYMBOLS = {
    "dxy": "DXY.INDX",
}


class EODHDNotConfiguredError(DataSourceNotConfiguredError):
    """选用 EODHD 但未配置 API 密钥时抛出的异常。"""


class EODHDSettings(BaseSettings):
    """读取 EODHD 配置，进程环境变量优先于项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    eodhd_api_key: str = ""


def get_api_key() -> str:
    """从进程环境变量或项目根目录的 .env 读取 EODHD API 密钥。"""
    api_key = EODHDSettings().eodhd_api_key
    if not api_key:
        raise EODHDNotConfiguredError(
            "未配置 EODHD_API_KEY，请在进程环境变量或项目根目录的 .env 中设置。"
        )
    return api_key


def _resolve_symbol(indicator: str) -> str:
    """将指标别名映射为 EODHD 标的代码，或直接返回原始代码。"""
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MARKET_SYMBOLS:
        return MARKET_SYMBOLS[key]
    candidate = indicator.strip().upper()
    if not candidate or len(candidate) > 30 or any(c.isspace() for c in candidate):
        raise DataSourceValidationError(
            f"'{indicator}' 不是已知指标别名或有效的 EODHD 标的代码。"
            "请使用 'dxy' 或完整标的代码，例如 'DXY.INDX'。"
        )
    return candidate


def _request(path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """通过 GET 请求 EODHD 接口，请求无效时展示响应中的错误信息。"""
    api_key = get_api_key()
    try:
        response = get_with_redacted_errors(
            f"{EODHD_API_BASE}/{path}",
            params={**params, "api_token": api_key, "fmt": "json"},
            timeout=REQUEST_TIMEOUT,
            secret=api_key,
            allowed_status_codes=(400,),
        )
    except requests.RequestException as exc:
        raise DataSourceUnavailableError(f"EODHD 请求失败：{exc}") from exc
    if response.status_code == 400:
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        raise DataSourceValidationError(f"EODHD 请求失败：{message}")
    try:
        return response.json()
    except ValueError as exc:
        raise DataSourceResponseError("EODHD 响应不是有效 JSON") from exc


def get_market_data(
    indicator: str,
    as_of_date: str,
    look_back_days: int | None = None,
) -> str:
    """获取 EODHD 历史收盘价，返回格式化的中文 Markdown 报告。

    参数：
        indicator: 指标别名 'dxy' 或完整 EODHD 标的代码，例如 'DXY.INDX'。
        as_of_date: 数据截止日期，格式为 yyyy-mm-dd，用于限定历史行情窗口。
        look_back_days: 回溯窗口的天数；为 None 时使用 DEFAULT_LOOKBACK_DAYS。

    返回：
        包含标的代码、数据窗口、最新收盘价、窗口内变化量及近期收盘价表格的报告。
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_date = date.fromisoformat(as_of_date)
    start_date = (end_date - timedelta(days=look_back_days)).isoformat()

    try:
        symbol = _resolve_symbol(indicator)
    except ValueError as exc:
        return f"EODHD 指标参数无效：{exc}"

    observations = _request(
        f"eod/{symbol}",
        {"from": start_date, "to": as_of_date, "period": "d", "order": "a"},
    )
    points = [
        (row["date"], row["close"])
        for row in observations
        if row.get("close") not in (".", None, "")
    ]

    header = (
        f"## EODHD: {symbol}\n- 频率：日度\n- 数据窗口：{start_date} 至 {as_of_date}\n"
    )
    if not points:
        return header + f"\n标的 {symbol} 在此窗口内没有可用的收盘价观测值。"

    first_date, first_val = points[0]
    last_date, last_val = points[-1]
    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**最新收盘价：** {last_val} ({last_date}) | "
            f"**{first_date} 至 {last_date} 的变化：** {delta:+.2f}{pct}，"
            f"起始值为 {first_val}\n"
        )
    except ValueError:
        summary = f"\n**最新收盘价：** {last_val} ({last_date})\n"

    shown = points
    note = ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        note = f"\n_（共 {len(points)} 条观测，仅展示最近 {MAX_ROWS} 条）_\n"

    table = (
        "\n| 日期 | 收盘价 |\n| --- | --- |\n"
        + "\n".join(f"| {day} | {value} |" for day, value in shown)
        + "\n"
    )
    return header + summary + note + table
