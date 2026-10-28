"""FRED 宏观经济数据源，从环境变量或 .env 读取 FRED_API_KEY。"""

from datetime import datetime, timedelta
from pathlib import Path

import pytz
import requests
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.new_tools.errors import (
    DataSourceNotConfiguredError,
    DataSourceResponseError,
    DataSourceUnavailableError,
    DataSourceValidationError,
)
from app.crypto_analysis.new_tools.net import get_with_redacted_errors

FRED_API_BASE = "https://api.stlouisfed.org/fred"

FRED_TZ = pytz.timezone("UTC")

REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40

# 常用指标别名。
MACRO_SERIES = {
    # 政策利率与国债收益率
    "fed_funds_rate": "FEDFUNDS",
    "federal_funds_rate": "FEDFUNDS",
    "fed_funds": "FEDFUNDS",
    "2y_treasury": "DGS2",
    "10y_treasury": "DGS10",
    "30y_treasury": "DGS30",
    "10y_2y_spread": "T10Y2Y",
    "yield_curve": "T10Y2Y",
    # 通货膨胀
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "inflation_expectations": "T10YIE",
    # 经济增长与产出
    "real_gdp": "GDPC1",
    "gdp": "GDP",
    "industrial_production": "INDPRO",
    # 劳动力市场
    "unemployment_rate": "UNRATE",
    "unemployment": "UNRATE",
    "nonfarm_payrolls": "PAYEMS",
    "payrolls": "PAYEMS",
    "initial_claims": "ICSA",
    # 货币与市场
    "m2": "M2SL",
    "money_supply": "M2SL",
    "vix": "VIXCLS",
    "dollar_index": "DTWEXBGS",
    # 消费者情绪与住房
    "consumer_sentiment": "UMCSENT",
    "housing_starts": "HOUST",
    "retail_sales": "RSAFS",
    # 欧元区指标
    "ecb_deposit_rate": "ECBDFR",
    "ecb_main_refi_rate": "ECBMRRFR",
    "euro_hicp": "CP0000EZ19M086NEST",
    "euro_core_hicp": "00XEFDEZ19M086NEST",
    "euro_real_gdp": "CLVMNACSCAB1GQEA19",
    "germany_10y": "IRLTLT01DEM156N",
    "france_10y": "IRLTLT01FRM156N",
    "eur_usd": "DEXUSEU",
}


class FredNotConfiguredError(DataSourceNotConfiguredError):
    """未配置 FRED API 密钥。"""


class FREDSettings(BaseSettings):
    """读取 FRED 配置，进程环境变量优先于项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fred_api_key: str = ""


def get_api_key() -> str:
    """读取 API 密钥，未配置时抛出异常。"""
    api_key = FREDSettings().fred_api_key
    if not api_key:
        raise FredNotConfiguredError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return api_key


def _resolve_series_id(indicator: str) -> str:
    """解析指标别名或原始序列 ID。"""
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MACRO_SERIES:
        return MACRO_SERIES[key]
    candidate = indicator.strip().upper()
    if not candidate or len(candidate) > 30 or any(c.isspace() for c in candidate):
        raise DataSourceValidationError(
            f"'{indicator}' is not a known macro alias or a valid FRED series ID. "
            f"Use an alias (e.g. 'cpi', 'unemployment', '10y_treasury') or a raw "
            f"FRED series ID (e.g. 'CPIAUCSL')."
        )
    return candidate


def _fred_today() -> str:
    """返回当前 UTC 日期。"""
    return datetime.now(FRED_TZ).strftime("%Y-%m-%d")


def _request(path: str, params: dict) -> dict:
    """请求 FRED 数据并检查响应状态。"""
    api_key = get_api_key()
    try:
        response = get_with_redacted_errors(
            f"{FRED_API_BASE}/{path}",
            params={**params, "api_key": api_key, "file_type": "json"},
            timeout=REQUEST_TIMEOUT,
            secret=api_key,
            allowed_status_codes=(400,),
        )
    except requests.RequestException as exc:
        raise DataSourceUnavailableError(f"FRED 请求失败：{exc}") from exc
    if response.status_code == 400:
        try:
            message = response.json().get("error_message", response.text)
        except ValueError:
            message = response.text
        raise DataSourceValidationError(f"FRED request failed: {message}")
    try:
        return response.json()
    except ValueError as exc:
        raise DataSourceResponseError("FRED 响应不是有效 JSON") from exc


def get_macro_data(
    indicator: str,
    as_of_date: str,
    look_back_days: int | None = None,
) -> str:
    """获取宏观序列并生成包含最新值、窗口变化和观测表格的 Markdown 报告。

    indicator 支持别名或序列 ID；as_of_date 格式为 YYYY-MM-DD。
    look_back_days 未指定时回溯 365 天，历史查询固定数据版本。
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=FRED_TZ)
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    # 固定历史数据版本，避免使用后续修订值。
    pit = min(as_of_date, _fred_today())
    realtime = {"realtime_start": pit, "realtime_end": pit}

    try:
        series_id = _resolve_series_id(indicator)
    except ValueError as e:
        return f"FRED 指标参数无效：{e}"

    meta = _request("series", {"series_id": series_id, **realtime}).get("seriess") or []
    if not meta:
        return (
            f"未找到 FRED 序列 '{series_id}'。请使用已知指标别名"
            f"（例如 'cpi'、'unemployment'）或有效的 FRED 序列 ID。"
        )
    info = meta[0]
    title = info.get("title", series_id)
    units = info.get("units_short") or info.get("units", "")
    frequency = info.get("frequency", "")
    seasonal = info.get("seasonal_adjustment_short", "")

    observations = _request(
        "series/observations",
        {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": as_of_date,
            "sort_order": "asc",
            **realtime,
        },
    ).get("observations", [])

    # FRED 使用 "." 表示缺失的观测值。
    points = [
        (o["date"], o["value"])
        for o in observations
        if o.get("value") not in (".", None, "")
    ]

    header = (
        f"## FRED: {title} ({series_id})\n"
        f"- 单位：{units}\n"
        f"- 频率：{frequency}"
        f"{f' ({seasonal})' if seasonal else ''}\n"
        f"- 数据窗口：{start_date} 至 {as_of_date}\n"
    )

    if not points:
        return header + (
            f"\n在截至 {pit} 的数据版本中，序列 {series_id} 在此窗口内没有观测值。"
            f"可能是序列发布间隔长于查询窗口（可尝试增大 look_back_days），"
            f"也可能是当时尚无已发布的数据版本"
            f"（截至 {pit} 尚未发布，或早于 ALFRED 历史版本的覆盖起点）。"
        )

    first_date, first_val = points[0]
    last_date, last_val = points[-1]
    try:
        delta = float(last_val) - float(first_val)
        base = float(first_val)
        pct = f" ({delta / base * 100:+.2f}%)" if base != 0 else ""
        summary = (
            f"\n**最新值：** {last_val} ({last_date}) | "
            f"**{first_date} 至 {last_date} 的变化：** {delta:+.2f}{pct}，"
            f"起始值为 {first_val}\n"
        )
    except ValueError:
        summary = f"\n**最新值：** {last_val} ({last_date})\n"

    shown = points
    note = ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        note = f"\n_（共 {len(points)} 条观测，仅展示最近 {MAX_ROWS} 条）_\n"

    table = (
        "\n| 日期 | 数值 |\n| --- | --- |\n"
        + "\n".join(f"| {d} | {v} |" for d, v in shown)
        + "\n"
    )

    return header + summary + note + table
