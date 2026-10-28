"""Binance 公开行情工具，每个工具对应一个接口并返回中文 Markdown。"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytz
import requests

from app.crypto_analysis.new_tools.errors import (
    DataSourceResponseError,
    DataSourceUnavailableError,
    DataSourceValidationError,
)
from app.crypto_analysis.new_tools.net import get_with_redacted_errors

SPOT_API_BASE = "https://api.binance.com"
FUTURES_API_BASE = "https://fapi.binance.com"
BINANCE_TZ = pytz.timezone("UTC")

REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40


def _request(url: str, params: dict[str, Any]) -> list[Any]:
    """复用现有同步请求方法，保留接口异常。"""
    try:
        response = get_with_redacted_errors(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            secret="",
        )
    except requests.RequestException as exc:
        raise DataSourceUnavailableError(f"Binance 请求失败：{exc}") from exc
    try:
        rows = response.json()
    except ValueError as exc:
        raise DataSourceResponseError("Binance 响应不是有效 JSON") from exc
    if not isinstance(rows, list):
        raise DataSourceResponseError("Binance 响应不是数组")
    return rows


def _cutoff_ms() -> int:
    """返回当前 UTC 日期零点的毫秒时间戳。"""
    cutoff = datetime.now(BINANCE_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(cutoff.timestamp() * 1000)


def _utc_time(timestamp: int) -> str:
    """将毫秒时间戳转换为 UTC 时间。"""
    return datetime.fromtimestamp(
        timestamp / 1000,
        tz=BINANCE_TZ,
    ).strftime("%Y-%m-%d %H:%M:%S")


def _render_report(
    title: str,
    metadata: str,
    columns: tuple[str, ...],
    points: list[tuple[Any, ...]],
    source: str,
    value_column: int,
    percentage_points: bool = False,
) -> str:
    """使用完整记录计算首末变化，表格仅展示最近若干条。"""
    header = f"## Binance：{title}\n{metadata}\n- 来源：{source}\n"

    if not points:
        return header + "\n查询窗口内没有可用记录。\n"

    display_points = points
    if percentage_points:
        display_points = [
            tuple(
                f"{(Decimal(str(value)) * 100).normalize():f}%"
                if index == value_column
                else value
                for index, value in enumerate(point)
            )
            for point in points
        ]

    latest = "；".join(
        f"{column}：{value}"
        for column, value in zip(columns, display_points[-1], strict=True)
    )
    first_time, last_time = points[0][0], points[-1][0]
    first_value = Decimal(str(points[0][value_column]))
    last_value = Decimal(str(points[-1][value_column]))
    delta = last_value - first_value
    if percentage_points:
        change = f"{delta * 100:+.6f} 个百分点"
    else:
        pct = f" ({delta / first_value * 100:+.2f}%)" if first_value != 0 else ""
        change = f"{delta:+.2f}{pct}"
    summary = (
        f"\n**{first_time} 至 {last_time} 的{columns[value_column]}变化：** "
        f"{change}，起始值为 {display_points[0][value_column]}\n"
    )
    note = (
        f"\n共获取 {len(points)} 条记录，展示最近 {min(len(points), MAX_ROWS)} 条。\n"
    )
    table_header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    table_rows = [
        "| " + " | ".join(str(value) for value in point) + " |"
        for point in display_points[-MAX_ROWS:]
    ]
    table = "\n".join([table_header, separator, *table_rows])
    return f"{header}\n**最新记录：** {latest}\n{summary}{note}\n{table}\n"


def get_binance_klines(target_asset: str) -> str:
    """查询截至当前 UTC 零点的已收盘现货日线，默认回溯 365 天。"""
    symbol = f"{target_asset.upper()}USDT"
    end = _cutoff_ms()
    url = f"{SPOT_API_BASE}/api/v3/klines"

    rows = _request(
        url,
        {
            "symbol": symbol,
            "interval": "1d",
            "limit": DEFAULT_LOOKBACK_DAYS,
            "endTime": end - 1,
            "timeZone": "0",
        },
    )
    points = [
        (_utc_time(int(row[0])), row[1], row[2], row[3], row[4], row[5])
        for row in sorted(rows, key=lambda row: int(row[0]))
    ]
    return _render_report(
        title=f"{symbol} 现货 K 线",
        metadata=(
            "- 周期：1d\n"
            f"- 回溯窗口：最近 {DEFAULT_LOOKBACK_DAYS} 天\n"
            f"- 收盘截止：{_utc_time(end)} UTC（不含）"
        ),
        columns=("开盘时间（UTC）", "开盘价", "最高价", "最低价", "收盘价", "成交量"),
        points=points,
        source=url,
        value_column=4,
    )


def get_binance_open_interest(
    target_asset: str, look_back_days: int = DEFAULT_LOOKBACK_DAYS
) -> str:
    """默认请求 365 条日度持仓量采样，实际历史受接口最近一个月的限制。"""
    if look_back_days <= 0:
        raise DataSourceValidationError("look_back_days 必须大于 0")
    symbol = f"{target_asset.upper()}USDT"
    end = _cutoff_ms()
    url = f"{FUTURES_API_BASE}/futures/data/openInterestHist"

    rows = _request(
        url,
        {
            "symbol": symbol,
            "period": "1d",
            "limit": look_back_days,
            "endTime": end,
        },
    )
    points = [
        (
            _utc_time(int(row["timestamp"])),
            row["sumOpenInterest"],
            row["sumOpenInterestValue"],
        )
        for row in sorted(rows, key=lambda row: int(row["timestamp"]))
    ]
    return _render_report(
        title=f"{symbol} 历史持仓量",
        metadata=(
            "- 采样周期：1 天\n"
            f"- 回溯窗口：最近 {look_back_days} 天\n"
            f"- 截止时点：{_utc_time(end)} UTC（含）\n"
            "- 数据范围：接口仅保留最近一个月，实际窗口以返回记录为准"
        ),
        columns=("采样时间（UTC）", "持仓量", "持仓名义价值（USDT）"),
        points=points,
        source=url,
        value_column=1,
    )


def get_binance_funding_rates(
    target_asset: str, look_back_days: int = DEFAULT_LOOKBACK_DAYS
) -> str:
    """按完整 UTC 日查询并累计资金费率，回溯天数默认 365。"""
    if look_back_days <= 0:
        raise DataSourceValidationError("look_back_days 必须大于 0")
    symbol = f"{target_asset.upper()}USDT"
    end = _cutoff_ms()
    start = end - int(timedelta(days=look_back_days).total_seconds() * 1000)
    url = f"{FUTURES_API_BASE}/fapi/v1/fundingRate"
    rows: list[dict[str, Any]] = []

    while True:
        page = _request(
            url,
            {
                "symbol": symbol,
                "startTime": start,
                "endTime": end - 1,
                "limit": 1000,
            },
        )
        rows.extend(page)
        if len(page) < 1000:
            break

        next_start = int(page[-1]["fundingTime"]) + 1
        if next_start <= start:
            raise DataSourceValidationError("Funding 分页时间未推进")
        if next_start >= end:
            break
        start = next_start

    # 按实际结算时间归入 UTC 自然日，不假定每天固定结算三次。
    daily_rates: dict[str, list[Decimal]] = {}
    for row in rows:
        day = (
            datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=BINANCE_TZ)
            .date()
            .isoformat()
        )
        daily_rates.setdefault(day, []).append(Decimal(str(row["fundingRate"])))
    points = [
        (day, sum(rates, Decimal(0))) for day, rates in sorted(daily_rates.items())
    ]
    return _render_report(
        title=f"{symbol} 历史资金费率",
        metadata=(
            "- 频率：日度（UTC）\n"
            f"- 回溯窗口：{look_back_days} 个完整日\n"
            f"- 截止边界：{_utc_time(end)} UTC（不含）\n"
            "- 统计口径：当天各次结算资金费率求和"
        ),
        columns=("日期（UTC）", "当日累计资金费率"),
        points=points,
        source=url,
        value_column=1,
        percentage_points=True,
    )
