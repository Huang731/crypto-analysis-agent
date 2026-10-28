"""通过 yfinance 查询目标币种新闻，按 UTC 时间筛选并返回中文 Markdown。"""

import contextlib
import time
from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta
from typing import Any

import pytz
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from app.core.observability import get_logger
from app.crypto_analysis.new_tools.errors import (
    DataSourceUnavailableError,
    DataSourceValidationError,
)

log = get_logger("crypto_analysis.news")

NEWS_TZ = pytz.timezone("UTC")
DEFAULT_LOOKBACK_DAYS = 365
MAX_NEWS_COUNT = 20
YAHOO_NEWS_URL = "https://finance.yahoo.com/quote/{symbol}/news/"


def _extract_article_data(article: dict[str, Any]) -> dict[str, Any]:
    """解析 Yahoo 新闻，兼容嵌套 content 和旧版扁平结构。

    可选字段既可能缺失，也可能返回 null，保留原有缺失字段的展示方式。
    """
    content = article.get("content")
    if isinstance(content, dict):
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        pub_date_str = content.get("pubDate") or ""
        pub_date = None
        if pub_date_str:
            with contextlib.suppress(ValueError, AttributeError):
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))

        return {
            "title": content.get("title") or "无标题",
            "summary": content.get("summary") or "",
            "publisher": (content.get("provider") or {}).get("displayName")
            or "未知来源",
            "link": url_obj.get("url") or "",
            "pub_date": pub_date,
        }
    # 扁平结构也解析发布时间，避免新闻绕过日期筛选。
    pub_date = None
    timestamp = article.get("providerPublishTime")
    if timestamp:
        # Unix 秒时间戳按 UTC 解析，避免受运行机器的本地时区影响。
        with contextlib.suppress(ValueError, OSError, TypeError):
            pub_date = datetime.fromtimestamp(timestamp, tz=NEWS_TZ)
    return {
        "title": article.get("title") or "无标题",
        "summary": article.get("summary") or "",
        "publisher": article.get("publisher") or "未知来源",
        "link": article.get("link") or "",
        "pub_date": pub_date,
    }


def _format_articles(articles: list[dict[str, Any]]) -> str:
    """格式化新闻，保留原始标题、摘要和链接。"""
    output = ""
    for data in articles:
        output += f"### {data['title']}（来源：{data['publisher']}）\n"
        if data["pub_date"] is not None:
            observed_at = to_utc(data["pub_date"])
            output += f"- 发布时间：{observed_at:%Y-%m-%d %H:%M:%S} UTC\n"
        if data["summary"]:
            output += f"{data['summary']}\n"
        if data["link"]:
            output += f"链接：{data['link']}\n"
        output += "\n"
    return output


def get_news_yfinance(target_asset: str) -> str:
    """查询 BTC、ETH 等币种的近期新闻，自动按 UTC 时间回溯 365 天筛选。"""
    asset = target_asset.strip().upper()
    if not asset or not asset.isascii() or not asset.isalnum():
        raise DataSourceValidationError("target_asset 必须是 BTC、ETH 等币种代码")
    symbol = f"{asset}-USD"
    now = datetime.now(NEWS_TZ)
    end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    source = YAHOO_NEWS_URL.format(symbol=symbol)
    header = (
        f"## Yahoo Finance：{asset} 新闻\n"
        f"- 查询代码：{symbol}\n"
        f"- 回溯窗口：{DEFAULT_LOOKBACK_DAYS} 天\n"
        f"- 时间窗口：{start_date} 至 {end_date} UTC\n"
        f"- 数据范围：最近最多 {MAX_NEWS_COUNT} 条新闻样本，非完整历史新闻\n"
        f"- 来源：{source}\n"
    )
    feed = yf_retry(lambda: yf.Ticker(symbol).get_news(count=MAX_NEWS_COUNT))
    articles = [_extract_article_data(article) for article in feed]
    if not feed:
        found, answered = _search_news(symbol, MAX_NEWS_COUNT)
        if not answered:
            raise DataSourceUnavailableError(
                f"Yahoo Finance 未返回 {symbol} 的新闻记录，无法确认新闻数据是否可用"
            )
        if not found:
            raise DataSourceUnavailableError(
                f"Yahoo Finance 未返回标记为 {symbol} 的新闻记录"
            )
        articles = [_extract_article_data(article) for article in found]
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=NEWS_TZ)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=NEWS_TZ)
    # 仅保留请求日期窗口内的新闻，避免引入窗口之后的新闻。
    in_range = [
        article
        for article in articles
        if in_window(article["pub_date"], start_dt, end_dt)
    ]
    if in_range:
        return (
            header
            + f"\n共筛选出 {len(in_range)} 条新闻。\n\n"
            + _format_articles(in_range)
        )

    if not feed:
        return (
            header + f"\nYahoo Finance 新闻未覆盖 {start_date} 至 {end_date}："
            f"Yahoo 搜索仅返回 {symbol} 的近期新闻样本，不能据此认定没有 {asset} 新闻。"
        )
    gap = coverage_gap(
        (article["pub_date"] for article in articles),
        start_date,
        end_date,
        "Yahoo Finance 新闻",
        f"{asset} 新闻",
    )
    return (
        header
        + "\n"
        + (gap or f"返回样本中没有 {start_date} 至 {end_date} 的 {asset} 新闻。")
    )


def _search_news(canonical: str, limit: int) -> tuple[list[dict[str, Any]], bool]:
    """查询标记为指定代码的新闻，并判断 Yahoo 是否返回了相关响应。

    新闻流为空时使用搜索结果。搜索返回按相关性选取的样本，部分新闻关联较宽泛，
    因此仅保留明确标记了指定代码的新闻。
    """
    search = yf_retry(lambda: yf.Search(canonical, news_count=limit))
    if search is None:  # 搜索没有返回响应，视为搜索服务不可用。
        return [], False
    tagged = [
        a
        for a in (search.news or [])
        if canonical in (t.upper() for t in a.get("relatedTickers") or [])
    ]
    knows = any(
        (q.get("symbol") or "").upper() == canonical for q in search.quotes or []
    )
    return tagged, knows or bool(search.news)


def to_utc(value: datetime) -> datetime:
    """统一转换为带 UTC 时区的时间，无时区值按原约定视为 UTC。"""
    return (
        value.replace(tzinfo=NEWS_TZ)
        if value.tzinfo is None
        else value.astimezone(NEWS_TZ)
    )


def in_window(pub_dt: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """判断新闻是否位于起始日到结束日的完整日期窗口。

    保留原有规则：只有窗口覆盖当前日期时，才允许没有发布时间的新闻。
    """
    end = to_utc(end_dt)
    if pub_dt is not None:
        return to_utc(start_dt) <= to_utc(pub_dt) < end + timedelta(days=1)
    return end >= datetime.now(NEWS_TZ) - timedelta(days=1)


def coverage_gap(
    dates: Iterable[datetime | None],
    start_date: str,
    end_date: str,
    source: str,
    subject: str,
) -> str | None:
    """检查新闻样本是否覆盖请求窗口，避免将未覆盖窗口误判为没有新闻。

    日期序列用于确定最早覆盖时间；没有可确认日期时，以当前时间为边界。
    """
    now = datetime.now(NEWS_TZ)
    oldest = min((to_utc(value) for value in dates if value is not None), default=now)
    if date.fromisoformat(end_date) > now.date():
        reason = "查询窗口超出了当前日期"
    elif oldest.date() > date.fromisoformat(start_date):
        reason = "接口仅返回近期新闻样本"
    else:
        return None
    return f"{source}未完整覆盖 {start_date} 至 {end_date}：{reason}，不能据此认定没有{subject}。"


def yf_retry(
    func: Callable[[], Any], max_retries: int = 3, base_delay: float = 2.0
) -> Any:
    """保留限流时的指数退避重试，请求失败时明确抛出数据源异常。

    每次重试重新执行传入的调用，避免复用已经缓存失败结果的 Ticker 实例。
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError as exc:
            if attempt >= max_retries:
                raise DataSourceUnavailableError(
                    f"Yahoo Finance 限流，重试 {max_retries} 次后仍然失败"
                ) from exc
            delay = base_delay * (2**attempt)
            log.warning(
                "Yahoo Finance 请求限流，%.0f 秒后重试（第 %d/%d 次）",
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)
        except Exception as exc:
            raise DataSourceUnavailableError(
                f"Yahoo Finance 请求失败：{type(exc).__name__}"
            ) from exc
    raise DataSourceValidationError("max_retries 必须大于或等于 0")
