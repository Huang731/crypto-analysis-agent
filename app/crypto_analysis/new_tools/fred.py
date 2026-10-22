"""FRED（美联储经济数据）宏观数据源。

通过圣路易斯联储的免费 API 获取政策利率、国债收益率、通胀、就业和经济增长等
宏观经济时间序列，供新闻分析师使用实际数据支撑宏观分析，而非仅依据新闻标题。

从环境变量或项目根目录的 .env 读取 ``FRED_API_KEY``，申请地址为：
https://fred.stlouisfed.org/docs/api/api_key.html 。未配置密钥时抛出
``FredNotConfiguredError``，使路由层将该数据源视为不可用，避免直接中断运行。
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytz
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.crypto_analysis.new_tools.errors import DataSourceNotConfiguredError
from app.crypto_analysis.new_tools.net import get_with_redacted_errors

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"

# FRED 的实时日期采用美国中部时间（圣路易斯联储所在地）。
# 超过其当前日期的实时日期会触发 HTTP 400，因此数据版本的截止日期不能晚于该日期，
# 而不是直接使用调用方的本地日期（#1275）。已有依赖 pytz 自带时区数据库，
# 因此即使系统缺少 tzdata，也能正常工作。
FRED_TZ = pytz.timezone("America/Chicago")

# 网络请求超时时间（秒），避免请求停滞导致 Agent 一直等待，与 Alpha Vantage 客户端一致。
REQUEST_TIMEOUT = 30

# 调用方未指定时使用的默认回溯窗口。一年的数据可覆盖大多数月度、季度序列的趋势和同比基期。
DEFAULT_LOOKBACK_DAYS = 365

# 渲染表格的最大行数：近期数值对决策最有参考价值，
# 限制行数可避免较长窗口内的日度序列（如收益率、VIX）占用过多上下文。
MAX_ROWS = 40

# 常用指标别名到 FRED 序列 ID 的映射。未列出的输入将直接作为原始序列 ID 使用，
# 因此熟悉 FRED 的用户不受此映射范围限制。
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
    # 欧元区指标，适用于欧洲交易所 .PA 等非美国证券代码。FRED 收录欧洲央行政策利率、
    # 欧盟统计局的调和消费者价格指数（HICP）与 GDP，以及 OECD 长期国债收益率，
    # 因此同样可以通过固定数据版本，确保仅使用指定时点已知的欧元区数据。
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
    """选用 FRED 但未配置 API 密钥时抛出的异常。

    继承 DataSourceNotConfiguredError，因此也属于 ValueError，
    可同时兼容路由层的数据源不可用处理和调用方已有的 ValueError 异常处理。
    """


class FREDSettings(BaseSettings):
    """读取 FRED 配置，进程环境变量优先于项目根目录的 .env。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fred_api_key: str = ""


def get_api_key() -> str:
    """从进程环境变量或项目根目录的 .env 读取 FRED API 密钥。"""
    api_key = FREDSettings().fred_api_key
    if not api_key:
        raise FredNotConfiguredError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return api_key


def _resolve_series_id(indicator: str) -> str:
    """将常用指标别名映射为 FRED 序列 ID，或直接返回原始序列 ID。

    输入既不是已知别名，也不符合序列 ID 的基本要求时，抛出 ``ValueError``。
    这通常是因为 LLM 传入了描述性短语，例如 "bank of japan rate"（日本央行利率）。
    FRED 序列 ID 通常较短，由字母和数字组成，因此提前拒绝此类输入并给出提示，
    避免向 API 发送会触发 HTTP 400 的请求。
    """
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_")
    if key in MACRO_SERIES:
        return MACRO_SERIES[key]
    candidate = indicator.strip().upper()
    # FRED 序列 ID 不含空白字符且长度较短；提前拒绝不符合要求的输入
    # （例如 LLM 传入的描述性短语），避免触发 API 的 HTTP 400 错误。
    if not candidate or len(candidate) > 30 or any(c.isspace() for c in candidate):
        raise ValueError(
            f"'{indicator}' is not a known macro alias or a valid FRED series ID. "
            f"Use an alias (e.g. 'cpi', 'unemployment', '10y_treasury') or a raw "
            f"FRED series ID (e.g. 'CPIAUCSL')."
        )
    return candidate


def _fred_today() -> str:
    """返回 FRED 当前的日历日期（美国中部时间），格式为 ``yyyy-mm-dd``。

    数据版本的截止日期不能晚于该日期：如果 ``realtime_start`` 超过 FRED 当天的日期，
    FRED 会返回 HTTP 400。实时运行时，``as_of_date`` 来自调用方的本地时钟，
    可能已经比芝加哥当地日期早进入下一天。
    """
    return datetime.now(FRED_TZ).strftime("%Y-%m-%d")


def _request(path: str, params: dict) -> dict:
    """通过 GET 请求 FRED 接口，请求无效时展示其 JSON 响应中的错误信息。"""
    api_key = get_api_key()
    response = get_with_redacted_errors(
        f"{FRED_API_BASE}/{path}",
        params={**params, "api_key": api_key, "file_type": "json"},
        timeout=REQUEST_TIMEOUT,
        secret=api_key,
        allowed_status_codes=(400,),
    )
    # 序列 ID 未知或参数格式错误时，FRED 返回 HTTP 400 和 {"error_message": ...}。
    # 将该 JSON 错误信息转换为清晰、便于调用方处理的异常。
    if response.status_code == 400:
        try:
            message = response.json().get("error_message", response.text)
        except ValueError:
            message = response.text
        raise ValueError(f"FRED request failed: {message}")
    return response.json()


def get_macro_data(
    indicator: str,
    as_of_date: str,
    look_back_days: int | None = None,
) -> str:
    """获取 FRED 宏观经济序列，返回格式化的 Markdown 报告。

    参数：
        indicator: 常用指标别名，例如 "cpi"、"unemployment"、"10y_treasury"，
            或原始 FRED 序列 ID，例如 "CPIAUCSL"、"DGS10"。
        as_of_date: 数据截止日期，格式为 yyyy-mm-dd。该日期既限定观测窗口，
            也固定数据版本：查询 FRED 时，将实时日期范围设为 ``as_of_date``，
            且不晚于 FRED 当天的日期，确保历史运行使用截至该日实际已发布的数值，
            而非后续修订值。否则，CPI、GDP 等经常修订的序列会在回测中引入未来信息（#1275）。
        look_back_days: 回溯窗口的天数；为 ``None`` 时使用 DEFAULT_LOOKBACK_DAYS。

    返回：
        Markdown 报告，包含序列标题、单位、频率、最新数值、窗口内的变化量及近期观测表格。
    """
    if look_back_days is None:
        look_back_days = DEFAULT_LOOKBACK_DAYS

    end_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    # 固定数据版本。FRED 的两个实时日期边界默认均为当天，会返回每条观测的最新修订值；
    # 对元数据和观测数据请求都设置同一天的实时日期区间，即可获取截至该日已知的数值（#1275）。
    # 截止日期不能晚于 FRED 当天的日期：实时运行时，as_of_date 是调用方的本地日期，
    # 可能比芝加哥早一天；超过 FRED 当前日期会触发 HTTP 400，导致路由层静默跳过宏观数据。
    # 历史 as_of_date 不受此限制影响，因此仍能保留仅使用指定历史时点已知数据的行为。
    pit = min(as_of_date, _fred_today())
    realtime = {"realtime_start": pit, "realtime_end": pit}

    # LLM 提供的指标无效时，返回提示信息而不抛出异常，避免错误参数中断整个运行。
    # 虽然路由层也会对宏观数据失败进行降级处理，但具体提示对分析师更有帮助。
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
