"""EODHD 官方 MCP 数据接口。"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.mcp_client.manager import MCPClientManager

_ROOT = Path(__file__).resolve().parents[3]  # 数据源目录比项目根目录多一层。
_DXY_SOURCE = "https://eodhd.com/api/eod/DXY.INDX"


class AnalysisMCPSettings(BaseSettings):
    """读取 EODHD 官方 MCP 服务所需的 API Key。"""

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    eodhd_api_key: str = ""


async def query_dxy_from_official_mcp(
    as_of: str,
) -> dict[str, Any]:  # 只获取并解析 EODHD 原始日线。
    """调用 EODHD 官方 MCP 日线工具，返回原始日线与来源。"""
    api_key = AnalysisMCPSettings().eodhd_api_key
    if not api_key:
        raise RuntimeError("未配置 EODHD_API_KEY")
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    end_date = cutoff.astimezone(timezone.utc).date()
    start_date = end_date - timedelta(days=365)
    # 官方 v1 用 URL 查询参数传递密钥，关闭 HTTP 请求地址的常规日志。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    endpoint = f"https://mcp.eodhd.com/v1/mcp?apikey={quote(api_key, safe='')}"
    arguments = {
        "ticker": "DXY.INDX",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "period": "d",
        "order": "a",
        "fmt": "json",
    }
    try:
        async with MCPClientManager({"eodhd": endpoint}) as mcp_client:
            response = await mcp_client.call_tool(
                "eodhd", "get_historical_stock_prices", arguments
            )
    except Exception:  # noqa: BLE001 - MCP 连接失败类型不固定，统一转成稳定的 RuntimeError。
        raise RuntimeError("EODHD 官方 MCP 连接或工具调用失败") from None
    raw_data = response.get("result")
    if not isinstance(raw_data, str):
        raise TypeError("EODHD 官方 MCP 未返回日线 JSON 字符串")
    try:
        rows = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ValueError("EODHD 官方 MCP 日线结果不是有效 JSON") from exc
    return {"rows": rows, "source": _DXY_SOURCE}  # 将原始数组交给程序计算器。
