"""Bocha 网页搜索工具，只返回候选链接，不把摘要视为正式证据。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.browser_tool import BrowserContext


class BochaSettings(BaseSettings):
    """从环境变量或项目 .env 读取 Bocha 配置。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bocha_api_key: str = ""
    bocha_base_url: str = "https://api.bocha.cn"


class WebSearchInput(BaseModel):
    """网页搜索参数。"""

    query: str = Field(min_length=1, description="搜索关键词")
    max_results: int = Field(default=6, ge=1, le=10, description="最多返回结果数")


@tool(args_schema=WebSearchInput)
async def search_bocha(query: str, max_results: int = 6) -> dict[str, Any]:
    """调用 Bocha Web Search，失败直接交给上层记录，不切换来源。"""
    if not query.strip() or not 1 <= max_results <= 10:
        raise ValueError("搜索词不能为空，max_results 必须为 1 至 10。")
    settings = BochaSettings()
    if not settings.bocha_api_key:
        raise RuntimeError("未配置 BOCHA_API_KEY")
    endpoint = f"{settings.bocha_base_url.rstrip('/')}/v1/web-search"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {settings.bocha_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "summary": True,
                "count": max_results,
            },
        )
    response.raise_for_status()
    payload = response.json()
    if payload["code"] != 200:
        raise RuntimeError(f"Bocha API 错误 {payload['code']}: {payload.get('msg')}")
    rows = payload["data"]["webPages"]["value"]
    return {
        "query": query,
        "source": endpoint,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {"title": row["name"], "url": row["url"], "snippet": row["snippet"]}
            for row in rows
        ],
    }


def build_web_search_tools() -> list[BaseTool]:
    """为研究 Agent 注册搜索工具，将候选链接保存到独立状态。"""

    @tool("search_bocha", args_schema=WebSearchInput)
    async def search_for_research(
        query: str,
        runtime: ToolRuntime[BrowserContext],
        max_results: int = 6,
    ) -> Command:
        """搜索候选网页并记录来源；摘要只用于定位，正式取证请调用 browse_for_evidence。"""
        result = await search_bocha.ainvoke(
            {"query": query, "max_results": max_results}
        )
        runtime.context.browser.register_search_results(result["results"])
        return Command(
            update={
                "search_records": {runtime.tool_call_id: result},
                "messages": [
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return [search_for_research]
