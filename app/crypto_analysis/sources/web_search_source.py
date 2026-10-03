"""Bocha 搜索来源，只返回候选链接，不把摘要视为正式证据。"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class BochaSettings(BaseSettings):
    """从环境变量或项目 .env 读取 Bocha 配置。"""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bocha_api_key: str = ""
    bocha_base_url: str = "https://api.bocha.cn"


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
