"""DuckDuckGo 搜索来源，只返回候选链接，不把摘要视为正式证据。"""

import asyncio
from datetime import datetime, timezone
from typing import Any


async def search_duckduckgo(query: str, max_results: int = 6) -> dict[str, Any]:
    """固定 DuckDuckGo 后端，不在失败时切换搜索引擎。"""
    if not query.strip() or not 1 <= max_results <= 10:
        raise ValueError("搜索词不能为空，max_results 必须为 1 至 10。")

    def search():
        from ddgs import DDGS

        return DDGS(timeout=20).text(query, backend="duckduckgo", max_results=max_results)

    rows = await asyncio.to_thread(search)
    return {
        "query": query,
        "source": "https://duckduckgo.com/",
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "results": [{"title": row["title"], "url": row["href"], "snippet": row["body"]} for row in rows],
    }
