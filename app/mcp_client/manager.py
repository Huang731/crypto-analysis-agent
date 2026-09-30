"""MCP Client 连接与结构化工具调用边界。"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import Client


class MCPToolError(RuntimeError):
    """MCP 工具调用失败或返回值不符合结构化数据约定。"""


class MCPClientManager:
    """在应用生命周期内维护到各 MCP Server 的异步连接。"""

    def __init__(self, server_urls: dict[str, str]) -> None:
        self._server_urls = server_urls
        self._stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}

    async def __aenter__(self) -> "MCPClientManager":
        for name, url in self._server_urls.items():
            self._clients[name] = await self._stack.enter_async_context(Client(url))
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._stack.aclose()

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = self._clients.get(server)
        if client is None:
            raise MCPToolError(f"MCP Server未连接：{server}")
        result = await client.call_tool(tool_name, arguments)
        if result.is_error:
            raise MCPToolError(f"MCP工具调用失败：{server}.{tool_name}")
        if result.structured_content is None:
            raise MCPToolError(f"MCP工具未返回structured_content：{server}.{tool_name}")
        return result.structured_content
