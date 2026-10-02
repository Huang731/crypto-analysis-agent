"""MCP Client 连接与结构化工具调用边界。"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any, Self

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client


class MCPToolError(RuntimeError):
    """MCP 工具调用失败或返回值不符合结构化数据约定。"""


class MCPClientManager:
    """在应用生命周期内维护到各 MCP Server 的异步连接。"""

    def __init__(self, server_urls: dict[str, str], server_headers: dict[str, dict[str, str]] | None = None) -> None:
        self._server_urls = server_urls
        self._server_headers = server_headers or {}
        self._stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}

    async def __aenter__(self) -> Self:
        for name, url in self._server_urls.items():
            if name in self._server_headers:
                # 认证只用于指定服务，不把ETF密钥发送给其他MCP。
                http_client = await self._stack.enter_async_context(
                    create_mcp_http_client(headers=self._server_headers[name])
                )
                transport = streamable_http_client(url, http_client=http_client)
                self._clients[name] = await self._stack.enter_async_context(Client(transport))
            else:
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

    async def call_json_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用已确认以单个文本块返回JSON的工具，保留原结构化调用契约。"""
        client = self._clients.get(server)
        if client is None:
            raise MCPToolError(f"MCP Server未连接：{server}")
        result = await client.call_tool(tool_name, arguments)
        if result.is_error:
            raise MCPToolError(f"MCP工具调用失败：{server}.{tool_name}")
        if len(result.content) != 1 or result.content[0].type != "text":
            raise MCPToolError(f"MCP工具未返回单个JSON文本块：{server}.{tool_name}")
        try:
            payload = json.loads(result.content[0].text)
        except json.JSONDecodeError as exc:
            raise MCPToolError("MCP工具文本不是有效JSON") from exc
        if not isinstance(payload, dict):
            raise MCPToolError("MCP工具JSON必须为对象")
        return payload
