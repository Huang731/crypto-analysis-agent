"""浏览器取证工具：批量读取详情页并保存原文证据。"""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from typing_extensions import Self

from app.config.config import settings
from app.observability import get_logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page, Playwright

_LOG = get_logger("browser_tool")
SENSITIVE_PATTERNS = (
    r"\.(exe|dmg|sh)$",
    r"/login",
    r"/signin",
    r"/pay",
    r"/checkout",
    r"/submit",
    r"/post",
)


def check_url_allowed(url: str) -> bool:
    """只接受带主机名的 HTTP/HTTPS 绝对地址，不限制域名。"""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def is_sensitive_url(url: str) -> bool:
    """研究取证不访问登录、支付、提交或程序下载地址。"""
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in SENSITIVE_PATTERNS)


@dataclass
class Evidence:
    """一条实际网页原文及其来源信息。"""

    content: str
    url: str
    accessed_at: str
    page_title: str = ""


class PageRequestError(RuntimeError):
    """保留失败地址和状态码，供网页重试判断使用。"""

    def __init__(self, url: str, status_code: int | None) -> None:
        self.url = url
        self.status_code = status_code
        status = status_code if status_code is not None else "无响应"
        super().__init__(f"网页请求失败：HTTP {status}")


def should_retry_page(exc: BaseException) -> bool:
    """仅重试超时、无响应及指定的临时服务故障。"""
    if isinstance(exc, PlaywrightTimeoutError):
        return True
    return isinstance(exc, PageRequestError) and exc.status_code in {
        None,
        502,
        503,
        504,
    }


class BrowserTool:
    """懒启动浏览器；每个研究任务独立持有实例，完成后关闭。"""

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()
        self._timeout = settings.browser_page_timeout
        self._headless = settings.browser_headless

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _check_url(self, url: str) -> None:
        if not check_url_allowed(url):
            raise ValueError(f"地址必须是有效的 HTTP/HTTPS 绝对地址：{url}")
        if is_sensitive_url(url):
            raise ValueError("不允许访问登录、支付、提交或程序下载页面")

    async def _ensure_browser(self) -> None:
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless
            )
            _LOG.info("浏览器已启动")

    async def close(self) -> None:
        """关闭浏览器与 Playwright，释放当前任务的资源。"""
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

    @asynccontextmanager
    async def _new_page(self) -> AsyncIterator["Page"]:
        """为一次取证创建独立页面，退出上下文时关闭页面。"""
        await self._ensure_browser()
        if self._browser is None:
            raise RuntimeError("浏览器尚未启动")
        page = await self._browser.new_page(viewport={"width": 1280, "height": 800})
        page.set_default_timeout(self._timeout * 1000)
        try:
            yield page
        finally:
            await page.close()

    async def _safe_goto(self, page: "Page", url: str) -> None:
        """校验导航响应，失败保留状态码，不自动切换来源。"""
        self._check_url(url)
        response = await page.goto(url, wait_until="domcontentloaded")
        if response is None or response.status >= 400:
            raise PageRequestError(
                url, response.status if response is not None else None
            )
        self._check_url(page.url)

    async def _extract_current_page(
        self,
        page: "Page",
        selectors: list[str] | None = None,
    ) -> Evidence:
        """保存实际页面原文，发送给模型的预览由工具适配层单独截断。"""
        url = page.url
        locator = page.locator(", ".join(selectors) if selectors else "body")
        parts = await locator.all_inner_texts()
        content = "\n\n".join(part.strip() for part in parts if part.strip())
        if not content:
            raise ValueError("页面指定区域没有可提取的原文")
        title = await page.title()
        if page.url != url:
            raise ValueError("提取过程中页面跳转")
        return Evidence(
            content=content,
            url=url,
            accessed_at=datetime.now(timezone.utc).isoformat(),
            page_title=title,
        )

    @retry(
        retry=retry_if_exception(should_retry_page),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=3),
        reraise=True,
    )
    async def extract_from_page(
        self,
        url: str,
        selectors: list[str] | None = None,
    ) -> Evidence:
        """单页最多尝试三次，每次独立创建和关闭页面。"""
        self._check_url(url)
        async with self._new_page() as page:
            await self._safe_goto(page, url)
            return await self._extract_current_page(page, selectors)

    async def browse_for_evidence(
        self,
        query: str,
        urls: list[str],
        max_pages: int = 3,
    ) -> list[Evidence]:
        """依次读取候选详情页；失败明确报错，不跳过失败页面。"""
        if not query.strip() or not urls or max_pages < 1:
            raise ValueError("研究问题、候选地址不能为空，max_pages 必须大于零")
        async with self._lock:
            selected = list(dict.fromkeys(urls))[:max_pages]
            for url in selected:
                self._check_url(url)
            _LOG.info("为研究问题读取 %s 个详情页", len(selected))
            return [await self.extract_from_page(url) for url in selected]


@dataclass
class BrowserContext:
    """研究节点注入独立浏览实例，不向模型暴露资源对象。"""

    browser: BrowserTool


class BrowseEvidenceInput(BaseModel):
    """详情页取证参数，每次成功读取后立即保存一条原文。"""

    query: str = Field(min_length=1, description="研究子问题")
    urls: list[str] = Field(min_length=1, description="搜索结果中的真实详情页链接")
    max_pages: int = Field(default=1, ge=1, le=6, description="每次最多读取一页")


def _evidence_command(
    runtime: ToolRuntime[BrowserContext], evidences: list[Evidence], requested_url: str
) -> Command:
    records = {
        f"{runtime.tool_call_id}:{index}": {
            "evidence_id": f"{runtime.tool_call_id}:{index}",
            **asdict(evidence),
        }
        for index, evidence in enumerate(evidences, 1)
    }
    previews = [
        {
            **record,
            "content": record["content"][:12000],
            "content_truncated": len(record["content"]) > 12000,
        }
        for record in records.values()
    ]
    return Command(
        update={
            "research_evidence": records,
            "visited_urls": [requested_url],
            "messages": [
                ToolMessage(
                    content=json.dumps(previews, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(args_schema=BrowseEvidenceInput)
async def browse_for_evidence(
    query: str,
    urls: list[str],
    runtime: ToolRuntime[BrowserContext],
    max_pages: int = 1,
) -> Command:
    """每次打开一个候选详情页，保存带证据 ID、来源 URL 和访问时间的原文。"""
    evidences = await runtime.context.browser.browse_for_evidence(
        query, urls, min(max_pages, 1)
    )
    return _evidence_command(runtime, evidences, urls[0])


def build_browser_tools() -> list[BaseTool]:
    """注册批量详情页取证工具。"""
    return [browse_for_evidence]
