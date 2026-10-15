"""FastAPI 入口：分析页面与 路由 流式问答接口。"""

import asyncio
import json
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated

if __package__ in (None, ""):
    # 以 `python api/main.py` 直接运行时，sys.path[0] 是 api/ 而非项目根目录。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sse_starlette.sse import EventSourceResponse

from api.schemas import AskRequest
from app.config.config import settings
from app.crypto_analysis.crypto_analysis_parent_agent import create_request_scope
from app.llm.model import make_fast_llm, make_smart_llm
from app.observability import (
    bind_request_context,
    get_logger,
    log_event,
    new_trace_id,
    reset_request_context,
)
from app.orchestration.routing import build_routing_graph
from app.orchestration.routing_service import RoutingService
from app.rag.retriever import KBRetriever
from app.tracing import start_trace

_INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"
_LOG = get_logger("crypto_analysis.api")
# 请求 ID 会写入日志和 checkpoint 的 thread_id，限制字符集与长度防止注入和超长键。
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时加载知识库和会话存储，退出时释放连接并等待检索结束。"""
    with start_trace("app.startup"):
        retriever = await asyncio.to_thread(KBRetriever)
    try:
        checkpoint_path = Path(settings.checkpoint_db_path).expanduser()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(
            str(checkpoint_path)
        ) as checkpointer:
            app.state.routing_service = _build_routing_service(retriever, checkpointer)
            yield
    finally:
        await retriever.aclose()
        if hasattr(app.state, "routing_service"):
            del app.state.routing_service


app = FastAPI(title="研究分析 Agent", version="0.1.0", lifespan=lifespan)
# 页面样式与 Markdown 组件由本服务提供，不依赖外部 CDN。
app.mount("/static", StaticFiles(directory=_INDEX.parent), name="static")


# ---------- 工具函数 ----------


@contextmanager
def _request_context(trace_id: str, thread_id: str) -> Iterator[None]:
    """在代码块内把请求 ID 绑定到日志上下文，退出时恢复。"""
    tokens = bind_request_context(trace_id, thread_id)
    try:
        yield
    finally:
        reset_request_context(tokens)


def _id_headers(trace_id: str, thread_id: str) -> dict[str, str]:
    """回传给前端的请求 ID 响应头，便于按 trace_id 排查日志。"""
    return {"X-Trace-Id": trace_id, "X-Thread-Id": thread_id}


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


# ---------- 依赖注入 ----------


def _build_routing_service(
    retriever: KBRetriever, checkpointer: BaseCheckpointSaver
) -> RoutingService:
    """构建共享检索器和会话存储的路由图，由应用生命周期持有。"""
    smart, fast = make_smart_llm(), make_fast_llm()
    graph = build_routing_graph(smart, fast, retriever, checkpointer=checkpointer)
    return RoutingService(graph)


async def get_routing_service(request: Request) -> RoutingService:
    """校验/生成请求 ID 并存入 request.state，再提供问答服务。

    前端可通过请求头传入 X-Thread-Id 以延续多轮会话；未传则每次新开会话。
    """
    thread_id = request.headers.get("X-Thread-Id") or f"web-{uuid.uuid4().hex[:8]}"
    trace_id = request.headers.get("X-Trace-Id") or new_trace_id()
    if not (_ID_PATTERN.fullmatch(thread_id) and _ID_PATTERN.fullmatch(trace_id)):
        raise HTTPException(
            status_code=400, detail="X-Thread-Id 或 X-Trace-Id 格式无效"
        )
    request.state.thread_id = thread_id
    request.state.trace_id = trace_id

    return request.app.state.routing_service


# ---------- 路由 ----------


@app.get("/")
async def index() -> FileResponse:
    """返回分析页面。"""
    return FileResponse(_INDEX)


@app.get("/health")
async def health() -> dict[str, str]:
    """存活探针，不依赖 LLM 与外部数据源。"""
    return {"status": "ok", "service": "crypto-analysis-agent"}


@app.post("/api/ask")
async def ask(
    req: AskRequest,
    request: Request,
    service: Annotated[RoutingService, Depends(get_routing_service)],
) -> EventSourceResponse:
    """以 SSE 流返回分派进度和最终回答。

    事件类型：progress / route / token / done / error。
    """
    thread_id: str = request.state.thread_id
    trace_id: str = request.state.trace_id
    # 在请求入口固定一次 UTC 截止时间，保证各个分析技能使用同一时间点的数据。
    scope = create_request_scope(req.question, thread_id, trace_id)

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        # 生成器在响应阶段才执行，需要在这里重新绑定日志上下文。
        with (
            start_trace(
                "api.ask",
                trace_id=trace_id,
                thread_id=thread_id,
                question_chars=len(req.question),
            ) as trace,
            _request_context(trace_id, thread_id),
        ):
            started = time.perf_counter()
            log_event(_LOG, "request.start", question_chars=len(req.question))
            try:
                async for item in service.stream_ask(req, thread_id, trace_id, scope):
                    # 客户端断开后立即停止，避免继续消耗 LLM 调用。
                    if await request.is_disconnected():
                        log_event(_LOG, "request.disconnected")
                        trace.output = {"status": "disconnected"}
                        return
                    if item.get("event") == "done":
                        payload = json.loads(item["data"])
                        trace.output = {
                            "status": "done",
                            "kind": payload.get("kind"),
                            "report_chars": len(payload.get("report", "")),
                        }
                    yield item
                log_event(_LOG, "request.done", duration_ms=_elapsed_ms(started))
                if trace.output is None:
                    trace.output = {"status": "done"}
            except Exception as exc:  # noqa: BLE001 - SSE 已开始推送，任何异常都必须转成 error 事件返回给前端。
                _LOG.exception(
                    "request.failed",
                    extra={
                        "event": "request.failed",
                        "duration_ms": _elapsed_ms(started),
                    },
                )
                trace.output = {"status": "error", "error_type": type(exc).__name__}
                error = {
                    "code": "routing_failed",
                    "message": "问答执行失败，请根据 trace_id 检查服务日志。",
                    "thread_id": thread_id,
                    "trace_id": trace_id,
                }
                yield {"event": "error", "data": json.dumps(error, ensure_ascii=False)}

    return EventSourceResponse(
        event_generator(), headers=_id_headers(trace_id, thread_id)
    )


def main() -> None:
    """启动本地分析服务（开发时可改用 `uvicorn api.main:app --reload`）。"""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8003, reload=False)


if __name__ == "__main__":
    main()
