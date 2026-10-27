"""步数预算与动作循环判断。"""

from __future__ import annotations

import hashlib
from typing import Any

from .config import settings
from .observability import get_logger, log_event


def step_delta(node: str, param: str = "") -> dict[str, Any]:
    """返回本批动作的步数增量及签名。"""
    sig = _signature(node, param)
    return {"step_count": 1, "action_history": [sig]}


def _signature(node: str, param: str) -> str:
    """生成动作签名。"""
    h = hashlib.md5(param[:80].encode("utf-8")).hexdigest()[:8]
    return f"{node}:{h}"


def detect_loop(action_history: list[str], window: int | None = None) -> bool:
    """判断末尾动作签名是否连续重复。"""
    if not action_history:
        return False
    w = window if window is not None else settings.loop_detect_window
    if len(action_history) < w:
        return False
    tail = action_history[-w:]
    return len(set(tail)) == 1


def should_truncate(state: dict[str, Any], *, next_steps: int = 0) -> tuple[bool, str]:
    """执行前检查本批动作是否超出预算。"""
    step_count = state["tool_call_counts"]
    if step_count + next_steps > settings.max_tool_calls_per_skill:
        return True, (
            f"工具调用预算超限（已用 {step_count}，本批 {next_steps}，"
            f"上限 {settings.max_tool_calls_per_skill}）"
        )
    if detect_loop(state.get("action_history", [])):
        return True, "检测到动作循环（连续重复同一动作签名）"
    return False, ""


def honest_truncation_delta(reason: str, **fields: Any) -> dict[str, Any]:
    """记录截断原因并返回标记，由调用方结束内部 Agent。"""
    log_event(
        get_logger("crypto_analysis.step_budget"),
        "工具调用预算耗尽",
        reason=reason,
        **fields,
    )
    return {"truncated": True}
