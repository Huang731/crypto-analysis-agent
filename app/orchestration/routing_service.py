"""把 路由 的路由和回答输出为 SSE 事件。"""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from api.schemas import AskRequest
from app.observability import get_logger, log_event
from app.tracing import trace_span


def _event(name: str, payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, str]:
    """按 QA 项目的格式序列化 SSE 事件。"""
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


_LOG = get_logger("crypto_analysis.routing")
_NODE_LABELS = {
    "rewrite": "理解问题",
    "router": "识别研究方向",
    "finalize": "整理最终报告",
    "crypto_analysis_agent": "加密货币研究",
    "topic_research_agent": "主题研究",
    "plan": "识别币种与规划工具",
    "skill_worker": "执行技能",
    "research_team": "并行研究",
    "writer": "撰写报告",
    "reviewer": "审稿",
    "split": "拆解研究主题",
    "researcher": "并行分析子问题",
    "calculate_metrics": "计算指标",
    "summarize": "汇总发现",
    "summarize_history": "更新历史摘要",
}


@dataclass
class _RoutingStreamState:
    """每个请求独立收集路由、回复与子图结果，不修改图状态。"""

    final_state: dict[str, Any] = field(
        default_factory=lambda: {"skill_results": [], "findings": []}
    )
    routed_agent: str | None = None
    direct_reply: str = ""

    def route(self, agent: str, conflict_reason: str) -> dict[str, str] | None:
        """首次路由通知前端，实际执行必须与路由目标一致。"""
        if self.routed_agent is not None and self.routed_agent != agent:
            raise ValueError(conflict_reason)
        if self.routed_agent is not None:
            return None
        self.routed_agent = agent
        return _event("route", {"agent": agent})

    def apply_update(
        self, namespace: tuple[str, ...], node: str, delta: Any
    ) -> Iterator[dict[str, str]]:
        """解析图更新，按原顺序输出路由、进度并收集报告数据。"""
        agents = {"crypto_analysis_agent", "topic_research_agent"}
        is_root = not namespace
        if is_root and node == "rewrite" and isinstance(delta, dict):
            self.final_state.update(delta)
            if delta["resolved_question"] != delta["scope"]["original_question"]:
                yield _event(
                    "progress",
                    {"label": "问题改写", "status": delta["resolved_question"]},
                )
            yield _event("progress", {"detail": "正在判断问题意图"})
            return
        if is_root and node == "router" and isinstance(delta, dict):
            agent = delta["route"]
            log_event(
                _LOG,
                "routing.state.update",
                route=agent,
                direct_reply=delta["direct_reply"],
            )
            if agent == "direct":
                self.direct_reply = delta["direct_reply"]
            else:
                if agent not in agents:
                    raise ValueError(f"路由分派了未知 Agent：{agent}")
                event = self.route(agent, "一轮请求分派了多个 Agent")
                if event is not None:
                    yield event

        # 根节点的实际执行也需验证，不能只依赖模型返回的路由目标。
        if is_root and node in agents:
            event = self.route(node, "路由 实际分派与计划目标不一致")
            if event is not None:
                yield event
        if (
            not isinstance(delta, dict)
            or node not in _NODE_LABELS
            or node == "calculate_metrics"
        ):
            return
        yield _event("progress", _summarize_update(node, delta))

        # 并行技能返回增量，父图返回公共状态；其余明细直接从子图收集。
        if node == "skill_worker":
            self.final_state["skill_results"].extend(delta["skill_results"])
        elif is_root:
            self.final_state.update(delta)
        elif node == "writer" and "report" in delta:
            self.final_state["report"] = delta["report"]
        elif node == "research_team" and "findings" in delta:
            self.final_state["findings"] = delta["findings"]

    def completion_event(self) -> dict[str, str]:
        """保持三类完成事件的字段，以及无路由、无回复时的明确失败。"""
        state = self.final_state
        if self.routed_agent == "crypto_analysis_agent":
            skills = sorted(state["skill_results"], key=lambda item: item["skill_name"])
            return _event(
                "done",
                {
                    "kind": "crypto_analysis",
                    "report": state["report"],
                    "target_asset": state["scope"]["target_asset"],
                    "as_of": state["scope"]["as_of"],
                    "skill_results": skills,
                },
            )
        if self.routed_agent == "topic_research_agent":
            findings = state["findings"]
            return _event(
                "done",
                {
                    "kind": "topic_research",
                    "report": state["report"],
                    "findings": findings,
                    "verification": "；".join(
                        dict.fromkeys(item["verification"] for item in findings)
                    ),
                },
            )
        if self.direct_reply.strip():
            return _event(
                "done", {"kind": "direct", "report": self.direct_reply.strip()}
            )
        _LOG.error(
            "routing.result.empty",
            extra={
                "event": "routing.result.empty",
                "routed_agent": self.routed_agent,
                "direct_reply": self.direct_reply,
                "final_state_keys": sorted(state),
            },
        )
        raise ValueError("路由 未分派研究 Agent 且未给出回复")


def _token_event(payload: Any) -> dict[str, str] | None:
    """研究与汇总 token 保持内部使用，其余非空内容沿用现有输出规则。"""
    message, metadata = payload
    content = message.content if hasattr(message, "content") else str(message)
    node = metadata.get("langgraph_node", "?") if isinstance(metadata, dict) else "?"
    if not content or node in {
        "rewrite",
        "router",
        "plan",
        "summarize",
        "model",
        "skill_worker",
        "summarize_history",
    }:
        return None
    # 加密结论使用结构化输出，JSON 增量不能作为报告正文展示。
    if isinstance(metadata, dict) and "crypto_structured_output" in metadata.get(
        "tags", []
    ):
        return None
    return _event("token", {"node": node, "content": content})


class RoutingService:
    """将用户问题交给 路由，并报告实际分派的 Agent。"""

    def __init__(self, routing_graph: Any) -> None:
        self._routing_graph = routing_graph

    async def stream_ask(
        self, request: AskRequest, thread_id: str, trace_id: str, scope: dict[str, str]
    ) -> AsyncIterator[dict[str, str]]:
        """依次完成问题改写、图事件转发和最终 SSE 回答。"""
        log_event(_LOG, "routing.start")
        scope = {**scope, "original_question": request.question}
        yield _event("progress", {"detail": "正在理解问题"})
        result = _RoutingStreamState()
        config = {
            "configurable": {"thread_id": thread_id},
        }
        with trace_span("routing_graph"):
            # v2 统一返回 type/ns/data；仅调整图事件读取方式，SSE 格式保持不变。
            async for part in self._routing_graph.astream(
                {"scope": scope},
                stream_mode=["updates", "messages"],
                subgraphs=True,
                version="v2",
                config=config,
            ):
                if part["type"] == "updates":
                    for node, delta in part["data"].items():
                        for event in result.apply_update(part["ns"], node, delta):
                            yield event
                elif part["type"] == "messages":
                    event = _token_event(part["data"])
                    if event is not None:
                        yield event
        yield result.completion_event()


def _summarize_update(node: str, delta: dict[str, Any]) -> dict[str, Any]:
    """把节点 state delta 摘要成前端可读的进度事件。"""
    info = {"node": node, "label": _NODE_LABELS.get(node, node)}
    if node == "router":
        info["status"] = (
            "直接回复" if delta["route"] == "direct" else _NODE_LABELS[delta["route"]]
        )
    elif node == "finalize":
        info["status"] = "报告整理完成"
    elif node == "topic_research_agent":  # 显示通用研究完成进度。
        info["status"] = "主题研究已完成"  # 与技能分开展示。
    elif node == "crypto_analysis_agent":  # 显示技能分析完成进度。
        info["status"] = "加密货币技能分析已完成"  # 保留币种分析语义。
    elif node == "researcher" and "findings" in delta:  # 显示单个子问题完成。
        info["status"] = (
            f"已分析：{delta['findings'][0]['question']}"  # 读取研究节点的结构化问题。
        )
    elif node == "research_team" and "research_summary" in delta:
        info["status"] = "研究完成，开始撰写"
        info["findings_count"] = len(delta.get("findings", []))
    elif node == "writer" and "report" in delta:
        info["status"] = f"报告已生成（{len(delta.get('report', ''))} 字）"
    elif node == "plan":  # 计划完成时显示模型识别的币种。
        info["status"] = f"已识别 {delta['scope']['target_asset']}，准备执行技能"
    elif node == "skill_worker":
        item = delta["skill_results"][0]
        status = "读取记忆" if item["cache_hit"] else "已生成"
        if not item["content"]:
            status = "暂无可展示内容"
        info["status"] = f"技能 {item['name']}：{status}"
    elif node == "summarize_history":
        info["status"] = "历史摘要已更新" if delta else "无需更新历史摘要"
    elif node == "reviewer":
        info["status"] = f"审稿：{delta.get('review_decision', '?')}"
        info["review_count"] = delta.get("review_count", 0)  # 展示当前审稿轮次。
    else:
        info["status"] = "完成"
        info["keys"] = list(delta.keys())
    return info
