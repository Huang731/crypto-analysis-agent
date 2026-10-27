"""把 路由 的路由和回答输出为 SSE 事件。"""

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import BaseMessage

from api.schemas import AskRequest
from app.core.observability import get_logger, log_event
from app.core.tracing import trace_span


def _event(name: str, payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, str]:
    """按 QA 项目的格式序列化 SSE 事件。"""
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


_NODE_LABELS = {
    "rewrite": "理解问题",
    "router": "识别研究方向",
    "crypto_analysis_agent": "加密货币研究",
    "topic_research_agent": "主题研究",
    "plan": "识别币种与规划工具",
    "skill_worker": "执行维度",
    "research_team": "并行研究",
    "writer": "撰写报告",
    "reviewer": "审稿",
    "split": "拆解研究主题",
    "researcher": "并行分析子问题",
    "summarize": "汇总发现",
    "summarize_history": "保存对话记忆",
}


@dataclass
class _RoutingStreamState:
    """每个请求独立收集路由与子图结果，不修改图状态。"""

    final_state: dict[str, Any] = field(
        default_factory=lambda: {"skill_results": [], "findings": []}
    )
    routed_agent: str | None = None
    writer_message_id: str | None = None

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
        if not isinstance(delta, dict) or node not in _NODE_LABELS:
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

    def apply_message(
        self, message: BaseMessage, metadata: dict[str, Any]
    ) -> Iterator[dict[str, str]]:
        """只转发写作正文，每次模型重试或报告重写都先清空旧稿。"""
        if metadata.get("langgraph_node") != "writer":
            return
        if not isinstance(message.content, str) or not message.content:
            return
        if not message.id:
            raise ValueError("报告流式片段缺少消息 ID")
        if message.id != self.writer_message_id:
            self.writer_message_id = message.id
            yield _event("report_start", {"node": "writer"})
        yield _event("token", {"node": "writer", "content": message.content})

    def completion_event(self) -> dict[str, str]:
        """保持两类专家完成事件的字段，无路由时明确失败。"""
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
        log_event(
            get_logger("crypto_analysis.routing"),
            "routing.result.empty",
            routed_agent=self.routed_agent,
            final_state_keys=sorted(state),
        )
        raise ValueError("路由未分派研究 Agent")


class RoutingService:
    """将用户问题交给 路由，并报告实际分派的 Agent。"""

    def __init__(self, routing_graph: Any) -> None:
        self._routing_graph = routing_graph

    async def stream_ask(
        self,
        request: AskRequest | None,
        thread_id: str,
        scope: dict[str, str] | None,
    ) -> AsyncIterator[dict[str, str]]:
        """新问题和检查点续跑共用进度、正文流及最终结果处理。"""
        result = _RoutingStreamState()
        config = {"configurable": {"thread_id": thread_id}}
        if request is not None:
            if scope is None:
                raise ValueError("新问题缺少请求范围")
            graph_input = {"scope": {**scope, "original_question": request.question}}
            progress = {"detail": "正在理解问题"}
        else:
            graph_input = None
            snapshot = await self._routing_graph.aget_state(config)
            if snapshot.created_at is None:
                raise HTTPException(
                    status_code=404, detail="当前会话没有可继续的任务，请先开始分析。"
                )
            progress = {
                "detail": "正在继续上次任务",
                "question": snapshot.values["scope"]["original_question"],
            }
        yield _event("progress", progress)

        with trace_span("routing_graph"):
            # 节点更新保留进度和最终状态，模型消息只转发写作节点的正文。
            async for part in self._routing_graph.astream(
                graph_input,
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
                    message, metadata = part["data"]
                    for event in result.apply_message(message, metadata):
                        yield event
        if request is None:
            # 成功分支可能不再发事件，结束时从专家检查点补齐完整结果。
            snapshot = await self._routing_graph.aget_state(config)
            route = snapshot.values["route"]
            expert = await self._routing_graph.aget_state(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": route}}
            )
            if snapshot.next or expert.next or expert.created_at is None:
                raise ValueError("恢复任务尚未完整结束")
            for key in ("trace_id", "as_of"):
                value = snapshot.values["scope"].get(key)
                if not value or value != expert.values["scope"].get(key):
                    raise ValueError("专家结果与本轮任务不一致")
            result.final_state = {**expert.values, **snapshot.values}
            event = result.route(route, "恢复任务的专家不一致")
            if event is not None:
                yield event
        yield result.completion_event()


def _summarize_update(node: str, delta: dict[str, Any]) -> dict[str, Any]:
    """把节点 state delta 摘要成前端可读的进度事件。"""
    info = {"node": node, "label": _NODE_LABELS[node]}
    if node == "router":
        info["status"] = _NODE_LABELS[delta["route"]]
    elif node == "topic_research_agent":  # 显示通用研究完成进度。
        info["status"] = "主题研究已完成"  # 与技能分开展示。
    elif node == "crypto_analysis_agent":  # 显示技能分析完成进度。
        info["status"] = "加密货币维度分析已完成"  # 保留币种分析语义。
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
        info["status"] = f"已识别 {delta['scope']['target_asset']}，准备执行维度"
    elif node == "skill_worker":
        item = delta["skill_results"][0]
        status = "读取记忆" if item["cache_hit"] else "已生成"
        if not item["content"]:
            status = "暂无可展示内容"
        info["status"] = f"维度 {item['name']}：{status}"
    elif node == "reviewer":
        info["status"] = f"审稿：{delta.get('review_decision', '?')}"
        info["review_count"] = delta.get("review_count", 0)  # 展示当前审稿轮次。
    elif node == "summarize_history":
        info["status"] = "对话记忆已保存"
    else:
        info["status"] = "完成"
        info["keys"] = list(delta.keys())
    return info
