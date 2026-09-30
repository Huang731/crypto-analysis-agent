"""把 Supervisor 的路由和回答输出为 SSE 事件。"""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from api.schemas import AskRequest
from app.observability import get_logger, log_event
from app.orchestration.query_rewriter import rewrite_question


def _event(name: str, payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, str]:
    """按 QA 项目的格式序列化 SSE 事件。"""
    return {"event": name, "data": json.dumps(payload, ensure_ascii=False)}


_LOG = get_logger("crypto_analysis.supervisor")
_NODE_LABELS = {
    "crypto_analysis_agent": "加密货币研究",
    "topic_research_agent": "主题研究",
    "plan": "识别币种与规划工具",
    "dimension_worker": "执行维度",
    "research_team": "并行研究",
    "writer": "撰写报告",
    "reviewer": "审稿",
    "split": "拆解研究主题",
    "researcher": "并行分析子问题",
    "calculate_metrics": "计算指标",
    "summarize": "汇总发现",
}

class SupervisorService:
    """将用户问题交给 Supervisor，并报告实际分派的 Agent。"""

    def __init__(self, supervisor: Any, rewriter_llm: BaseChatModel | None = None) -> None:
        self._supervisor = supervisor
        self._rewriter_llm = rewriter_llm  # 为空时跳过问题改写。

    async def stream_ask(
        self, request: AskRequest, thread_id: str, trace_id: str, scope: dict[str, str]
    ) -> AsyncIterator[dict[str, str]]:
        """输出分派过程和 Supervisor 的最终回答。"""
        log_event(_LOG, "supervisor.start")
        question = request.question
        if self._rewriter_llm is not None:
            yield _event("progress", {"detail": "正在理解问题"})
            question = await rewrite_question(self._rewriter_llm, question)
            if question != request.question:  # 让用户看到系统实际按什么问题处理。
                yield _event("progress", {"label": "问题改写", "status": question})
        yield _event("progress", {"detail": "正在判断问题意图"})
        final_state: dict = {"dimension_results": [], "findings": []}  # 累积所选 Agent 的结果。
        routed_agent: str | None = None  # 记录 Supervisor 实际交接的 Agent。
        direct_reply = ""  # Supervisor 未分派时的直接回复。

        config = {"configurable": {"thread_id": thread_id}}
        async for namespace, mode, payload in self._supervisor.astream(
            {"messages": [HumanMessage(content=question)], "scope": scope},
            stream_mode=["updates", "messages", "custom"],
            subgraphs=True,
            config=config,
        ):
            if mode == "updates":
                # payload = {node_name: state_delta}
                for node, delta in payload.items():
                    if not namespace and node == "supervisor" and isinstance(delta, dict):  # 从工具调用识别实际分派。
                        for message in delta.get("messages", []):  # Supervisor 可能一次返回多条消息。
                            if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
                                direct_reply = message.text  # 不带工具调用的 AI 消息即直接回复。
                            for call in getattr(message, "tool_calls", []):  # 只读取结构化交接工具。
                                tool_name = call.get("name", "")  # 提取工具名称。
                                if tool_name.startswith("transfer_to_"):  # 默认交接工具以该前缀命名。
                                    agent_name = tool_name.removeprefix("transfer_to_")  # 提取子 Agent 名称。
                                    if agent_name not in {"crypto_analysis_agent", "topic_research_agent"}:  # 拒绝未知交接目标。
                                        raise ValueError(f"Supervisor 分派了未知 Agent：{agent_name}")  # 不猜测路由。
                                    if routed_agent is not None and routed_agent != agent_name:  # 第一版只允许单个 Agent。
                                        raise ValueError("Supervisor 一轮请求分派了多个 Agent")  # 避免混合两类报告。
                                    if routed_agent is None:  # 首次识别交接时通知前端。
                                        routed_agent = agent_name  # 固定本轮路由。
                                        log_event(_LOG, "supervisor.route", agent=agent_name)  # 记录实际分派。
                                        yield _event("route", {"agent": agent_name})  # 推送分派事件。
                    if not namespace and node in {"crypto_analysis_agent", "topic_research_agent"}:  # 再根据实际执行节点确认分派。
                        if routed_agent is not None and routed_agent != node:  # 工具交接与实际节点不得冲突。
                            raise ValueError("Supervisor 实际分派与计划目标不一致")  # 防止展示错误类型。
                        if routed_agent is None:  # 某些更新模式只显示已执行的节点。
                            routed_agent = node  # 以实际执行节点确定本轮类型。
                            log_event(_LOG, "supervisor.route", agent=node)  # 记录分派结果。
                            yield _event("route", {"agent": node})  # 通知前端分派对象。
                    if (
                        not isinstance(delta, dict)
                        or node not in _NODE_LABELS
                        or node == "calculate_metrics"
                    ):
                        continue
                    # 进度事件：告诉前端哪个节点完成了什么
                    progress = _summarize_update(node, delta)
                    yield _event("progress", progress)
                    # 父图更新构成最终状态，并行列表需要逐项累积。
                    if not namespace:
                        if node == "dimension_worker":  # 每个维度会单独返回一次增量。
                            final_state["dimension_results"].extend(delta["dimension_results"])  # 合并维度结果。
                            final_state["findings"].extend(delta["findings"])  # 合并指标事实。
                        else:
                            final_state.update(delta)  # 保存计划范围或最终报告。
                    elif node == "writer" and "report" in delta:
                        final_state["report"] = delta["report"]
                    elif node == "research_team" and "findings" in delta:
                        final_state["findings"] = delta["findings"]

            elif mode == "messages":
                # payload = (message_chunk, metadata)
                msg_chunk, meta = payload
                content = msg_chunk.content if hasattr(msg_chunk, "content") else str(msg_chunk)
                if content:  # 空 chunk 跳过
                    node = meta.get("langgraph_node", "?") if isinstance(meta, dict) else "?"
                    yield _event("token", {"node": node, "content": content})
            elif mode == "custom" and payload.get("event") == "dimension_completed":
                dimension_id = payload.get("dimension_id", "")  # 读取完成的维度编号。
                yield _event(
                    "progress",
                    {
                        "node": "dimension_worker",
                        "label": _NODE_LABELS["dimension_worker"],
                        "status": f"维度 {dimension_id} 已完成",
                    },
                )

        if routed_agent == "crypto_analysis_agent":  # 维度路径沿用原有完成事件字段。
            dimensions = sorted(final_state["dimension_results"], key=lambda item: item["dimension_id"])  # 固定维度展示顺序。
            indicators = [indicator for dimension in dimensions for indicator in dimension["indicators"]]  # 汇总各维指标。
            yield _event(  # 保留前端现有维度数据结构。
                "done",  # 标记本轮分析完成。
                {  # 维度报告及结构化事实。
                    "kind": "crypto_analysis",  # 标记当前返回类型。
                    "report": final_state["report"],  # 已生成的维度报告。
                    "findings": final_state["findings"],  # 指标事实清单。
                    "target_asset": final_state["scope"]["target_asset"],  # 使用校验过的币种。
                    "as_of": final_state["scope"]["as_of"],  # 返回固定截止时间。
                    "dimensions": dimensions,  # 返回维度状态。
                    "indicators": indicators,  # 返回全部指标记录。
                },  # 结束维度数据结构。
            )  # 推送维度完成事件。
        elif routed_agent == "topic_research_agent":  # 通用研究不依赖币种和维度字段。
            yield _event(  # 推送独立的主题研究结果。
                "done",  # 标记本轮研究完成。
                {  # 模型知识研究的返回结构。
                    "kind": "topic_research",  # 标记普通主题研究。
                    "report": final_state["report"],  # 已审稿的主题报告。
                    "findings": final_state["findings"],  # 三个子问题的研究发现。
                    "verification": "未联网核验",  # 明确第一版无外部资料。
                },  # 结束主题研究数据结构。
            )  # 完成主题研究输出。
        elif direct_reply.strip():  # 闲聊或超出研究范围的问题，由 Supervisor 直接回复。
            log_event(_LOG, "supervisor.direct_reply", reply_chars=len(direct_reply))
            # 以独立 kind 返回，前端不会把它当作研究报告展示。
            yield _event("done", {"kind": "direct", "report": direct_reply.strip()})
        else:  # 既未分派也没有回复内容。
            raise ValueError("Supervisor 未分派研究 Agent 且未给出回复")


def _summarize_update(node: str, delta: dict) -> dict:
    """把节点 state delta 摘要成前端可读的进度事件。"""
    info = {"node": node, "label": _NODE_LABELS.get(node, node)}
    if node == "topic_research_agent":  # 显示通用研究完成进度。
        info["status"] = "主题研究已完成"  # 与维度分开展示。
    elif node == "crypto_analysis_agent":  # 显示维度分析完成进度。
        info["status"] = "加密货币维度分析已完成"  # 保留币种分析语义。
    elif node == "researcher" and "findings" in delta:  # 显示单个子问题完成。
        info["status"] = f"已分析：{delta['findings'][0]['question']}"  # 读取研究节点的结构化问题。
    elif node == "research_team" and "research_summary" in delta:
        info["status"] = "研究完成，开始撰写"
        info["findings_count"] = len(delta.get("findings", []))
    elif node == "writer" and "report" in delta:
        info["status"] = f"报告已生成（{len(delta.get('report', ''))} 字）"
    elif node == "plan":  # 计划完成时显示模型识别的币种。
        info["status"] = f"已识别 {delta['scope']['target_asset']}，准备执行维度"
    elif node == "dimension_worker":  # 维度执行完成时显示状态。
        item = delta["dimension_results"][0]  # 当前增量只对应一个维度。
        info["status"] = f"维度 {item['dimension_id']}：{item['status']}"
    elif node == "reviewer":
        info["status"] = f"审稿：{delta.get('review_decision', '?')}"
        info["review_count"] = delta.get("review_count", 0)  # 展示当前审稿轮次。
    else:
        info["status"] = "完成"
        info["keys"] = list(delta.keys())
    return info
