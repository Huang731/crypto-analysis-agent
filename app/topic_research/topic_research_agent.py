"""通用主题研究 Agent：拆题、并行分析、汇总、写作和审稿。"""

import asyncio  # 运行现有命令行入口。
import json  # 将并行发现完整交给汇总模型。
import re  # 清理拆题结果中的常见编号。
import sys  # 读取命令行中的研究主题。
from datetime import datetime, timezone  # 为命令行研究记录分析时间。
from typing import Any  # 标注模型和消息集合。

from langchain_core.messages import AIMessage, HumanMessage  # 读取原问题并回写报告。
from langgraph.graph import END, START, StateGraph  # 保留原有父图和子图结构。
from langgraph.types import Send  # 并行分派研究子问题。

from app.config.config import settings  # 读取模型配置用于成本估算。
from app.llm.model import make_fast_llm, make_smart_llm  # 复用项目现有模型工厂。
from app.observability import get_logger, log_event, timed_node  # 保留请求观测。
from app.tracing import estimate_tokens, start_trace, trace_generation, trace_span  # 记录请求与模型调用。
from app.topic_research.topic_research_state import AnalysisState, SystemState  # 使用主题研究专属状态。

_LOG = get_logger("crypto_analysis.crypto_analysis_agent")  # 沿用原有日志名称。
_NUMBER_PREFIX = re.compile(r"^\s*(?:\d+[.)、]\s*|[-*]\s+)")  # 识别模型偶尔生成的行首编号。
_SUBQUESTION_COUNT = 3  # 第一版固定并行研究三个子问题。
_MAX_REVIEWS = 2  # 审稿最多执行两轮，避免图无限循环。


def _latest_question(messages: list[Any]) -> str:  # 从 Supervisor 消息历史中找用户原问题。
    for message in reversed(messages):  # 交接消息可能排在用户消息后面。
        if isinstance(message, HumanMessage) and isinstance(message.content, str):  # 只接受真实用户文本。
            question = message.content.strip()  # 去除问题首尾空白。
            if question:  # 空问题不能进入研究图。
                return question  # 返回最近的一轮用户主题。
    raise ValueError("主题研究缺少有效用户问题")  # 不把交接工具消息当作主题。


def make_split(fast_llm: Any):  # 创建研究主题拆解节点。
    """把用户主题拆成三个可独立分析的子问题。"""

    @timed_node  # 沿用现有节点耗时和失败日志。
    async def split(state: AnalysisState) -> dict[str, list[str]]:  # 返回并行研究任务。
        with trace_span("topic.split") as node_span:
            try:
                prompt = (  # 约束模型只返回研究问题。
                    "你是通用主题研究规划员。将用户问题拆成三个互补、可独立分析的子问题。"
                    "每行只写一个问题，不写说明、标题或引用。不要假装已经联网检索。"
                    f"\n用户问题：{state['question']}"  # 提供原始研究主题。
                )  # 完成拆题提示词。
                with trace_generation("topic_split", model=settings.fast_model) as generation:  # 记录拆题模型调用。
                    response = await fast_llm.ainvoke(prompt)  # 让快模型生成子问题。
                    if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                        generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
                if not isinstance(response.content, str):  # 消息内容必须是文本。
                    raise ValueError("主题拆解模型未返回文本")  # 不猜测非文本内容。
                subquestions = [  # 按行读取研究任务。
                    _NUMBER_PREFIX.sub("", line).strip()  # 只清理编号和空白。
                    for line in response.content.splitlines()  # 遍历模型返回的每一行。
                    if line.strip()  # 忽略纯空白行。
                ]  # 完成子问题列表。
                if len(subquestions) != _SUBQUESTION_COUNT or len(set(subquestions)) != _SUBQUESTION_COUNT or any(not item for item in subquestions):  # 要求恰好三个不同的问题。
                    raise ValueError("主题拆解必须返回三个非空且不同的子问题")  # 拒绝缺漏或重复任务。
                return {"subquestions": subquestions}  # 交给 Send 并行分派。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return split  # 返回可注册到子图的节点。


def make_researcher(fast_llm: Any):  # 创建单个子问题的研究节点。
    """只根据模型已有知识分析子问题，不调用外部检索。"""

    @timed_node  # 记录各并行任务的耗时。
    async def researcher(state: dict[str, Any]) -> dict[str, list[dict[str, str]]]:  # 返回一项研究发现。
        with trace_span("topic.researcher") as node_span:
            try:
                subquestion = state["subquestion"]  # 读取本节点负责的问题。
                prompt = (  # 明确第一版的证据边界。
                    "你是主题研究员。只根据已有知识分析下面的子问题，给出主要观点、依据和不确定性。"
                    "你没有联网或读取外部资料，不得声称做过检索，不得编造引用、来源、当前数据或最新事件。"
                    "对需要实时核实的事实写明待核实。"
                    f"\n原主题：{state['question']}\n子问题：{subquestion}"  # 同时提供整体和局部语境。
                )  # 完成研究提示词。
                with trace_generation("topic_researcher", model=settings.fast_model) as generation:  # 记录子问题模型调用。
                    response = await fast_llm.ainvoke(prompt)  # 并行调用快模型。
                    if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                        generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
                if not isinstance(response.content, str) or not response.content.strip():  # 检查研究结果。
                    raise ValueError(f"研究子问题未返回有效内容：{subquestion}")  # 不填充虚构发现。
                finding = {  # 将模型分析与证据状态一起保存。
                    "question": subquestion,  # 对应的研究子问题。
                    "analysis": response.content.strip(),  # 模型生成的分析文本。
                    "verification": "未联网核验",  # 明确这不是外部资料或实时事实。
                }  # 完成一项研究发现。
                log_event(_LOG, "research.finding.created", question=subquestion)  # 记录完成的子问题。
                return {"findings": [finding]}  # reducer 将并行结果合并。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return researcher  # 返回可注册到子图的节点。


def send_subquestions(state: AnalysisState) -> list[Send]:  # 分派拆出的全部问题。
    return [  # 每个问题交给同一个研究节点独立执行。
        Send("researcher", {"question": state["question"], "subquestion": item})  # 传入原主题和子问题。
        for item in state["subquestions"]  # 按规划结果产生三个任务。
    ]  # 返回并行任务列表。


def make_summarize(smart_llm: Any):  # 创建全部发现的汇总节点。
    """综合并行研究发现，并保留未核验边界。"""

    @timed_node  # 延续原图的节点观测。
    async def summarize(state: AnalysisState) -> dict[str, str]:  # 返回主题摘要。
        with trace_span("topic.summarize") as node_span:
            try:
                findings = state["findings"]  # 读取 reducer 合并的所有发现。
                if len(findings) != _SUBQUESTION_COUNT:  # 缺少并行结果时停止。
                    raise ValueError("主题研究发现数量不完整")  # 不用部分结果伪装完整研究。
                prompt = (  # 要求汇总模型保留事实边界。
                    "你是研究综合分析师。综合下列三个子问题的分析，归纳一致观点、分歧和待核实事项。"
                    "只使用提供的内容，不补造外部来源或最新事实。"
                    f"\n研究发现：{json.dumps(findings, ensure_ascii=False)}"  # 提供全部并行结果。
                )  # 完成汇总提示词。
                with trace_generation("topic_summarize", model=settings.smart_model) as generation:  # 记录汇总模型调用。
                    response = await smart_llm.ainvoke(prompt)  # 调用主模型生成摘要。
                    if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                        generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
                if not isinstance(response.content, str) or not response.content.strip():  # 摘要不能为空。
                    raise ValueError("主题研究汇总为空")  # 阻止空摘要进入写作。
                log_event(_LOG, "research.findings.merged", count=len(findings))  # 记录汇总数量。
                return {"research_summary": response.content.strip()}  # 把摘要交给父图。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return summarize  # 返回汇总节点。


def build_analysis_sub_graph(smart_llm: Any, fast_llm: Any):  # 保留原有拆题并行汇总子图。
    graph = StateGraph(AnalysisState)  # 创建研究子图。
    graph.add_node("split", make_split(fast_llm))  # 先拆分研究主题。
    graph.add_node("researcher", make_researcher(fast_llm))  # 并行分析各子问题。
    graph.add_node("summarize", make_summarize(smart_llm))  # 汇总全部发现。
    graph.add_edge(START, "split")  # 从拆题进入子图。
    graph.add_conditional_edges("split", send_subquestions)  # 使用 Send 分派三个任务。
    graph.add_edge("researcher", "summarize")  # 等待并行研究结束后汇总。
    graph.add_edge("summarize", END)  # 汇总后结束子图。
    return graph.compile(name="topic_research_subgraph")  # 返回可由父图调用的研究图。


def make_research_team(sub_graph: Any):  # 保留原有父图对子图的调用边界。
    """从用户消息取得主题，并把子图结果回写父图。"""

    @timed_node  # 记录并行研究整体耗时。
    async def research_team(state: SystemState) -> dict[str, Any]:  # 返回摘要和发现。
        with trace_span("topic.research_team") as node_span:
            try:
                question = _latest_question(state["messages"])  # 避免把 Supervisor 交接消息当问题。
                with trace_span("topic.research_subgraph"):
                    result = await sub_graph.ainvoke({"question": question, "findings": [], "scope": state["scope"]})  # 运行拆题并行汇总子图。
                return {"findings": result["findings"], "research_summary": result["research_summary"]}  # 回流父图。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return research_team  # 返回父图节点。


def make_writer(smart_llm: Any):  # 创建研究报告写作节点。
    """依据研究摘要写报告，并处理审稿意见。"""

    @timed_node  # 保留原有写作节点观测。
    async def writer(state: SystemState) -> dict[str, Any]:  # 返回报告消息。
        with trace_span("topic.writer") as node_span:
            try:
                question = _latest_question(state["messages"])  # 读取本轮原始研究主题。
                feedback = state.get("feedback", "")  # 重写时读取审稿意见。
                prompt = (  # 限定报告来源与结构。
                    "你是通用主题研究报告撰写者。依据下列研究摘要写中文报告，包含概述、核心观点、"
                    "不确定性和待核实事项。不得虚构参考来源、实时数据或已经完成的检索。"
                    f"\n用户问题：{question}\n研究摘要：{state['research_summary']}"  # 提供问题和汇总结果。
                    f"\n审稿修改意见：{feedback}"  # 首轮为空，重写时提供具体反馈。
                )  # 完成报告提示词。
                with trace_generation("topic_writer", model=settings.smart_model) as generation:  # 记录写作模型调用。
                    response = await smart_llm.ainvoke(prompt)  # 主模型撰写报告。
                    if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                        generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
                if not isinstance(response.content, str) or not response.content.strip():  # 报告必须有正文。
                    raise ValueError("主题研究报告为空")  # 不返回空报告。
                report = "资料范围：以下分析基于模型已有知识，未联网核验。\n\n" + response.content.strip()  # 固定展示资料边界。
                return {"report": report, "messages": [AIMessage(content=report)]}  # 保存最终报告及消息。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return writer  # 返回父图写作节点。


def make_reviewer(smart_llm: Any):  # 创建研究报告审稿节点。
    """审查报告完整性与证据边界，最多要求重写一次。"""

    @timed_node  # 保留审稿耗时日志。
    async def reviewer(state: SystemState) -> dict[str, Any]:  # 返回审稿决定和意见。
        with trace_span("topic.reviewer") as node_span:
            try:
                count = state.get("review_count", 0) + 1  # 记录当前审稿轮次。
                prompt = (  # 让审稿模型只评估已有报告。
                    "你是研究报告审稿人。检查报告是否回答原问题、结构完整、明确未联网核验，"
                    "并且没有声称查过外部来源或提供未经核实的实时数据。"
                    "第一行只能回答“合格”或“不合格”；不合格时第二行写具体修改意见。"
                    f"\n报告：{state['report']}"  # 提供实际生成的报告。
                )  # 完成审稿提示词。
                with trace_generation("topic_reviewer", model=settings.smart_model) as generation:  # 记录审稿模型调用。
                    response = await smart_llm.ainvoke(prompt)  # 主模型评估报告。
                    if isinstance(response.content, str):  # 仅文本输出可估算 token 用量。
                        generation.usage = {"input": estimate_tokens(prompt), "output": estimate_tokens(response.content)}  # 估算输入输出用量。
                if not isinstance(response.content, str) or not response.content.strip():  # 审稿结果必须可读。
                    raise ValueError("主题研究审稿结果为空")  # 不默认视为合格。
                lines = response.content.strip().splitlines()  # 读取首行判断和后续意见。
                decision = lines[0].strip(" *")  # 去除无关空白或 Markdown 强调符。
                if decision not in {"合格", "不合格"}:  # 只接受明确结论。
                    raise ValueError("主题研究审稿结论无效")  # 不猜测模糊回复。
                if decision == "不合格" and count >= _MAX_REVIEWS:  # 两轮审稿仍未通过时停止。
                    raise ValueError("主题研究报告经两轮审稿仍不合格")  # 不把失败报告当作合格结果。
                feedback = "\n".join(lines[1:]).strip() if decision == "不合格" else ""  # 仅不合格时保留修改意见。
                if decision == "不合格" and not feedback:  # 重写需要可执行的反馈。
                    raise ValueError("主题研究审稿未提供修改意见")  # 不执行无目标重写。
                return {"review_decision": "pass" if decision == "合格" else "rework", "review_count": count, "feedback": feedback}  # 更新审稿状态。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return reviewer  # 返回父图审稿节点。


def review_route(state: SystemState) -> str:  # 根据审稿结果保持原有条件边。
    if state["review_decision"] == "pass":  # 审稿通过后结束。
        return END  # 返回报告。
    return "writer"  # 带审稿意见重写一次。


def build_topic_research_agent(smart_llm: Any, fast_llm: Any):  # 使用明确名称构建主题研究入口。
    """把主题研究子图封装为 Supervisor 可调用的 Agent。"""
    sub_graph = build_analysis_sub_graph(smart_llm, fast_llm)  # 复用拆题并行汇总设计。
    builder = StateGraph(SystemState)  # 创建研究报告父图。
    builder.add_node("research_team", make_research_team(sub_graph))  # 子图研究节点。
    builder.add_node("writer", make_writer(smart_llm))  # 报告写作节点。
    builder.add_node("reviewer", make_reviewer(smart_llm))  # 有界审稿节点。
    builder.add_edge(START, "research_team")  # 从用户问题开始研究。
    builder.add_edge("research_team", "writer")  # 研究完成后写报告。
    builder.add_edge("writer", "reviewer")  # 每份报告都进入审稿。
    builder.add_conditional_edges("reviewer", review_route)  # 通过或重写。
    return builder.compile(name="topic_research_agent")  # Supervisor 以此名称分派普通主题。


async def main() -> None:  # 保留原有命令行入口。
    """独立运行一次通用主题研究。"""
    question = " ".join(sys.argv[1:]).strip() or "分析人工智能对教育行业的影响"  # 读取用户研究主题。
    agent = build_topic_research_agent(make_smart_llm(), make_fast_llm())  # 创建主题研究图。
    with start_trace("topic_research.cli", question_chars=len(question)) as trace:  # 为命令行运行建立根 trace。
        result = await agent.ainvoke({"messages": [HumanMessage(content=question)], "scope": {"as_of": datetime.now(timezone.utc).isoformat()}, "findings": []})  # 直接运行父图。
        trace.output = {"status": "done", "report_chars": len(result["report"])}  # 只记录结果状态和长度。
        print(result["report"])  # 输出已通过审稿的报告。


if __name__ == "__main__":  # 仅在直接运行模块时执行示例。
    asyncio.run(main())  # 启动异步图。
