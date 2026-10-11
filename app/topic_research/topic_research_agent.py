"""通用主题研究 Agent：拆题、并行分析、汇总、写作和审稿。"""

import asyncio  # 运行现有命令行入口。
import json  # 将并行发现完整交给汇总模型。
import re  # 清理拆题结果中的常见编号。
import sys  # 读取命令行中的研究主题。
from collections.abc import Awaitable, Callable  # 标注节点工厂返回的可调用对象。
from typing import Any  # 标注模型和消息集合。

import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import (  # 工具 Agent 的消息交互。
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph  # 保留原有父图和子图结构。
from langgraph.graph.state import CompiledStateGraph  # 标注已编译研究图。
from langgraph.types import Command, Send  # 并行分派研究子问题。
from playwright.async_api import Error as BrowserError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.browser_tool import BrowserContext, BrowserTool, build_browser_tools
from app.config.config import settings  # 读取模型配置用于成本估算。
from app.llm.model import make_fast_llm, make_smart_llm  # 复用项目现有模型工厂。
from app.observability import get_logger, log_event, timed_node  # 保留请求观测。
from app.topic_research.topic_research_state import (  # 使用主题研究专属状态。
    AnalysisState,
    ResearchAgentState,
    SystemState,
)
from app.tracing import (  # 记录请求与模型调用。
    start_trace,
    trace_generation_call,
    trace_span,
)
from app.web_search_tool import build_web_search_tools
from prompts.topic_research import (  # 提示词与节点逻辑分离。
    RESEARCH_SYSTEM_PROMPT,
    research_prompt,
    review_prompt,
    split_prompt,
    summarize_prompt,
    writer_prompt,
)

_LOG = get_logger("crypto_analysis.crypto_analysis_agent")  # 沿用原有日志名称。
_NUMBER_PREFIX = re.compile(
    r"^\s*(?:\d+[.)、]\s*|[-*]\s+)"
)  # 识别模型偶尔生成的行首编号。
_SUBQUESTION_COUNT = 3  # 第一版固定并行研究三个子问题。
_MAX_REVIEWS = 2  # 审稿最多执行两轮，避免图无限循环。


def _latest_question(
    scope: dict[str, str],
) -> str:  # 校验本轮原始问题并读取已补全的研究问题。
    original = scope["original_question"]
    if not isinstance(original, str) or not original.strip():
        raise ValueError("主题研究缺少有效用户问题")
    resolved = scope["resolved_question"]
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError("主题研究缺少有效补全问题")
    return resolved.strip()  # 拆题、写作和审稿使用相同的研究主题。


def make_split(
    fast_llm: Any,
) -> Callable[
    [AnalysisState], Awaitable[dict[str, list[str]]]
]:  # 创建研究主题拆解节点。
    """把用户主题拆成三个可独立分析的子问题。"""

    @timed_node  # 沿用现有节点耗时和失败日志。
    async def split(state: AnalysisState) -> dict[str, list[str]]:  # 返回并行研究任务。
        with trace_span("topic.split") as node_span:
            try:
                prompt = split_prompt(state["question"])  # 提供原始研究主题。
                response = await _invoke_traced_model(
                    fast_llm, prompt, "topic_split", settings.fast_model
                )
                if not isinstance(response.content, str):  # 消息内容必须是文本。
                    raise TypeError("主题拆解模型未返回文本")  # 不猜测非文本内容。
                subquestions = [  # 按行读取研究任务。
                    _NUMBER_PREFIX.sub("", line).strip()  # 只清理编号和空白。
                    for line in response.content.splitlines()  # 遍历模型返回的每一行。
                    if line.strip()  # 忽略纯空白行。
                ]  # 完成子问题列表。
                if (
                    len(subquestions) != _SUBQUESTION_COUNT
                    or len(set(subquestions)) != _SUBQUESTION_COUNT
                    or any(not item for item in subquestions)
                ):  # 要求恰好三个不同的问题。
                    raise ValueError(
                        "主题拆解必须返回三个非空且不同的子问题"
                    )  # 拒绝缺漏或重复任务。
                return {"subquestions": subquestions}  # 交给 Send 并行分派。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return split  # 返回可注册到子图的节点。


@retry(
    stop=stop_after_attempt(2),
    wait=wait_random_exponential(min=1, max=2),
    reraise=True,
)
async def _invoke_model(llm: Any, prompt: str) -> Any:
    """每次调用保留超时，失败最多尝试两次，耗尽后向上报告。"""
    return await asyncio.wait_for(
        llm.ainvoke(prompt),
        timeout=120,
    )


@trace_generation_call(
    name=lambda arguments: arguments.arguments["trace_name"],
    model=lambda arguments: arguments.arguments["model_name"],
    output_attr="content",
    skip_non_text_output=True,
)
async def _invoke_traced_model(
    llm: Any, prompt: str, trace_name: str, model_name: str
) -> Any:
    """保留整体重试统计与非文本用量未知的处理。"""
    return await _invoke_model(llm, prompt)


class ResearchModelTrace(AgentMiddleware):
    """逐轮记录工具决策模型调用，并限制单轮等待时间。"""

    @trace_generation_call(
        name="topic_researcher",
        model=settings.fast_model,
        input_arg="request",
        input_format="model_request",
        output_attr="result",
        output_format="messages",
    )
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await asyncio.wait_for(handler(request), timeout=60)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        with trace_span("topic.research_tool", tool=request.tool_call["name"]) as span:
            try:
                return await handler(request)
            except (
                BrowserError,
                httpx.HTTPError,
                TimeoutError,
                ValueError,
                RuntimeError,
                KeyError,
                TypeError,
            ) as exc:
                span.metadata["status"] = "error"
                error = {
                    "tool": request.tool_call["name"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return Command(
                    update={
                        "tool_errors": [error],
                        "messages": [
                            ToolMessage(
                                content=json.dumps(error, ensure_ascii=False),
                                tool_call_id=request.tool_call["id"],
                                status="error",
                            )
                        ],
                    }
                )


def build_research_agent(fast_llm: Any) -> CompiledStateGraph:
    """复用工具图，每次执行通过上下文注入独立浏览会话。"""
    return create_agent(
        model=fast_llm,
        tools=[*build_web_search_tools(), *build_browser_tools()],
        state_schema=ResearchAgentState,
        context_schema=BrowserContext,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=settings.browser_max_steps, exit_behavior="error"
            ),
            ToolCallLimitMiddleware(
                run_limit=settings.browser_max_steps * 2, exit_behavior="error"
            ),
            ModelRetryMiddleware(
                max_retries=1, retry_on=(Exception,), on_failure="error"
            ),
            ResearchModelTrace(),
        ],
        name="topic_evidence_agent",
    )


def make_researcher(
    fast_llm: Any,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, list[dict[str, Any]]]]]:
    """让研究 Agent 搜索、浏览原文，再根据实际工具证据分析子问题。"""
    research_agent = build_research_agent(fast_llm)

    @timed_node
    async def researcher(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        with trace_span("topic.researcher") as node_span:
            try:
                subquestion = state["subquestion"]
                prompt = research_prompt(state["question"], subquestion)
                async with BrowserTool() as browser:
                    result = await research_agent.ainvoke(
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "search_records": {},
                            "research_evidence": {},
                            "tool_errors": [],
                        },
                        context=BrowserContext(browser=browser),
                        config={"recursion_limit": settings.browser_max_steps * 4 + 10},
                    )
                response = result["messages"][-1]
                if (
                    not isinstance(response, AIMessage)
                    or not isinstance(response.content, str)
                    or not response.content.strip()
                ):
                    raise ValueError(f"研究子问题未返回有效内容：{subquestion}")
                evidence = list(result["research_evidence"].values())
                searches = list(result["search_records"].values())
                verification = (
                    "已提取网页原文"
                    if evidence
                    else "仅有搜索摘要，未提取原文"
                    if searches
                    else "未取得外部证据"
                )
                finding = {
                    "question": subquestion,
                    "analysis": response.content.strip(),
                    "verification": verification,
                    "evidence": evidence,
                    "search_records": searches,
                    "tool_errors": result["tool_errors"],
                }
                log_event(
                    _LOG,
                    "research.finding.created",
                    question=subquestion,
                    evidence_count=len(evidence),
                )
                return {"findings": [finding]}
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return researcher


@timed_node
def send_subquestions(state: AnalysisState) -> list[Send]:  # 分派拆出的全部问题。
    return [  # 每个问题交给同一个研究节点独立执行。
        Send(
            "researcher",
            {
                "question": state["question"],
                "subquestion": item,
                "scope": state["scope"],
            },
        )  # 传入原主题和子问题。
        for item in state["subquestions"]  # 按规划结果产生三个任务。
    ]  # 返回并行任务列表。


def make_summarize(
    smart_llm: Any,
) -> Callable[[AnalysisState], Awaitable[dict[str, str]]]:  # 创建全部发现的汇总节点。
    """综合并行研究发现，并保留原文来源与证据不足的边界。"""

    @timed_node  # 延续原图的节点观测。
    async def summarize(state: AnalysisState) -> dict[str, str]:  # 返回主题摘要。
        with trace_span("topic.summarize") as node_span:
            try:
                findings = state["findings"]  # 读取 reducer 合并的所有发现。
                if len(findings) != _SUBQUESTION_COUNT:  # 缺少并行结果时停止。
                    raise ValueError(
                        "主题研究发现数量不完整"
                    )  # 不用部分结果伪装完整研究。
                prompt = summarize_prompt(
                    json.dumps(findings, ensure_ascii=False)
                )  # 提供全部并行结果。
                response = await _invoke_traced_model(
                    smart_llm, prompt, "topic_summarize", settings.smart_model
                )
                if (
                    not isinstance(response.content, str)
                    or not response.content.strip()
                ):  # 摘要不能为空。
                    raise ValueError("主题研究汇总为空")  # 阻止空摘要进入写作。
                log_event(
                    _LOG, "research.findings.merged", count=len(findings)
                )  # 记录汇总数量。
                return {
                    "research_summary": response.content.strip()
                }  # 把摘要交给父图。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return summarize  # 返回汇总节点。


def build_analysis_sub_graph(
    smart_llm: Any, fast_llm: Any
) -> CompiledStateGraph:  # 保留原有拆题并行汇总子图。
    graph = StateGraph(AnalysisState)  # 创建研究子图。
    graph.add_node("split", make_split(fast_llm))  # 先拆分研究主题。
    graph.add_node("researcher", make_researcher(fast_llm))  # 并行分析各子问题。
    graph.add_node("summarize", make_summarize(smart_llm))  # 汇总全部发现。
    graph.add_edge(START, "split")  # 从拆题进入子图。
    graph.add_conditional_edges("split", send_subquestions)  # 使用 Send 分派三个任务。
    graph.add_edge("researcher", "summarize")  # 等待并行研究结束后汇总。
    graph.add_edge("summarize", END)  # 汇总后结束子图。
    return graph.compile(name="topic_research_subgraph")  # 返回可由父图调用的研究图。


def make_research_team(
    sub_graph: Any,
) -> Callable[
    [SystemState], Awaitable[dict[str, Any]]
]:  # 保留原有父图对子图的调用边界。
    """从本轮请求范围取得主题，并把子图结果回写父图。"""

    @timed_node  # 记录并行研究整体耗时。
    async def research_team(state: SystemState) -> dict[str, Any]:  # 返回摘要和发现。
        with trace_span("topic.research_team") as node_span:
            try:
                question = _latest_question(state["scope"])
                with trace_span("topic.research_subgraph"):
                    result = await sub_graph.ainvoke(
                        {"question": question, "findings": [], "scope": state["scope"]}
                    )  # 运行拆题并行汇总子图。
                return {
                    "findings": result["findings"],
                    "research_summary": result["research_summary"],
                }  # 回流父图。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return research_team  # 返回父图节点。


def make_writer(
    smart_llm: Any,
) -> Callable[[SystemState], Awaitable[dict[str, Any]]]:  # 创建研究报告写作节点。
    """依据研究摘要写报告，并处理审稿意见。"""

    @timed_node  # 保留原有写作节点观测。
    async def writer(state: SystemState) -> dict[str, Any]:  # 返回报告正文。
        with trace_span("topic.writer") as node_span:
            try:
                question = _latest_question(state["scope"])
                feedback = state.get("feedback", "")  # 重写时读取审稿意见。
                prompt = writer_prompt(
                    question,
                    state["research_summary"],
                    feedback,
                    json.dumps(state["findings"], ensure_ascii=False),
                )  # 首轮反馈为空，重写时提供具体意见。
                # 仅本节点关闭深度思考，不修改共享模型的配置。
                writer_llm = smart_llm.bind(
                    extra_body={"thinking": {"type": "disabled"}}
                )
                response = await _invoke_traced_model(
                    writer_llm, prompt, "topic_writer", settings.smart_model
                )
                if (
                    not isinstance(response.content, str)
                    or not response.content.strip()
                ):  # 报告必须有正文。
                    raise ValueError("主题研究报告为空")  # 不返回空报告。
                evidence_count = sum(
                    len(item["evidence"]) for item in state["findings"]
                )
                covered_count = sum(
                    bool(item["evidence"]) for item in state["findings"]
                )
                report = (
                    f"资料范围：提取 {evidence_count} 条网页原文，覆盖 {covered_count}/{len(state['findings'])} 个子问题。\n\n"
                    + response.content.strip()
                )
                return {
                    "report": report,
                }  # 保存最终报告。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return writer  # 返回父图写作节点。


def make_reviewer(
    smart_llm: Any,
) -> Callable[[SystemState], Awaitable[dict[str, Any]]]:  # 创建研究报告审稿节点。
    """审查报告完整性与证据边界，最多要求重写一次。"""

    @timed_node  # 保留审稿耗时日志。
    async def reviewer(state: SystemState) -> dict[str, Any]:  # 返回审稿决定和意见。
        with trace_span("topic.reviewer") as node_span:
            try:
                count = state.get("review_count", 0) + 1  # 记录当前审稿轮次。
                prompt = review_prompt(
                    state["report"],
                    _latest_question(state["scope"]),
                    json.dumps(state["findings"], ensure_ascii=False),
                )  # 提供实际生成的报告。
                response = await _invoke_traced_model(
                    smart_llm, prompt, "topic_reviewer", settings.smart_model
                )
                if (
                    not isinstance(response.content, str)
                    or not response.content.strip()
                ):  # 审稿结果必须可读。
                    raise ValueError("主题研究审稿结果为空")  # 不默认视为合格。
                lines = (
                    response.content.strip().splitlines()
                )  # 读取首行判断和后续意见。
                decision = lines[0].strip(" *")  # 去除无关空白或 Markdown 强调符。
                if decision not in {"合格", "不合格"}:  # 只接受明确结论。
                    raise ValueError("主题研究审稿结论无效")  # 不猜测模糊回复。
                if (
                    decision == "不合格" and count >= _MAX_REVIEWS
                ):  # 两轮审稿仍未通过时停止。
                    raise ValueError(
                        "主题研究报告经两轮审稿仍不合格"
                    )  # 不把失败报告当作合格结果。
                feedback = (
                    "\n".join(lines[1:]).strip() if decision == "不合格" else ""
                )  # 仅不合格时保留修改意见。
                if decision == "不合格" and not feedback:  # 重写需要可执行的反馈。
                    raise ValueError("主题研究审稿未提供修改意见")  # 不执行无目标重写。
                return {
                    "review_decision": "pass" if decision == "合格" else "rework",
                    "review_count": count,
                    "feedback": feedback,
                }  # 更新审稿状态。
            except Exception:
                node_span.metadata["status"] = "error"
                raise

    return reviewer  # 返回父图审稿节点。


@timed_node
def review_route(state: SystemState) -> str:  # 根据审稿结果保持原有条件边。
    if state["review_decision"] == "pass":  # 审稿通过后结束。
        return END  # 返回报告。
    return "writer"  # 带审稿意见重写一次。


def build_topic_research_agent(
    smart_llm: Any, fast_llm: Any
) -> CompiledStateGraph:  # 使用明确名称构建主题研究入口。
    """把主题研究子图封装为 路由 可调用的 Agent。"""
    sub_graph = build_analysis_sub_graph(smart_llm, fast_llm)  # 复用拆题并行汇总设计。
    builder = StateGraph(SystemState)  # 创建研究报告父图。
    builder.add_node("research_team", make_research_team(sub_graph))  # 子图研究节点。
    builder.add_node("writer", make_writer(smart_llm))  # 报告写作节点。
    builder.add_node("reviewer", make_reviewer(smart_llm))  # 有界审稿节点。
    builder.add_edge(START, "research_team")  # 从用户问题开始研究。
    builder.add_edge("research_team", "writer")  # 研究完成后写报告。
    builder.add_edge("writer", "reviewer")  # 每份报告都进入审稿。
    builder.add_conditional_edges("reviewer", review_route)  # 通过或重写。
    return builder.compile(name="topic_research_agent")  # 路由 以此名称分派普通主题。


async def main() -> None:  # 保留原有命令行入口。
    """独立运行一次通用主题研究。"""
    question = (
        " ".join(sys.argv[1:]).strip() or "分析人工智能对教育行业的影响"
    )  # 读取用户研究主题。
    agent = build_topic_research_agent(
        make_smart_llm(), make_fast_llm()
    )  # 创建主题研究图。
    with start_trace(
        "topic_research.cli", question_chars=len(question)
    ) as trace:  # 为命令行运行建立根 trace。
        result = await agent.ainvoke(
            {
                "scope": {"original_question": question, "resolved_question": question},
                "findings": [],
            }
        )  # 直接运行父图。
        trace.output = {
            "status": "done",
            "report_chars": len(result["report"]),
        }  # 只记录结果状态和长度。
        print(result["report"])  # 输出已通过审稿的报告。


if __name__ == "__main__":  # 仅在直接运行模块时执行示例。
    asyncio.run(main())  # 启动异步图。
