"""通用主题研究 Agent：拆题、并行分析、汇总、写作和审稿。"""

import asyncio
import json  # 将并行发现完整交给汇总模型。
import re  # 清理拆题结果中的常见编号。
from collections.abc import Awaitable, Callable  # 标注节点工厂返回的可调用对象。
from typing import Any  # 标注模型和消息集合。

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (  # 工具 Agent 的消息交互。
    AIMessage,
    BaseMessage,
)
from langgraph.graph import END, START, StateGraph  # 保留原有父图和子图结构。
from langgraph.graph.state import CompiledStateGraph  # 标注已编译研究图。
from langgraph.types import Send  # 并行分派研究子问题。
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.config import settings  # 读取模型配置用于成本估算。
from app.core.observability import timed_node  # 保留节点耗时观测。
from app.core.tracing import trace_generation_call
from app.llm.model import (
    ainvoke_model,
    make_glm_4_5_air_llm,
    make_glm_4_flash_llm,
)
from app.topic_research.tools.browser_tool import (
    BrowserContext,
    BrowserTool,
    build_browser_tools,
)
from app.topic_research.tools.web_search_tool import build_web_search_tools
from app.topic_research.topic_research_state import (  # 使用主题研究专属状态。
    RESEARCH_EVIDENCE_TARGET,
    AnalysisState,
    ResearchAgentState,
    SystemState,
)
from prompts.topic_research import (  # 提示词与节点逻辑分离。
    research_continue_messages,
    research_messages,
    review_messages,
    split_messages,
    summarize_messages,
    writer_messages,
)

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


def make_split() -> Callable[
    [AnalysisState], Awaitable[dict[str, list[str]]]
]:  # 创建研究主题拆解节点。
    """把用户主题拆成三个可独立分析的子问题。"""
    glm_4_flash_llm = make_glm_4_flash_llm()

    @timed_node(label=lambda state: state["question"])  # 沿用现有节点耗时和失败日志。
    @trace_generation_call(name="topic.split", kind="span")
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    async def split(state: AnalysisState) -> dict[str, list[str]]:  # 返回并行研究任务。
        messages = split_messages(state["question"])  # 提供原始研究主题。
        response = await _invoke_traced_model(
            glm_4_flash_llm, messages, "topic_split", settings.glm_4_flash_model
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

    return split  # 返回可注册到子图的节点。


async def _invoke_model(llm: Any, messages: list[BaseMessage]) -> Any:
    """执行单次调用，重试由包含结果校验的业务节点负责。"""
    return await ainvoke_model(llm, messages)


@trace_generation_call(
    name=lambda arguments: arguments.arguments["trace_name"],
    model=lambda arguments: arguments.arguments["model_name"],
    input_arg="messages",
    input_format="messages",
    output_attr="content",
    skip_non_text_output=True,
    kind="generation",
)
async def _invoke_traced_model(
    llm: Any, messages: list[BaseMessage], trace_name: str, model_name: str
) -> Any:
    """记录单次调用统计，保留非文本用量未知的处理。"""
    return await _invoke_model(llm, messages)


class ResearchModelTrace(AgentMiddleware):
    """逐轮记录模型调用，并限制单轮等待时间。"""

    @trace_generation_call(
        name="topic_researcher",
        model=settings.glm_smart_model,
        input_arg="request",
        input_format="model_request",
        output_attr="result",
        output_format="messages",
        kind="generation",
    )
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(min=1, max=2),
            reraise=True,
        )
        async def invoke(current_request: ModelRequest) -> ModelResponse:
            """只重试当前模型请求，不重发工具调用或研究流程。"""
            return await asyncio.wait_for(handler(current_request), timeout=60)

        # 仅研究模型调用关闭深度思考，不修改共享 Air 模型。
        request = request.override(
            model_settings={
                **request.model_settings,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        )
        if len(request.state.get("research_evidence", {})) >= RESEARCH_EVIDENCE_TARGET:
            request = request.override(tools=[], tool_choice="none")
        response = await invoke(request)
        # 按工具名称和完整参数判断重复，调用 ID 不参与比较。
        seen = [
            (call["name"], call["args"])
            for message in request.messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        for message in response.result:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls:
                key = (call["name"], call["args"])
                if key in seen:
                    # 重复调用不写入历史，关闭工具后根据已有结果结束研究。
                    request = request.override(tools=[], tool_choice="none")
                    return await invoke(request)
                # 同步记录本批新调用，避免同一响应中重复请求相同工具和参数。
                seen.append(key)
        return response


def build_research_agent() -> CompiledStateGraph:
    """复用工具图，每次执行通过上下文注入独立浏览会话。"""
    glm_4_5_air_llm = make_glm_4_5_air_llm()
    return create_agent(
        model=glm_4_5_air_llm,
        tools=[*build_web_search_tools(), *build_browser_tools()],
        state_schema=ResearchAgentState,
        context_schema=BrowserContext,
        middleware=[
            ResearchModelTrace(),
        ],
        name="topic_evidence_agent",
    )


def make_researcher() -> Callable[
    [dict[str, Any]], Awaitable[dict[str, list[dict[str, Any]]]]
]:
    """让研究 Agent 搜索、浏览原文，再根据实际工具证据分析子问题。"""
    research_agent = build_research_agent()

    @timed_node(label=lambda state: state["subquestion"])
    @trace_generation_call(name="topic.researcher", kind="span")
    async def researcher(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        subquestion = state["subquestion"]
        messages = research_messages(
            state["question"], subquestion, as_of=state["scope"]["as_of"]
        )
        research_state: ResearchAgentState = {
            "messages": messages,
            "search_records": {},
            "research_evidence": {},
            "visited_urls": [],
            "tool_errors": [],
        }
        async with BrowserTool() as browser:
            while True:
                result = await research_agent.ainvoke(
                    research_state,
                    context=BrowserContext(browser=browser),
                )

                # 每个子问题至少取得一条原文，才允许结束取证。
                if len(result["research_evidence"]) >= RESEARCH_EVIDENCE_TARGET:
                    break

                visited_urls = set(result["visited_urls"])
                search_records = list(result["search_records"].values())

                # 从已有搜索结果中找出尚未尝试的地址，并按 URL 去重。
                candidate_urls = list(
                    dict.fromkeys(
                        item["url"]
                        for record in search_records
                        for item in record["results"]
                        if item["url"] not in visited_urls
                    )
                )
                searched_queries = list(
                    dict.fromkeys(record["query"] for record in search_records)
                )

                # 保留已有状态继续研究，不因证据不足抛出异常。
                research_state = {
                    **result,
                    "messages": [
                        *result["messages"],
                        *research_continue_messages(
                            candidate_urls,
                            searched_queries,
                        ),
                    ],
                }
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
        return {"findings": [finding]}

    return researcher


@timed_node(label=lambda state: state["question"])
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


def make_summarize() -> Callable[
    [AnalysisState], Awaitable[dict[str, str]]
]:  # 创建全部发现的汇总节点。
    """综合并行研究发现，并保留原文来源与证据不足的边界。"""
    glm_4_flash_llm = make_glm_4_flash_llm()

    @timed_node(label=lambda state: state["question"])  # 延续原图的节点观测。
    @trace_generation_call(name="topic.summarize", kind="span")
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    async def summarize(state: AnalysisState) -> dict[str, str]:  # 返回主题摘要。
        findings = state["findings"]  # 读取 reducer 合并的所有发现。
        if len(findings) != _SUBQUESTION_COUNT:  # 缺少并行结果时停止。
            raise ValueError("主题研究发现数量不完整")  # 不用部分结果伪装完整研究。
        messages = summarize_messages(
            json.dumps(findings, ensure_ascii=False)
        )  # 提供全部并行结果。
        response = await _invoke_traced_model(
            glm_4_flash_llm,
            messages,
            "topic_summarize",
            settings.glm_4_flash_model,
        )
        if (
            not isinstance(response.content, str) or not response.content.strip()
        ):  # 摘要不能为空。
            raise ValueError("主题研究汇总为空")  # 阻止空摘要进入写作。
        return {"research_summary": response.content.strip()}  # 把摘要交给父图。

    return summarize  # 返回汇总节点。


def build_analysis_sub_graph() -> CompiledStateGraph:  # 保留原有拆题并行汇总子图。
    graph = StateGraph(AnalysisState)  # 创建研究子图。
    graph.add_node("split", make_split())  # 先拆分研究主题。
    graph.add_node("researcher", make_researcher())  # 并行分析各子问题。
    graph.add_node("summarize", make_summarize())  # 汇总全部发现。
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

    @trace_generation_call(name="topic.research_subgraph", kind="span")
    async def invoke_subgraph(state: AnalysisState) -> dict[str, Any]:
        """记录子图执行过程，模型用量由子图内部的调用记录。"""
        return await sub_graph.ainvoke(state)

    @timed_node(
        label=lambda state: state["scope"]["resolved_question"]
    )  # 记录并行研究整体耗时。
    @trace_generation_call(name="topic.research_team", kind="span")
    async def research_team(state: SystemState) -> dict[str, Any]:  # 返回摘要和发现。
        question = _latest_question(state["scope"])
        result = await invoke_subgraph(
            {"question": question, "findings": [], "scope": state["scope"]}
        )  # 运行拆题并行汇总子图。
        return {
            # 专家状态跨轮保存，本轮证据和审核状态仍需独立初始化。
            "findings": result["findings"],
            "research_summary": result["research_summary"],
            "report": "",
            "review_decision": "pending",
            "review_count": 0,
            "feedback": "",
        }  # 回流父图。

    return research_team  # 返回父图节点。


def make_writer() -> Callable[
    [SystemState], Awaitable[dict[str, Any]]
]:  # 创建研究报告写作节点。
    """依据研究摘要写报告，并处理审稿意见。"""
    glm_4_5_air_llm = make_glm_4_5_air_llm()

    @timed_node(
        label=lambda state: state["scope"]["resolved_question"]
    )  # 保留原有写作节点观测。
    @trace_generation_call(name="topic.writer", kind="span")
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    async def writer(state: SystemState) -> dict[str, Any]:  # 返回报告正文。
        question = _latest_question(state["scope"])
        feedback = state.get("feedback", "")  # 重写时读取审稿意见。
        messages = writer_messages(
            question,
            state["research_summary"],
            feedback,
            json.dumps(state["findings"], ensure_ascii=False),
        )  # 首轮反馈为空，重写时提供具体意见。
        response = await _invoke_traced_model(
            glm_4_5_air_llm,
            messages,
            "topic_writer",
            settings.glm_smart_model,
        )
        if (
            not isinstance(response.content, str) or not response.content.strip()
        ):  # 报告必须有正文。
            raise ValueError("主题研究报告为空")  # 不返回空报告。
        return {
            "report": response.content.strip(),
        }  # 保存最终报告。

    return writer  # 返回父图写作节点。


def make_reviewer() -> Callable[
    [SystemState], Awaitable[dict[str, Any]]
]:  # 创建研究报告审稿节点。
    """审查报告完整性与证据边界，最多要求重写一次。"""
    glm_4_5_air_llm = make_glm_4_5_air_llm()

    @timed_node(
        label=lambda state: state["scope"]["resolved_question"]
    )  # 保留审稿耗时日志。
    @trace_generation_call(name="topic.reviewer", kind="span")
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    async def reviewer(state: SystemState) -> dict[str, Any]:  # 返回审稿决定和意见。
        count = state.get("review_count", 0) + 1  # 记录当前审稿轮次。
        messages = review_messages(
            state["report"],
            _latest_question(state["scope"]),
            json.dumps(state["findings"], ensure_ascii=False),
        )  # 提供实际生成的报告。
        response = await _invoke_traced_model(
            glm_4_5_air_llm,
            messages,
            "topic_reviewer",
            settings.glm_smart_model,
        )
        if (
            not isinstance(response.content, str) or not response.content.strip()
        ):  # 审稿结果必须可读。
            raise ValueError("主题研究审稿结果为空")  # 不默认视为合格。
        lines = response.content.strip().splitlines()  # 读取首行判断和后续意见。
        decision = lines[0].strip(" *")  # 去除无关空白或 Markdown 强调符。
        if decision not in {"合格", "不合格"}:  # 只接受明确结论。
            raise ValueError("主题研究审稿结论无效")  # 不猜测模糊回复。
        if (
            decision == "不合格" and count >= _MAX_REVIEWS
        ):  # 达到审稿上限后，沿用加密货币报告的强制通过处理。
            return {
                "review_decision": "pass",
                "review_count": count,
                "feedback": f"已达{_MAX_REVIEWS}轮审稿上限，强制通过。",
            }
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

    return reviewer  # 返回父图审稿节点。


@timed_node(label=lambda state: state["scope"]["resolved_question"])
def review_route(state: SystemState) -> str:  # 根据审稿结果保持原有条件边。
    if state["review_decision"] == "pass":  # 审稿通过后结束。
        return END  # 返回报告。
    return "writer"  # 带审稿意见重写一次。


def build_topic_research_agent() -> CompiledStateGraph:
    """把主题研究子图封装为 路由 可调用的 Agent。"""
    sub_graph = build_analysis_sub_graph()  # 复用拆题并行汇总设计。
    builder = StateGraph(SystemState)  # 创建研究报告父图。
    builder.add_node("research_team", make_research_team(sub_graph))  # 子图研究节点。
    builder.add_node("writer", make_writer())  # 报告写作节点。
    builder.add_node("reviewer", make_reviewer())  # 有界审稿节点。
    builder.add_edge(START, "research_team")  # 从用户问题开始研究。
    builder.add_edge("research_team", "writer")  # 研究完成后写报告。
    builder.add_edge("writer", "reviewer")  # 每份报告都进入审稿。
    builder.add_conditional_edges("reviewer", review_route)  # 通过或重写。
    return builder.compile(checkpointer=True, name="topic_research_agent")
