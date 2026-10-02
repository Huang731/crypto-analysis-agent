"""使用 Supervisor 在维度分析与通用主题研究之间分派问题。"""

from typing import Annotated, Any  # 声明 Supervisor 共享状态。

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langgraph.graph.message import add_messages  # 累积 Supervisor 交接消息。
from langgraph.managed import RemainingSteps  # 提供 Supervisor 工具循环所需的剩余步数。
from langgraph_supervisor import create_supervisor  # 创建双 Agent 调度图。
from typing_extensions import NotRequired, TypedDict  # 声明跨 Agent 的状态字段。

from app.crypto_analysis.crypto_analysis_agent import (
    build_dynamic_analysis_agent,  # 创建维度加密货币分析图。
)
from app.topic_research.topic_research_agent import (
    build_topic_research_agent,  # 创建通用主题研究图。
)


class SupervisorState(TypedDict, total=False):  # 两个 Agent 共用的外层状态。
    messages: Annotated[list[Any], add_messages]  # 用户问题、交接消息和回答。
    remaining_steps: NotRequired[RemainingSteps]  # 供 LangGraph 控制工具循环。
    scope: dict[str, str]  # 整轮请求固定的截止时间和追踪标识。
    report: str  # 当前被分派 Agent 生成的报告。
    findings: list[Any]  # 子 Agent 已合并的事实，外层直接接收完整列表。
    dimension_results: list[dict[str, Any]]  # 子 Agent 已合并的维度结果。


def build_supervisor_system(smart: BaseChatModel, fast: BaseChatModel):  # 沿用 API 现有构建入口。
    """让 Supervisor 只在维度分析和主题研究之间分派。"""
    crypto_agent = build_dynamic_analysis_agent(smart, fast)  # 保留已实现的维度图。
    topic_agent = build_topic_research_agent(smart, fast)  # 将主题研究 Agent 接入调度图。
    workflow = create_supervisor(  # 创建实际运行的中央分派图。
        agents=[crypto_agent, topic_agent],  # 注册两个独立的研究职责。
        model=fast,  # 使用主模型判断用户问题意图。
        prompt=(  # 要求每轮只交给一个专业 Agent。
            "你是研究任务调度员，按问题类型三选一：\n"
            "1. 涉及加密货币行情、币种、投资判断或维度指标的问题，交给 crypto_analysis_agent；\n"
            "2. 其他值得展开分析或研究的主题（行业、技术、经济、历史等），交给 topic_research_agent；\n"
            "3. 寒暄闲聊、询问你的能力、需要实时信息的日常问题（如天气、新闻、时间）等不需要研究的问题，"
            "不要分派，直接用中文简短回复：能用常识回答的就回答；需要实时数据的说明你无法获取实时信息，"
            "并提示你可以分析加密货币或研究某个主题；介绍自己时称为“研究分析助手”。\n"
            "需要分派时每轮只能分派给一个 Agent；Agent 返回后直接结束，不再次分派。"
            "完整保留 Agent 的报告及其数据或资料边界，不补造来源、实时数据或评分。"
        ),  # 完成调度说明。
        state_schema=SupervisorState,  # 让请求范围和报告跨 Agent 传递。
        output_mode="last_message",  # 只把子 Agent 的最终消息交回调度中心。
        parallel_tool_calls=None,  # 智谱未声明此参数，不向兼容接口发送。
    )  # 返回尚未编译的 Supervisor 图。
    return workflow.compile(name="research_supervisor")  # 向现有 API 提供可运行图。
