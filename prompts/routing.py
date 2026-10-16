"""研究任务调度员提示词。"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def routing_messages(question: str) -> list[BaseMessage]:
    """分开路由系统规则与本轮已改写问题。"""
    return [
        SystemMessage(content=routing_prompt()),
        HumanMessage(content=question),
    ]


def routing_prompt() -> str:
    """按任务目的选择加密分析或主题研究，只返回 Agent 名称。"""
    return """\
你是任务路由员，只输出以下两个名称之一：

- crypto_analysis_agent：加密资产的价格趋势、买卖仓位决策、链上或衍生品指标分析。
- topic_research_agent：其他所有情况，包括加密行业的公司、技术、政策研究，
  以及日常问答和意图不明确的问题。

按任务目的判断，不能仅凭“行情、投资、指标”等词判为加密分析。
只输出 Agent 名称，不输出 JSON 或解释。
"""
