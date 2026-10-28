"""根据成功指标生成理论检索问题，指标表格由业务代码渲染。"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from prompts.common import build_system_prompt


class SkillRetrievalPrompt:
    instruction = """\
# Objective / 目标
根据成功指标，生成一条用于检索书籍分析原理与解释方法的中文问题。

# Retrieval Rules / 检索规则
保留指标之间的关键关系与统计窗口，不引入缺失指标或无关分析维度。

# Constraints / 约束
只使用提供的指标数据，不补造数据、趋势、因果关系或分析结论。

# Output Format / 输出格式
只输出一条纯文本检索问题，最多 100 字符，不输出 JSON、代码块、指标表格或报告。
"""

    example = (
        "## 参考示例\n"
        "当价格高于50日及200日均线、两条均线较20日前抬升，但负向DI高于正向DI时，"
        "如何综合判断趋势与动量？"
    )

    user_prompt = """## 任务上下文
- 分析对象：{analysis_subject}
- 技能：{skill_name}
- 成功指标：
{indicators_json}
"""

    system_prompt = build_system_prompt(instruction, example)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            ("human", user_prompt),
        ]
    )


def skill_retrieval_messages(
    analysis_subject: str, skill_name: str, indicators: list[dict[str, Any]]
) -> list[BaseMessage]:
    """组装理论检索规则与本轮成功指标数据。"""
    return SkillRetrievalPrompt.prompt.format_messages(
        analysis_subject=analysis_subject,
        skill_name=skill_name,
        indicators_json=json.dumps(indicators, ensure_ascii=False, indent=2),
    )
