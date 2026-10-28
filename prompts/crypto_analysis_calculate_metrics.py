"""加密资产指标执行节点提示词。"""

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from prompts.common import build_system_prompt


class CalculateMetricsPrompt:
    instruction = """\
# Role / 身份
你是加密资产指标执行员，负责调用已注册工具获取真实数据并完成指标计算。

# Objective / 目标
通过工具完成本轮 Skill 指定的指标，所有数据和计算结果均来自工具。
成功标准：调用与指标匹配的工具，保留真实结果，不补造缺失数值。

# Tool Rules / 工具规则
- **范围限定**：仅调用完成本轮指标所需的工具。
- **组合工具**：优先使用组合工具，调用后不得再单独调用其底层取数或计算工具。

# Constraints / 约束
- 不自行编写代码或使用模型内部知识进行数值计算。
- 无法取得真实证据时，不补造、估算或修改任何数值。

# Workflow / 执行流程
1. 根据指标清单选择工具，按工具 description 填写参数并调用。
2. 若返回参数错误，按该工具的 description 修正后重试，不猜测参数。

# Output Format / 输出格式
只输出工具调用动作及工具原始结果，不添加主观文本。
"""

    example = ""

    user_prompt = "完成以下所有指标：\n{indicator_list}"

    system_prompt = build_system_prompt(instruction, example)
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("history"),
            ("human", user_prompt),
        ]
    )


def calculate_metrics_messages(
    skill_config: dict[str, Any],
    *,
    history: list[BaseMessage] | None = None,
) -> list[BaseMessage]:
    """保留历史消息并追加本轮指标清单，仅在历史为空时加入系统规则。"""
    indicator_list = "\n".join(
        f"- {metric['id']}" for metric in skill_config["scoring"]["metrics"]
    )
    if not history:
        history = [SystemMessage(content=CalculateMetricsPrompt.system_prompt)]
    return CalculateMetricsPrompt.prompt.format_messages(
        history=history,
        indicator_list=indicator_list,
    )
