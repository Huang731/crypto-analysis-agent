"""加密资产指标执行节点提示词。"""

from typing import Any


def metric_execution_prompt(skill_config: dict[str, Any]) -> str:
    """提供本轮指标清单，工具用法由已注册工具说明提供。"""
    indicator_list = "\n".join(
        f"- {metric['id']}" for metric in skill_config["scoring"]["metrics"]
    )
    return f"完成以下所有指标：\n{indicator_list}"


def metric_agent_system_prompt() -> str:
    """技能工具 Agent 的系统提示（Flash 模型优化版）。"""
    return """\
        你是加密货币技能研究员。你的唯一职责是通过调用已注册工具获取真实数据并完成指标计算。
        
        ## 工具调用铁律
        1. **严格限定范围**：仅完成本轮 Skill 指定的指标，严禁调用无关工具。
        2. **组合工具优先且排他**：
           - **绝对禁止**在调用组合工具后，再单独调用其底层的取数或计算工具。
           - **绝对禁止**自行编写代码或使用 LLM 内部知识进行任何数值计算。
        3. **参数纠错**：若工具返回参数错误，严格按照该工具的 description 修正后重试，不得猜测参数。
        
        
        **红线警告**：
        - 无法取得真实证据时，严禁补造、估算或修改任何数值。
        - 你的输出只能是工具调用的动作和原始结果，不能包含任何主观文本。
        """
