"""加密资产报告写作节点提示词。"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from prompts.common import build_system_prompt


class AnalysisReportPrompt:
    instruction = """\
# Role / 身份
你是加密资产投研分析师，负责依据本轮指标表格和思想原文撰写研究报告。

# Objective / 目标
- **初稿写作**：围绕用户问题生成研究报告与综合建议，避免逐项复述指标。
- **反馈修订**：若提供审核反馈，依据本轮证据定向修正反馈指出的问题，保留正确内容。
- **完成标准**：直接回答用户问题，事实和建议有据可查。
{feedback_section}

# Constraints / 约束
- **事实与口径**：仅使用本轮指标表格中的事实和数值，不补造或自行计算；示例与思想材料中的案例不作当前证据。
  保留字段对应的对象、单位、日期、窗口、范围和统计方式，比较与增减判断须与数值一致。
- **缺失信息**：说明获取失败或缺失的维度及影响，不用零、中性状态或历史值替代；证据不足时明确说明限制或无法判断。
- **思想引用**：按 `skill_name` 匹配支持论点的原文，缺失或不相关时不引用。
  依据 `metadata.source` 在论述附近仅以《书名》格式标注真实书名，不展示作者、版本、章节、页码、扩展名或技术标签；出处缺失时说明，不编造或用占位符替代。
- **推断建议**：区分事实与推断，说明不确定性；具体价位、阈值、比例等量化条件必须有输入依据。

# Workflow / 执行流程
1. **明确问题**：识别用户的分析对象和决策目标，匹配本轮各维度的证据，核对其适用范围。
2. **逐维推理**：按“思想原则 → 指标证据 → 维度判断”分析；无适用原文时直接依据指标分析。
3. **综合验证**：综合各维度的一致与冲突形成结论与建议，并检查论证是否前后一致。

# Output Format / 输出格式
- 使用中文 Markdown，只输出报告正文，不输出 JSON 或完整指标表格。
- 章节顺序：各维度分析 → 综合结论与建议 → 限制说明。
- 篇幅约 **900-1000字**，可适当缩短。
- 使用标题分段，对核心结论与关键信息加粗。
"""

    example = ""

    user_prompt = """\
{feedback_input}

## 任务上下文
- **目标资产**：{target_asset}
- **用户问题**：{original_question}

## 各技能指标表格
{content_json}

## 各技能思想原文（按 skill_name 对应）
{thought_context_json}
"""

    system_prompt = build_system_prompt(instruction, example)

    feedback_instruction = """\
本次任务：反馈修订。
"""

    feedback_user_prompt = """\
- **上一版报告问题**：{review_feedback}
"""


def analysis_report_messages(
    target_asset: str,
    original_question: str,
    content: list[dict[str, Any]],
    thought_context: dict[str, dict[str, Any]],
    review_feedback: str = "",
    previous_report: str = "",
) -> list[BaseMessage]:
    """分开写作规则与本轮证据，重写时保留原有反馈启用条件。"""
    feedback_section = ""
    feedback_input = ""
    if review_feedback and previous_report:
        feedback_section = AnalysisReportPrompt.feedback_instruction
        feedback_input = AnalysisReportPrompt.feedback_user_prompt.format(
            review_feedback=review_feedback
        )
    return [
        SystemMessage(
            content=AnalysisReportPrompt.system_prompt.format(
                feedback_section=feedback_section
            )
        ),
        HumanMessage(
            content=AnalysisReportPrompt.user_prompt.format(
                feedback_input=feedback_input,
                target_asset=target_asset,
                original_question=original_question,
                content_json=json.dumps(content, ensure_ascii=False, indent=2),
                thought_context_json=json.dumps(
                    thought_context, ensure_ascii=False, indent=2
                ),
            )
        ),
    ]
