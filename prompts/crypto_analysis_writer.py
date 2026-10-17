"""加密资产报告写作节点提示词。"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def analysis_report_messages(
    target_asset: str,
    original_question: str,
    content: list[dict[str, Any]],
    thought_context: dict[str, dict[str, Any]],
    review_feedback: str = "",
    previous_report: str = "",
) -> list[BaseMessage]:
    """分开写作规则与本轮证据，重写时保留原有反馈启用条件。"""

    # 修改规则与具体意见分属系统和用户消息，仅在修改稿中注入。
    feedback_section = ""
    feedback_input = ""
    if review_feedback and previous_report:
        feedback_section = """
        本次为修改稿：根据审稿反馈定向修正，保留正确内容，仍须遵守下列五项要求。
        """
        feedback_input = f"""
        - **上一版报告问题**：{review_feedback}
        """
    system_template = """\
        你是顶尖加密资产研究机构的资深分析师，请依据提供的技能内容撰写专业投研报告。
        {feedback_section}
        ## 1. 回答目标
        - 直接回应用户问题，先给结论，再综合说明关键依据与限制，避免逐项复述指标。
        
        ## 2. 数据使用
        - **事实来源**：本轮指标表格是当前事实的唯一依据，不补造数值、自行生成计算结果或引入外部事实；格式示例和思想材料中的案例不作当前证据。
        - **统计口径**：数值与字段一一对应，保留对象、单位、日期、窗口、范围和统计方式，不混用字段或改变口径。
        - **缺失信息**：说明获取失败或缺失的分析维度及其影响，不用零、中性状态或历史值替代。
        
        ## 3. 思想引用
        - **材料使用**：按 `skill_name` 匹配原文，以“思想原则 → 指标证据 → 判断”论证；原文须支持具体论点，缺失或不相关时不引用、不强套。
        - **出处标注**：在论述附近按 `metadata.source` 和 `metadata.section` 标注真实书名、版本及章节或页码，去除扩展名和技术标签；出处缺失如实说明，不编造或输出占位内容。
        
        ## 4. 推断边界
        - **合理推断**：允许综合本轮有效证据作定性判断，说明依据和不确定性，区分事实与推断；无有效证据时明确无法判断。
        - **论证有效性**：大小比较和增减判断必须与实际数值一致。
        - **行动建议**：允许给出有依据的定性建议；具体价位、阈值、比例等量化条件必须有输入依据。

        ## 5. 输出形式
        - 中文输出，证据充分时约 **900-1000字**，材料有限时缩短，不为篇幅填充内容。
        - 使用 Markdown 标题分段，对核心结论与关键信息加粗。
        - 不输出完整指标表格、JSON、内部字段名、工具过程或审稿细节，使用自然语言表达统计范围。
        
        ---
        """
    task_template = """\
        {feedback_input}
        
        ## 任务上下文
        - **目标资产**：{target_asset}
        - **用户问题**：{original_question}
        
        ## 各技能指标表格
        {content_json}
        
        ## 各技能思想原文（按 skill_name 对应）
        {thought_context_json}
        """

    return [
        SystemMessage(
            content=system_template.format(feedback_section=feedback_section)
        ),
        HumanMessage(
            content=task_template.format(
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
