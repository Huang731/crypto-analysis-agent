"""加密资产报告写作节点提示词。"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def analysis_report_messages(
    target_asset: str,
    original_question: str,
    content: list[dict[str, Any]],
    thought_context: dict[str, dict[str, Any]],
    history_summary: str,
    review_feedback: str = "",
    previous_report: str = "",
) -> list[BaseMessage]:
    """分开写作规则与本轮证据，重写时保留原有反馈启用条件。"""

    # 修改规则与具体意见分属系统和用户消息，仅在修改稿中注入。
    feedback_section = ""
    feedback_input = ""
    if review_feedback and previous_report:
        feedback_section = """
        ## 审稿反馈与重写指引（本次任务为修改稿）
        - **修改原则**：仅针对上述问题进行定向修正，保留上一版中正确的数据引用、思想论证结构和排版格式。
        - **禁止行为**：不得为满足审稿意见而编造新指标、曲解思想原文或引入外部事实。若意见要求超出当前数据/思想范围，请在报告中如实说明局限。
        """
        feedback_input = f"""
        - **上一版报告问题**：{review_feedback}
        """
    system_template = """\
        你是顶尖加密资产研究机构的资深分析师，请依据提供的技能内容撰写专业投研报告。
        {feedback_section}
        ## 1. 回答目标与风格
        - **结论先行**：首段必须用 1-2 句话直接给出核心结论（如适合/不适合加仓，或明确的趋势判断），然后再展开分析。
        
        ## 2. 指标表格输入使用
        - **数值依据**：以指标表格为准，保留单位、统计范围。**严禁**自行计算、补造数值或引入外部事实。
        
        ## 3. 各技能思想原文引用
        - **关联匹配**：按 `skill_name` 将各技能表格与对应的思想材料进行匹配。
        - **论证结构**：按“思想原则 → 当前指标如何符合或不符合 → 对判断的影响”组织论证。
        - **引用与出处**：引用思想材料时，必须在对应论述附近标注出处，不得只写作者姓名。书名取自对应文档的 `metadata.source`，章节或页码取自 `metadata.section`；保留原有书名、版本及定位信息，不得编造，出处缺失时如实说明。
          使用“（来源：《实际书名》，实际章节或第几页）”。去除文件扩展名和重复书名号，将章节路径转成自然中文，将 P数字 转成“第数字页”，不展示 image、table 等技术标签。
        - **诚实原则**：未提供原文时不声称已核验；材料受限时如实说明，不强套理论。
        
        ## 4. 输出格式与排版
        - 用中文输出约 **900-1000字** 的深度报告，合理使用 Markdown 标题（如 ###）进行分段。
        - **视觉焦点**：对**核心结论、关键指标数值、重要支撑/阻力位、思想框架名称**使用 Markdown 加粗（**文本**），提升阅读效率。
        - **禁止输出**：完整指标表格、JSON 代码块、内部字段名（indicator_id, metadata, 条记录等）、工具调用过程或审稿交互细节。
        
        ---
        """
    task_template = """\
        {feedback_input}
        
        ## 任务上下文
        - **目标资产**：{target_asset}
        - **用户问题**：{original_question}
        - **此前结论摘要**：
        {history_summary}
        
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
                history_summary=history_summary or "（无历史摘要）",
                content_json=json.dumps(content, ensure_ascii=False, indent=2),
                thought_context_json=json.dumps(
                    thought_context, ensure_ascii=False, indent=2
                ),
            )
        ),
    ]
