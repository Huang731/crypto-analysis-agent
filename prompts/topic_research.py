"""通用主题研究各节点的提示词。动态字段只做原文拼接。"""


def split_prompt(question: str) -> str:
    """约束模型只返回研究问题。"""
    return (
        "你是通用主题研究规划员。将用户问题拆成三个互补、可独立分析的子问题。"
        "每行只写一个问题，不写说明、标题或引用。不要假装已经联网检索。"
        f"\n用户问题：{question}"
    )


def research_prompt(question: str, subquestion: str) -> str:
    """明确第一版的证据边界。"""
    return (
        "你是主题研究员。只根据已有知识分析下面的子问题，给出主要观点、依据和不确定性。"
        "你没有联网或读取外部资料，不得声称做过检索，不得编造引用、来源、当前数据或最新事件。"
        "对需要实时核实的事实写明待核实。"
        f"\n原主题：{question}\n子问题：{subquestion}"
    )


def summarize_prompt(findings_json: str) -> str:
    """要求汇总模型保留事实边界。"""
    return (
        "你是研究综合分析师。综合下列三个子问题的分析，归纳一致观点、分歧和待核实事项。"
        "只使用提供的内容，不补造外部来源或最新事实。"
        f"\n研究发现：{findings_json}"
    )


def writer_prompt(question: str, research_summary: str, feedback: str) -> str:
    """限定报告来源与结构。"""
    return (
        "你是通用主题研究报告撰写者。依据下列研究摘要写中文报告，包含概述、核心观点、"
        "不确定性和待核实事项。不得虚构参考来源、实时数据或已经完成的检索。"
        f"\n用户问题：{question}\n研究摘要：{research_summary}"
        f"\n审稿修改意见：{feedback}"
    )


def review_prompt(report: str) -> str:
    """让审稿模型只评估已有报告。"""
    return (
        "你是研究报告审稿人。检查报告是否回答原问题、结构完整、明确未联网核验，"
        "并且没有声称查过外部来源或提供未经核实的实时数据。"
        "第一行只能回答“合格”或“不合格”；不合格时第二行写具体修改意见。"
        f"\n报告：{report}"
    )
