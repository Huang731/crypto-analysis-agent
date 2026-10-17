"""审核结果相关性评测使用的任务描述。"""


def reviewer_evaluation_question(asset: str, question: str, report: str) -> str:
    """相关性对应审核任务，避免把审核结果当成投资报告。"""
    return (
        f"请审核关于{asset}的以下报告是否回应用户问题、是否符合本轮证据，"
        "给出核查过程、是否通过和修改意见。\n"
        f"用户问题：{question}\n待审核报告：\n{report}"
    )
