"""RAG 图片提取与看图问答的提示词。"""

from langchain_core.messages import BaseMessage, HumanMessage

from prompts.common import build_system_prompt


class DescribeImagePrompt:
    instruction = """\
# Objective / 目标
请仔细分析这张图片，提取结构化信息用于知识库检索。

# Extraction Rules / 提取规则
1. 图表类型（柱状图/折线图/饼图/表格/示意图/其他）
2. 所有能读到的数值和它们对应的标签（如「Q1: 1800」「Q2: 2400」）
3. 坐标轴标签和单位
4. 关键趋势或对比结论（如「Q4 最高」「逐季上升」）
5. 如果是示意图/流程图，描述其中的文字和结构关系

# Output Format / 输出格式
用简洁的条目列出，不要写「这是一张...」式的废话开头。直接给数据和结论。
"""

    example = ""

    user_prompt = build_system_prompt(instruction, example)


class AnswerWithImagePrompt:
    user_prompt = """\
# Objective / 目标
请根据图片回答：{question}。

# Output Format / 输出格式
直接给答案，简明扼要。
"""


def describe_image_messages(image_url: str) -> list[BaseMessage]:
    """组装图片与描述要求，保留原有用户消息角色。"""
    return [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": DescribeImagePrompt.user_prompt},
            ]
        )
    ]


def answer_with_image_messages(image_url: str, question: str) -> list[BaseMessage]:
    """组装图片与用户问题，保留原有多模态内容顺序。"""
    return [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": AnswerWithImagePrompt.user_prompt.format(question=question),
                },
            ]
        )
    ]
