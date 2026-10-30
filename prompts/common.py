"""提示词公共组装方法。"""


def build_system_prompt(instruction: str = "", example: str = "") -> str:
    """拼接任务指令和参考示例。"""
    delimiter = "\n\n---\n\n"

    if example:
        example = delimiter + example.strip()

    return instruction.strip() + example
