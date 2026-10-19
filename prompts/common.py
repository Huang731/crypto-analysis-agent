"""提示词公共组装方法。"""


def build_system_prompt(
    instruction: str = "",
    example: str = "",
    pydantic_schema: str = "",
) -> str:
    """拼接任务指令、输出结构和参考示例。"""
    delimiter = "\n\n---\n\n"
    schema = ""

    if pydantic_schema:
        schema = (
            delimiter
            + "你的回答必须是 JSON，并严格遵循如下 Schema：\n"
            + f"```\n{pydantic_schema.strip()}\n```"
        )

    if example:
        example = delimiter + example.strip()

    return instruction.strip() + schema + example
