"""结构化输出的公共格式提示。"""

import json

from langchain_core.messages import SystemMessage
from pydantic import BaseModel


def structured_output_instruction(schema: type[BaseModel]) -> SystemMessage:
    """按当前模型定义生成 JSON 格式要求，避免业务提示词重复维护字段。"""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return SystemMessage(
        content=(
            "返回一个符合以下 JSON Schema 的 JSON 对象。"
            "不要包含 Markdown 代码块或额外说明。\n"
            f"{schema_json}"
        )
    )
