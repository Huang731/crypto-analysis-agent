"""节点评测候选采集：只保存实际输入输出，预期答案由人工标注。"""

from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class NodeCandidate(BaseModel):
    """保留节点名称、请求范围和实际执行结果。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(default_factory=lambda: uuid4().hex)
    node: str
    scope: dict[str, str]
    input: dict[str, Any]
    actual_output: Any


def append_node_candidate(
    *,
    node: str,
    scope: dict[str, str],
    input: dict[str, Any],
    actual_output: Any,
    path: str | Path,
) -> None:
    """整条序列化后一次追加，保留中文、模型消息及图更新结构。"""
    candidate = NodeCandidate(
        node=node,
        scope=scope,
        input=input,
        actual_output=actual_output,
    )
    line = candidate.model_dump_json(ensure_ascii=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line)
