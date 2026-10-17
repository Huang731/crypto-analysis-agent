"""LLM 评测候选采集：保存节点名称、实际输入和输出。"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class NodeCandidate(BaseModel):
    """保留节点名称、LLM 输入和实际输出。"""

    model_config = ConfigDict(extra="forbid")
    node: str
    input: Any
    actual_output: Any


def append_node_candidate(
    *,
    node: str,
    input: Any,
    actual_output: Any,
    path: str | Path,
) -> None:
    """向指定路径追加节点输入输出，保留中文。"""
    candidate = NodeCandidate(
        node=node,
        input=input,
        actual_output=actual_output,
    )
    line = candidate.model_dump_json(ensure_ascii=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(line)
