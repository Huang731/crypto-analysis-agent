"""将标准化指标渲染为中文 Markdown 表格。"""

import json
from html import escape
from typing import Any

from app.crypto_analysis.metric_service import metric_field_labels
from app.crypto_analysis.skill_loader import LoadedSkill


def _table_cell(value: Any) -> str:
    """保留空值、有效零值与列表，并转义表格中的特殊字符。"""
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return "<br>".join(_table_cell(item) for item in value) if value else "[]"
    if isinstance(value, (bool, dict)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return (
        escape(text, quote=False)
        .replace("\\", "\\\\")
        .replace("|", "&#124;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def render_metric_tables(skill: LoadedSkill, indicators: list[dict[str, Any]]) -> str:
    """按输入指标和字段定义顺序生成表格，只展示计算成功的指标。"""
    metric_names = {metric.id: metric.label for metric in skill.scoring.metrics}
    tables: list[str] = []
    for indicator in indicators:
        if indicator["status"] != "ok":
            continue
        metric_id = indicator["indicator_id"]
        values = {**indicator["raw_values"], **indicator["calculated_values"]}
        rows: list[tuple[str, Any]] = [("指标标识", metric_id)]
        rows.extend(
            (label, values[field])
            for field, label in metric_field_labels(metric_id).items()
            if field in values
        )
        rows.extend(
            [
                ("单位", indicator["unit"]),
                ("观测日期", indicator["observed_at"]),
                ("来源链接", indicator["url"]),
            ]
        )
        lines = [
            f"### {_table_cell(metric_names[metric_id])}",
            "",
            "| 字段 | 内容 |",
            "| --- | --- |",
        ]
        lines.extend(
            f"| {_table_cell(label)} | {_table_cell(value)} |" for label, value in rows
        )
        tables.append("\n".join(lines))
    return "\n\n".join(tables)
