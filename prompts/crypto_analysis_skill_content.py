"""加密资产思想检索与指标表格生成节点提示词。"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def thought_retrieval_prompt(
    target_asset: str, skill_name: str, metric_labels: list[str]
) -> str:
    """按固定模板检索技能理论材料，不额外调用模型生成查询。"""
    metric_names = "、".join(metric_labels)
    return f"检索与{target_asset}的{skill_name}相关的知识。相关指标：{metric_names}。"


def skill_output_messages(
    analysis_subject: str,
    skill_name: str,
    indicators: list[dict[str, Any]],
) -> list[BaseMessage]:
    """分开表格展示规则与本轮成功指标数据。"""

    system_template = """\
        根据成功指标，生成中文 Markdown 指标表格。

        ## 输出规则
        1. 每个成功指标单独输出一个三级标题和一张连续的“字段／内容”两列表格。
        2. 每张表的表头只出现一次，**不要**拆分为原始值表和计算值表。
        3. 标题使用易懂的指标中文名称。
        4. 只使用提供的指标数据，**严禁**补造数据、趋势、来源或因果关系。
        5. 只输出完整的 Markdown 内容，不输出 JSON、字段路径或工具执行过程。

        ## 表格格式模板
        ### {{指标中文名称}}

        | 字段 | 内容 |
        | --- | --- |
        | 指标标识 | {{indicator_id}} |
        | {{field_labels 中对应原始字段的固定中文名称}} | {{对应原始值}} |
        | {{field_labels 中对应计算字段的固定中文名称}} | {{对应计算值}} |
        | 指标分数 | {{score}} |
        | 观测日期 | {{observed_at}} |
        | 来源链接 | {{url}} |

        ## 参考示例
        ### 趋势

        | 字段 | 内容 |
        | --- | --- |
        | 指标标识 | Trend |
        | 交易对 | BTCUSDT |
        | 首根日线开盘时间 | 2025-12-18 08:00:00（北京时间） |
        | 末根日线收盘时间 | 2026-10-14 07:59:59（北京时间） |
        | 最新收盘价（USDT） | 83169.0 |
        | 50日均线（USDT） | 81130.26 |
        | 200日均线（USDT） | 72166.65 |
        | 20日前50日均线（USDT） | 74558.47 |
        | 20日前200日均线（USDT） | 70794.82 |
        | 14日平均趋向指数ADX | 29.56 |
        | 正向方向指标DI | 21.71 |
        | 负向方向指标DI | 21.91 |
        | 指标分数 | 5 |
        | 观测日期 | 2026-10-13 |
        | 来源链接 | https://www.binance.com/ |"""
    task_template = """\
        ## 任务上下文
        - 分析对象：{analysis_subject}
        - 技能：{skill_name}
        - 成功指标：
        {indicators_json}"""

    return [
        SystemMessage(content=system_template.format()),
        HumanMessage(
            content=task_template.format(
                analysis_subject=analysis_subject,
                skill_name=skill_name,
                indicators_json=json.dumps(indicators, ensure_ascii=False, indent=2),
            )
        ),
    ]
