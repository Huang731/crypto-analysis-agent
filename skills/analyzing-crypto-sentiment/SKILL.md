---
name: analyzing-crypto-sentiment
description: >-
  按恐惧贪婪指数评估全市场情绪，适用于恐慌、贪婪和极端情绪分析；不代表单个币种的价格或资金。
  get_cmc_fear_greed → calculate_fear_greed_indicator，使用返回的 evidence_id。
tools:
  - get_cmc_fear_greed
  - calculate_fear_greed_indicator
scoring:
  name: 市场情绪
  metrics:
  - id: Fear_Greed
    label: Fear_Greed
    weight: 1.0
    kind: numeric
---

# 市场情绪

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### 市场情绪

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 恐惧与贪婪指数 | `raw_values.value` |
| 情绪分类 | `raw_values.value_classification` |
| 记录时间 | `raw_values.record_timestamp` |
| 信号参考 | `calculated_values.signal_reference` |
| 指标状态 | `status` |
| 状态原因 | `reason` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源工具 | `source.action` |
| 来源链接 | `source.url` |

分析仅依据工具结果，不重算或修改数值和分数。缺失值或不适用字段写“N/A”，数值 0 必须保留；无有效数据时说明缺失原因。
