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
| 信号参考 | `calculated_values.signal_reference` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径；引用多条材料时逐条列出并去重，保留章节与解读的对应关系；章节缺失、未检索到材料或检索失败时写“N/A”，不推测章节 |
| 当前解读 | 结合恐惧贪婪指数、情绪分类及信号参考，按相关原则解释当前情绪及其含义；区分事实与推断 |

仅使用检索到的相关原则，不补造思想体系内容。无相关材料时写“未检索到相关原则”；检索失败时写“思想材料检索失败”，当前解读仅依据指标。

分析仅依据工具结果，不重算或修改数值和分数。缺失值或不适用字段写“N/A”，数值 0 必须保留；无有效数据时说明缺失原因。
