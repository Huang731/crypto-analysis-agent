---
name: analyzing-crypto-sentiment
description: >-
  按恐惧贪婪指数评估全市场情绪，适用于恐慌、贪婪和极端情绪分析；不代表单个币种的价格或资金。
  一次调用完成情绪指数取数与 Fear_Greed 计算。
tools:
  - analyze_sentiment
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
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径|
| 当前解读 | 按检索到的材料结合恐惧贪婪指数、情绪分类和信号参考给出解读 |
