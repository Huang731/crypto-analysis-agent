---
name: analyzing-crypto-trend
description: >-
  用 MA50、MA200、ADX 和 DI 评估目标资产的趋势、强度与确认情况，适用于走势及均线分析。
  一次调用完成日线取数与 Trend 计算。
tools:
  - analyze_trend
scoring:
  name: 趋势与均线
  metrics:
  - id: Trend
    label: Trend
    weight: 1.0
    kind: numeric
---

# 趋势与均线

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### 趋势与均线

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 交易对 | `raw_values.symbol` |
| 首根 K 线开盘时间 | `raw_values.first_open_time` |
| 最后一根 K 线收盘时间 | `raw_values.last_close_time` |
| 最新收盘价 | `raw_values.close` |
| MA50 | `calculated_values.ma50` |
| MA200 | `calculated_values.ma200` |
| 二十日前 MA50 | `calculated_values.ma50_20_days_ago` |
| 二十日前 MA200 | `calculated_values.ma200_20_days_ago` |
| ADX14 | `calculated_values.adx14` |
| +DI | `calculated_values.plus_di` |
| -DI | `calculated_values.minus_di` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径|
| 当前解读 | 按检索到的材料结合 MA50、MA200、ADX 和 DI 给出解读 |
