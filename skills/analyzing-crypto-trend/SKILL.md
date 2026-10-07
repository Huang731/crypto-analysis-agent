---
name: analyzing-crypto-trend
description: >-
  用 MA50、MA200、ADX 和 DI 评估目标资产的趋势、强度与确认情况，适用于走势及均线分析。
  get_binance_trend_data → calculate_trend_indicator，使用返回的 evidence_id。
tools:
  - get_binance_trend_data
  - calculate_trend_indicator
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
| K 线数量 | `raw_values.kline_count` |
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
| 准入结果 | `calculated_values.admission` |
| 时机结果 | `calculated_values.timing` |
| 确认结果 | `calculated_values.confirmation` |
| 计算中间值 | `calculated_values.intermediate_values` |
| 计算方法 | `calculated_values.calculation_method` |
| 指标状态 | `status` |
| 状态原因 | `reason` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源工具 | `source.action` |
| 来源链接 | `source.url` |

分析仅依据工具结果，不重算或修改数值和分数。缺失值或不适用字段写“N/A”，数值 0 必须保留；无有效数据时说明缺失原因。
