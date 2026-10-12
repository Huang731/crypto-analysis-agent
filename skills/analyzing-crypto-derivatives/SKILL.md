---
name: analyzing-crypto-derivatives
description: >-
  评估 OI 与资金费率，适用于杠杆、持仓量和多空拥挤分析；不判断价格趋势。
  一次取数并计算 OI、Funding，共用同一份原始证据。
tools:
  - analyze_derivatives
scoring:
  name: 衍生品数据
  metrics:
  - id: OI
    label: OI
    weight: 0.6
    kind: numeric
  - id: Funding
    label: Funding
    weight: 0.4
    kind: numeric
---

# 衍生品数据

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### 持仓量 OI

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 交易对 | `raw_values.symbol` |
| 最新持仓量 | `raw_values.latest_oi` |
| 基准持仓量 | `raw_values.baseline_oi` |
| 最新价格 | `raw_values.latest_price` |
| 基准价格 | `raw_values.baseline_price` |
| 持仓量变化（%） | `calculated_values.oi_change_pct` |
| 价格变化（%） | `calculated_values.price_change_pct` |
| 平均资金费率（%） | `calculated_values.funding_average_pct` |
| 资金费率分数 | `calculated_values.funding_score` |
| 市场状态 | `calculated_values.market_state` |
| 持仓量二十四小时变化（%） | `calculated_values.oi_24h_change_pct` |
| 价格二十四小时变化（%） | `calculated_values.price_24h_change_pct` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 资金费率 Funding

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 交易对 | `raw_values.symbol` |
| 平均资金费率（%） | `calculated_values.funding_average_pct` |
| 资金费率观察窗口价格变化（%） | `calculated_values.funding_price_change_pct` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径|
| 当前解读 | 按检索到的材料结合价格、OI 和资金费率变化给出解读 |
