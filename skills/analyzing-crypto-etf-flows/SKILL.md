---
name: analyzing-crypto-etf-flows
description: >-
  评估现货 ETF 近 5 日与 20 日净流量，适用于 ETF 资金及机构申购赎回分析；不代表全部市场需求。
  一次调用完成 ETF 取数与 ETF_Flows 计算。
tools:
  - analyze_etf_flows
scoring:
  name: ETF资金流
  metrics:
  - id: ETF_Flows
    label: ETF_Flows
    weight: 1.0
    kind: numeric
---

# ETF资金流

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### ETF 资金流

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 最新日净流量（亿美元） | `raw_values.latest_flow_usd_100m` |
| 前一个五日窗口净流量（亿美元） | `raw_values.previous_five_day_flow_usd_100m` |
| 五日净流量（亿美元） | `calculated_values.five_day_flow_usd_100m` |
| 二十日净流量（亿美元） | `calculated_values.twenty_day_flow_usd_100m` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径|
| 当前解读 | 按检索到的材料结合最新日、近五日和近二十日 ETF 净流量给出解读 |
