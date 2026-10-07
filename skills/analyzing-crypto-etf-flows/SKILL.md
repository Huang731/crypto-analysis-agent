---
name: analyzing-crypto-etf-flows
description: >-
  评估现货 ETF 近 5 日与 20 日净流量，适用于 ETF 资金及机构申购赎回分析；不代表全部市场需求。
  get_etf_flows → calculate_etf_indicator，使用返回的 evidence_id。
tools:
  - get_etf_flows
  - calculate_etf_indicator
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
| 交易日期 | `raw_values.trading_dates` |
| 每日净流量（百万美元） | `raw_values.daily_flows_usd_m` |
| 最新日净流量（百万美元） | `raw_values.latest_flow_usd_m` |
| 前一个五日窗口净流量（百万美元） | `raw_values.previous_five_day_flow_usd_m` |
| 五日净流量（百万美元） | `calculated_values.five_day_flow_usd_m` |
| 二十日净流量（百万美元） | `calculated_values.twenty_day_flow_usd_m` |
| 五日分数 | `calculated_values.five_day_score` |
| 二十日分数 | `calculated_values.twenty_day_score` |
| 信号分歧 | `calculated_values.conflict_flag` |
| 指标状态 | `status` |
| 状态原因 | `reason` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源工具 | `source.action` |
| 来源链接 | `source.url` |

分析仅依据工具结果，不重算或修改数值和分数。缺失值或不适用字段写“N/A”，数值 0 必须保留；无有效数据时说明缺失原因。
