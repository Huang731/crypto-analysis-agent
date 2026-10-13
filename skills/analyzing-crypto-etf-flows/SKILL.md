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

## 解读重点

按检索到的材料结合最新日、近五日和近二十日 ETF 净流量给出解读。
