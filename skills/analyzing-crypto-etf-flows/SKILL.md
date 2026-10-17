---
name: analyzing-crypto-etf-flows
description: 适用于现货 ETF 资金面、净流入净流出及机构申购赎回分析。
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
