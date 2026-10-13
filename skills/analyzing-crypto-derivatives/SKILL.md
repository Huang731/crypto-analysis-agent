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

## 解读重点

按检索到的材料结合价格、OI 和资金费率变化给出解读。
