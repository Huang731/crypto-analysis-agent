---
name: analyzing-crypto-derivatives
description: 适用于合约资金面、持仓量（OI）、资金费率、杠杆和多空拥挤分析。
tools:
  - analyze_derivatives
scoring:
  name: 衍生品数据
  metrics:
  - id: OI
    label: 未平仓量
    weight: 0.6
    kind: numeric
  - id: Funding
    label: 资金费率
    weight: 0.4
    kind: numeric
---

# 衍生品数据

## 解读重点

按检索到的材料结合价格、OI 和资金费率变化给出解读。
