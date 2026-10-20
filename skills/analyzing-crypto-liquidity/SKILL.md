---
name: analyzing-crypto-liquidity
description: 适用于美元及加密市场流动性分析，包括 DXY、M2、WALCL 和 USDT+USDC 稳定币市值。
tools:
  - analyze_liquidity
scoring:
  name: 美元流动性与加密内部流动性
  metrics:
  - id: DXY_Trend
    label: 美元指数
    weight: 0.35
    kind: numeric
  - id: Stable_MCap
    label: 稳定币总市值
    weight: 0.3
    kind: numeric
  - id: M2_YoY
    label: M2 同比变化
    weight: 0.2
    kind: numeric
  - id: Fed_Balance
    label: 美联储资产负债表
    weight: 0.15
    kind: numeric
---

# 美元流动性与加密内部流动性

## 解读重点

按检索到的材料结合 DXY、稳定币市值、M2 和 WALCL 变化给出解读。
