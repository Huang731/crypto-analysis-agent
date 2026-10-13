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

## 解读重点

按检索到的材料结合 MA50、MA200、ADX 和 DI 给出解读。
