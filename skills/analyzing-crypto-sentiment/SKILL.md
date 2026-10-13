---
name: analyzing-crypto-sentiment
description: >-
  按恐惧贪婪指数评估全市场情绪，适用于恐慌、贪婪和极端情绪分析；不代表单个币种的价格或资金。
  一次调用完成情绪指数取数与 Fear_Greed 计算。
tools:
  - analyze_sentiment
scoring:
  name: 市场情绪
  metrics:
  - id: Fear_Greed
    label: Fear_Greed
    weight: 1.0
    kind: numeric
---

# 市场情绪

## 解读重点

按检索到的材料结合恐惧贪婪指数、情绪分类和信号参考给出解读。
