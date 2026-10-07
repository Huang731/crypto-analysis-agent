---
name: analyzing-crypto-sentiment
description: 评估加密市场整体情绪，使用恐惧贪婪指数及其固定分档。用户问恐慌、贪婪、市场情绪或极端情绪时选用；这是全市场指标，不是单个币种的价格或资金数据。
scoring:
  dimension_id: '06'
  name: 市场情绪
  metrics:
  - id: Fear_Greed
    label: Fear_Greed
    weight: 1.0
    kind: numeric
  overall_weight: 0.15
---

# 市场情绪

get_cmc_fear_greed → calculate_fear_greed_indicator。

- 使用CMC市场恐惧贪婪指数，固定分档与信号以工具输出为准，不自行改成0至10线性分数。
- 指数是市场整体情绪，不是目标币种独有指标。
- 极端情绪只提供观察视角，不作为独立买卖依据；观测时间按当前计算器口径展示。
