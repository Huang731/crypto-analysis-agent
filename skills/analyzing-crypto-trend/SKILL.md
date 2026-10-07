---
name: analyzing-crypto-trend
description: 评估目标资产的价格趋势，覆盖 MA50、MA200、趋势强度 ADX 和方向指标 DI。用户问走势、均线、趋势强弱或涨跌是否确认时选用。
tools:
  - get_binance_trend_data
  - calculate_trend_indicator
scoring:
  name: 趋势与均线
  metrics:
  - id: Trend
    label: Trend
    weight: 1.0
    kind: numeric
---

# 趋势与均线

get_binance_trend_data → calculate_trend_indicator。

- 使用目标资产完整UTC日线，由计算器计算均线、Wilder ADX和DI。
- MA200与二十日前MA200用于趋势准入；ADX和DI用于时机；MA50与MA200关系用于确认。
- 区分趋势背景、强度和方向；强趋势不自动等于上涨，不单凭均线突破断言趋势确认。
