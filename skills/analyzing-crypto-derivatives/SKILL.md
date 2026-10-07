---
name: analyzing-crypto-derivatives
description: 评估目标资产合约市场的杠杆与拥挤程度，覆盖持仓量 OI 和资金费率 Funding。用户问杠杆、多空拥挤、资金费率或持仓量时选用；不判断价格趋势本身。
tools:
  - get_binance_derivatives_data
  - calculate_oi_indicator
  - calculate_funding_indicator
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

get_binance_derivatives_data → calculate_oi_indicator、calculate_funding_indicator。

- 两项计算复用同一份真实证据；核对目标合约，勿混用不同资产的数据。
- OI结合七日价格变化、OI变化及Funding解释；OI增长本身不能确定多空方向。
- Funding采用完整七日结算记录的八小时等效口径，单位和方向以计算工具为准。
- 解释计算器返回的杠杆压力、去杠杆或扩张信号，不把高杠杆等同持续上涨。
