---
name: analyzing-crypto-liquidity
description: 评估美元与加密市场的流动性环境，覆盖美元指数 DXY、USDT 与 USDC 稳定币总市值、美国 M2 同比变化和美联储资产负债表 WALCL。用户问流动性、美元强弱、稳定币供给或宽货币时选用；不分析政策利率升降。
tools:
  - get_eodhd_dxy
  - calculate_dxy_indicator
  - get_cmc_stablecoin_market_cap
  - calculate_stablecoin_indicator
  - get_fred_observations
  - calculate_m2_indicator
  - calculate_fed_balance_indicator
scoring:
  dimension_id: '01'
  name: 美元流动性与加密内部流动性
  metrics:
  - id: DXY_Trend
    label: 美元指数 DXY
    weight: 0.35
    kind: numeric
  - id: Stable_MCap
    label: 稳定币总市值 Stable_MCap
    weight: 0.3
    kind: numeric
  - id: M2_YoY
    label: M2 同比变化 M2_YoY
    weight: 0.2
    kind: numeric
  - id: Fed_Balance
    label: 美联储资产负债表 WALCL
    weight: 0.15
    kind: numeric
  overall_weight: 0.2
---

# 美元流动性与加密内部流动性

get_eodhd_dxy → calculate_dxy_indicator。

get_cmc_stablecoin_market_cap → calculate_stablecoin_indicator。

get_fred_observations(series_id="M2SL") → calculate_m2_indicator。

get_fred_observations(series_id="WALCL") → calculate_fed_balance_indicator。

- DXY按相对MA50的偏离反向评分；MA200和均线方向用于补充观察。
- 稳定币为USDT+USDC，按MA30偏离与七日变化等权评分；两者反向时说明分歧，三十日变化作补充。
- M2评分看最新月与上月同比增速之差（百分点），不是同比本身。
- WALCL看最新值相对连续四周均值的偏离。
- 宏观与稳定币指标不等于目标资产的实际买入资金。
