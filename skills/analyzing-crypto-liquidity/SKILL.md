---
name: analyzing-crypto-liquidity
description: >-
  评估 DXY、USDT+USDC 市值、M2 和 WALCL，适用于美元及加密市场流动性分析；不分析政策利率。
  取数后使用对应 evidence_id 计算：
  get_eodhd_dxy → calculate_dxy_indicator；
  get_cmc_stablecoin_market_cap → calculate_stablecoin_indicator；
  get_fred_observations(series_id="M2SL") → calculate_m2_indicator；
  get_fred_observations(series_id="WALCL") → calculate_fed_balance_indicator。
tools:
  - get_eodhd_dxy
  - calculate_dxy_indicator
  - get_cmc_stablecoin_market_cap
  - calculate_stablecoin_indicator
  - get_fred_observations
  - calculate_m2_indicator
  - calculate_fed_balance_indicator
scoring:
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
---

# 美元流动性与加密内部流动性

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### 美元指数 DXY

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| DXY | `raw_values.dxy` |
| MA50 | `raw_values.ma50` |
| MA200 | `raw_values.ma200` |
| MA50 偏离比例（%） | `calculated_values.ma50_deviation_pct` |
| MA50 方向 | `calculated_values.ma50_trend` |
| MA200 方向 | `calculated_values.ma200_trend` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 稳定币市值

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| USDT 市值（亿美元） | `raw_values.usdt_market_cap` |
| USDC 市值（亿美元） | `raw_values.usdc_market_cap` |
| 稳定币总市值（亿美元） | `raw_values.total_market_cap` |
| 稳定币市值 MA30（亿美元） | `raw_values.ma30_market_cap` |
| 七日前市值（亿美元） | `raw_values.seven_day_market_cap` |
| 三十日前市值（亿美元） | `raw_values.thirty_day_market_cap` |
| MA30 偏离比例（%） | `calculated_values.ma30_change_pct` |
| 七日变化（%） | `calculated_values.seven_day_change_pct` |
| 三十日变化（%） | `calculated_values.thirty_day_change_pct` |
| 信号分歧 | `calculated_values.conflict_flag` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### M2 同比变化

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 最新值 | `raw_values.latest_value` |
| 上月值 | `raw_values.previous_value` |
| 最新月去年同期值 | `raw_values.latest_year_ago_value` |
| 上月去年同期值 | `raw_values.previous_year_ago_value` |
| 比较月份 | `raw_values.comparison_months` |
| 最新月同比（%） | `calculated_values.latest_yoy_pct` |
| 上月同比（%） | `calculated_values.previous_yoy_pct` |
| 同比增速变化（百分点） | `calculated_values.yoy_change_pp` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 美联储资产负债表 WALCL

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 最新值 | `raw_values.latest_value` |
| 连续四周均值 | `raw_values.ma4_value` |
| 比较日期 | `raw_values.comparison_dates` |
| 各周数值 | `raw_values.weekly_values` |
| 四周均值偏离比例（%） | `calculated_values.ma4_change_pct` |
| 指标状态 | `status` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 来源章节 | 使用当前解读实际引用的思想材料的 `metadata.section`，展示章节路径|
| 当前解读 | 按检索到的材料结合 DXY、稳定币市值、M2 和 WALCL 变化给出解读 |
