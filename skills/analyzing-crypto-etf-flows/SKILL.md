---
name: analyzing-crypto-etf-flows
description: 评估目标资产现货 ETF 的资金流入和流出，覆盖近 5 日与近 20 日净流入。用户问 ETF、机构申购赎回或现货 ETF 资金时选用；不能代表全部市场需求。
scoring:
  dimension_id: '03'
  name: ETF资金流
  metrics:
  - id: ETF_Flows
    label: ETF_Flows
    weight: 1.0
    kind: numeric
  overall_weight: 0.15
---

# ETF资金流

get_etf_flows → calculate_etf_indicator。

- 获取目标资产的ETF资金流；五日与二十日金额分档分别占40%、60%，金额统一为百万美元。
- 两个窗口方向相反时分别解释短期变化与较长窗口背景。
- ETF资金流不能代表全部市场需求；资产无适用ETF或数据缺失时保留工具状态。
