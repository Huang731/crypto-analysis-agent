---
name: analyzing-crypto-fed-policy
description: 评估美联储已经发生的政策利率变化，用联邦基金目标区间近三个月净变动判断偏鹰、偏鸽或中性。用户问加息、降息、利率路径或货币政策立场时选用；不分析资产负债表、M2 或美元指数。
tools:
  - get_fred_target_range
  - calculate_fed_rate_change_indicator
scoring:
  name: 美联储政策与利率倾向
  metrics:
  - id: Fed_Rate_Change_3M
    label: 三个月政策利率净变化
    weight: 1.0
    kind: numeric
---

# 美联储政策与利率倾向

get_fred_target_range → calculate_fed_rate_change_indicator。

- 使用FRED上下限共同有效的日期，以目标区间中点计算三个月净变化（基点）。
- 分数来自计算工具；分数大于6解释为偏鸽，小于4为偏鹰，其余为中性或相互抵消。
- 三个月净变化为零不代表期间没有调整，也不代表未来政策承诺。
- 本技能评估已发生的利率变化，不将新闻预期代替政策利率事实。
