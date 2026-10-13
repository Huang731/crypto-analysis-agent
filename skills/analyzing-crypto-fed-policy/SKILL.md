---
name: analyzing-crypto-fed-policy
description: >-
  按近三个月政策利率净变化判断偏鹰、偏鸽或中性，适用于加息、降息和利率政策分析；不分析 M2、WALCL 或 DXY。
  一次调用完成政策利率取数与 Fed_Rate_Change_3M 计算。
tools:
  - analyze_fed_policy
scoring:
  name: 美联储政策与利率倾向
  metrics:
  - id: Fed_Rate_Change_3M
    label: 三个月政策利率净变化
    weight: 1.0
    kind: numeric
---

# 美联储政策与利率倾向

## 解读重点

按检索到的材料结合近三个月政策利率净变化给出解读。
