---
name: analyzing-crypto-fed-policy
description: 适用于美联储加息、降息、政策利率变化及鹰鸽倾向分析。
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
