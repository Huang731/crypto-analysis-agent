---
name: analyzing-crypto-fed-policy
description: >-
  按近三个月政策利率净变化判断偏鹰、偏鸽或中性，适用于加息、降息和利率政策分析；不分析 M2、WALCL 或 DXY。
  get_fred_target_range → calculate_fed_rate_change_indicator，使用返回的 evidence_id。
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

## 分析输出

按以下“字段／内容”表格逐项输出，保留所有字段并将说明文字替换为实际结果。每个指标表的字段路径对应该指标的标准化工具结果。列表和对象完整展示，多条内容用 `<br>` 分隔。

### 政策利率

| 字段 | 内容 |
| --- | --- |
| 指标标识 | `indicator_id` |
| 当前利率观测日期 | `raw_values.current_date` |
| 基准利率观测日期 | `raw_values.baseline_date` |
| 三个月比较日期 | `raw_values.comparison_date` |
| 当前利率下限（%） | `raw_values.current_lower_pct` |
| 当前利率上限（%） | `raw_values.current_upper_pct` |
| 基准利率下限（%） | `raw_values.baseline_lower_pct` |
| 基准利率上限（%） | `raw_values.baseline_upper_pct` |
| 当前利率中点（%） | `calculated_values.current_midpoint_pct` |
| 基准利率中点（%） | `calculated_values.baseline_midpoint_pct` |
| 利率变化（基点） | `calculated_values.change_bp` |
| 政策变化方向 | `calculated_values.policy_change` |
| 指标状态 | `status` |
| 状态原因 | `reason` |
| 指标分数 | `score` |
| 单位 | `unit` |
| 观测日期 | `observed_at` |
| 来源链接 | `source.url` |

### 思想体系解读

| 字段 | 内容 |
| --- | --- |
| 当前解读 | 结合近三个月利率净变化，按相关原则解释政策方向及对资金成本和风险偏好的影响；区分已发生变化与未来预期 |

仅使用检索到的相关原则，不补造思想体系内容。无相关材料时写“未检索到相关原则”；检索失败时写“思想材料检索失败”，当前解读仅依据指标。

分析仅依据工具结果，不重算或修改数值和分数。缺失值或不适用字段写“N/A”，数值 0 必须保留；无有效数据时说明缺失原因。
