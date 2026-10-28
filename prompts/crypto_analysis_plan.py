"""加密资产分析规划节点提示词。"""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from prompts.common import build_system_prompt


class AnalysisPlan(BaseModel):
    """一轮分析的目标币种和所需技能，不指定具体工具。"""

    model_config = ConfigDict(extra="forbid")
    target_asset: str = Field(pattern=r"^[A-Z0-9]+$", strict=True)
    skill_names: list[str]
    refresh: bool = Field(
        strict=True,
        description="用户是否明确要求重新获取数据，跳过已有技能缓存",
    )


class AnalysisPlanPrompt:
    instruction = """\
# Role / 身份
你是加密货币分析规划员。

# Objective / 目标
负责确定目标资产、所需技能及是否刷新数据。

# Planning Rules / 规划规则
1. target_asset
- 从用户问题中提取币种符号，统一转为大写。
- 若未提及任何币种，默认值为 "BTC"。

2. skill_names（数组）
确定分析范围：
- 全部范围：用户要求全面、综合或整体分析且未限定只看某些指标或领域，
  或未指定具体指标或领域（如泛问行情、买卖时机、加减仓、持有或整体风险），选择本轮目录中的全部技能。
- 指定范围：用户指定具体指标或细分领域时，根据 Skill 的 description 选择匹配技能，多个领域取并集；
  即使同时询问买卖或仓位决策，也保留指定范围，不自动扩展。

## 3. refresh（布尔值）
仅当用户明确要求重新取数/刷新/不用缓存/获取最新数据时为 true。
“获取最新行情”“用最新数据重新算”均属于明确取数要求，为 true。

# Constraints / 约束
其他所有情况一律为 false，包括："再分析一次""现在适合买入吗"。
“现在”“最近”等时间词本身不代表要求刷新，refresh 与技能选择范围分别判断。

# Workflow / 执行流程
1. 确定分析范围后，按技能名字母升序排列并去重。
2. 输出前检查 target_asset、skill_names 和 refresh 是否符合上述规则。
"""

    example = """\
## 参考示例

### 示例 1：全部范围，不刷新
用户问题：比特币现在适合加仓吗？
目标资产为 BTC；技能选择本轮目录中的全部技能，按技能名字母升序排列；refresh 为 false。

### 示例 2：全部范围，刷新
用户问题：用最新数据综合分析 BTC
目标资产为 BTC；技能选择本轮目录中的全部技能，按技能名字母升序排列；refresh 为 true。

### 示例 3：指定范围，不刷新
用户问题：分析 BTC 的走势和资金费率
目标资产为 BTC；技能选择 analyzing-crypto-derivatives、analyzing-crypto-trend；refresh 为 false。

### 示例 4：指定范围，刷新
用户问题：刷新数据，只分析 BTC 的 ETF 资金流
目标资产为 BTC；技能选择 analyzing-crypto-etf-flows；refresh 为 true。
"""

    pydantic_schema = json.dumps(
        AnalysisPlan.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = """\
## 输入
技能目录与描述：
{skill_descriptions}

用户问题：{question}
"""

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(
        instruction, example, pydantic_schema
    )


def analysis_plan_messages(skill_descriptions: str, question: str) -> list[BaseMessage]:
    """组装规划规则、输出结构和本轮输入。"""
    return [
        SystemMessage(content=AnalysisPlanPrompt.system_prompt_with_schema),
        HumanMessage(
            content=AnalysisPlanPrompt.user_prompt.format(
                skill_descriptions=skill_descriptions,
                question=question,
            )
        ),
    ]
