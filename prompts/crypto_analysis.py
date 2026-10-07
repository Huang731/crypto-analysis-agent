"""加密货币分析规划、技能执行和报告撰写提示词。"""

import yaml


def metric_execution_prompt(skill_config: dict) -> str:
    """计算阶段只提供 Skill 的 YAML 配置。"""
    config_yaml = yaml.safe_dump(skill_config, allow_unicode=True, sort_keys=False)
    return f"按以下 Skill 配置完成所有指标的取数和计算：\n{config_yaml}"


def metric_agent_system_prompt() -> str:
    """技能工具 Agent 的系统提示。"""
    return (
        "你是加密货币技能研究员。先调用取数工具，再用返回的 evidence_id 调用计算工具，"
        "不要编造证据 ID。只完成本技能指标；取不到就停止，不要撰写投资报告。"
    )


def analysis_plan_prompt(
    schema_json: str, skill_descriptions: str, question: str
) -> str:
    """根据用户问题和技能目录选择目标币种与所需 Skill。"""
    return (
        "你是加密货币分析规划员。识别单个目标币种，未写明时用 BTC，"
        "多个或无法识别时不要猜测。只返回 target_asset 和 skill_names。"
        "skill_names 必须是下方目录中的完整名称，按目录顺序、不重复。"
        "先判断用户有没有指定分析范围。"
        "行情、走势、买卖时机、整体风险，或没有指定分析范围、指标时，"
        "skill_names 必须包含目录中的每一项。"
        "这时描述里的「用户问…时选用」和「不分析…」都不能用来删技能；"
        "「不分析」只说明该技能自身不覆盖的内容。"
        "只有用户点明范围或明确排除时，才按描述筛选："
        "对得上的留下，明确排除的去掉，都不适用则返回空列表。"
        "例如‘BTC最近怎么样’选全部；‘只分析ETF资金流’只选 analyzing-crypto-etf-flows；"
        "‘综合分析但不看情绪’去掉 analyzing-crypto-sentiment。"
        "仅返回符合 JSON Schema 的 JSON 对象。"
        f"\nJSON Schema：{schema_json}"
        f"\n技能描述：{skill_descriptions}"
        f"\n任务描述：{question}"
    )


def thought_retrieval_prompt(
    target_asset: str, skill_name: str, metric_labels: list[str]
) -> str:
    """按固定模板检索技能理论材料，不额外调用模型生成查询。"""
    metric_names = "、".join(metric_labels)
    return (
        f"如何解释{skill_name}对{target_asset}的影响？"
        f"相关指标：{metric_names}。"
        "请查找相关理论原则、作用机制、信号分歧和适用边界。"
    )


def analysis_report_prompt(
    target_asset: str,
    original_question: str,
    facts_json: str,
    output_templates: str,
) -> str:
    """整合已计算的指标与检索材料，回答用户问题。"""
    return (
        f"用中文分析 {target_asset}。"
        f"\n用户问题：{original_question}"
        "\n先简答，再按所选技能模板完整填写 Markdown 表格，最后总结。"
        "\n总结以回答用户问题为目标，以技能结果和数据为依据，不逐项复述技能结果。"
        "\n遵循模板规则，以 indicators 为事实依据，不生成综合分。"
        "\n仅展示用户可读内容，隐藏字段路径、模板说明、内部键值及旧维度编号。"
        f"\n技能分析输出模板：\n{output_templates}"
        f"\n技能结果：{facts_json}"
    )
