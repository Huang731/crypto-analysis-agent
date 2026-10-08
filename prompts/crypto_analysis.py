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
        f"检索与{target_asset}的{skill_name}相关的知识。"
        f"相关指标：{metric_names}。"
    )


def analysis_report_prompt(
    target_asset: str,
    original_question: str,
    facts_json: str,
    output_templates: str,
) -> str:
    """有效指标按技能模板展示，无有效指标时围绕用户问题自然回答。"""
    return (
        f"用中文分析 {target_asset}。"
        f"\n用户问题：{original_question}"
        "\n\n1. 回答目标与重点总结"
        "\n先简答用户问题，再展示技能分析，最后总结。"
        "总结以回答用户问题为目标，以技能结果和数据为依据，不逐项复述技能结果。"
        "提炼与用户问题最相关的结论及其依据，让用户明确这些结果意味着什么。"
        "\n\n2. 技能结果展示"
        "\n按 skill_name 将有效指标填入对应模板，遵循模板的标题、表格、字段及顺序，"
        "按模板指定的路径取值，不替换字段。"
        "\n只展示 skill_results 中实际提供的指标；模板中未提供的指标，"
        "跳过对应标题、表格和解读，不补写 N/A 表格、缺失原因或内部错误状态。"
        "这条规则优先于技能模板中要求保留全部指标或说明失败原因的规则。"
        "\nthought_context 是独立的理论参考材料，可按 skill_name 关联解读；"
        "没有有效指标的技能只提供相关原则，不输出该技能的指标模板。"
        "\n将模板说明替换为实际内容，不展示取值路径或模板未要求的内部字段。"
        "\n\n3. 信息边界与无有效指标时的回答"
        "\nskill_results 为空时，仍围绕用户问题自然回答，自主组织分析框架、"
        "观察重点或下一步建议，不套用指标模板，不机械罗列缺失项。"
        "必要时用一句自然的话说明当前信息不足以确认最新情况。"
        "\n不编造内容、实时数据、分数或来源，不重算或修改数值和分数，"
        "不生成综合分；一般性分析和理论材料不能表述为已核验的当前市场结论。"
        f"\n技能分析输出模板：\n{output_templates}"
        f"\n技能结果：{facts_json}"
    )
