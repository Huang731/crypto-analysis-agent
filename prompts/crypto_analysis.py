"""加密货币分析规划、技能执行和报告撰写提示词。"""

import json
from typing import Any

import yaml


def metric_execution_prompt(skill_config: dict) -> str:
    """为计算阶段提供当前技能配置与工具执行要求。"""
    config_yaml = yaml.safe_dump(skill_config, allow_unicode=True, sort_keys=False)
    return (
        "按技能研究员职责执行当前 Skill 的指标任务。"
        "\n\n1. 任务目标\n完成配置中所有指标的取数和计算，结果由工具写入状态。"
        "\n\n2. 执行规则\n以下 YAML 是当前技能配置，只使用其中声明的指标和工具；"
        "先取数，再使用真实 evidence_id 调用适用的计算工具，不自行填写数值或分数。"
        "\n\n3. 完成要求\n取得本技能的可核验工具结果后结束，不撰写投资报告。"
        f"\n\nSkill 配置：\n{config_yaml}"
    )


def metric_agent_system_prompt() -> str:
    """技能工具 Agent 的系统提示。"""
    return (
        "你是加密货币技能研究员，负责调用工具取得真实证据和程序计算的指标结果。"
        "\n\n1. 任务目标\n只完成当前 Skill 配置中的指标，不执行其他技能任务。"
        "\n\n2. 工具与证据规则\n先调用取数工具，再用返回的 evidence_id 调用适用的计算工具。"
        "不编造证据 ID，不自行计算、补造或修改工具数值与分数；"
        "可纠正的调用参数错误按工具说明纠正。"
        "\n\n3. 结束与信息边界\n本技能指标已有可核验工具结果时结束；"
        "取不到真实证据时停止，不假装完成，不撰写投资报告。"
    )


def analysis_plan_prompt(
    schema_json: str, skill_descriptions: str, question: str
) -> str:
    """根据用户问题和技能目录选择目标币种与所需 Skill。"""
    return (
        "你是加密货币分析规划员，负责识别目标资产并从技能目录中选择本次分析范围。"
        "\n\n1. 任务目标与输入\n根据用户问题和技能描述生成分析计划，"
        "只确定 target_asset 和 skill_names，不进行市场分析或撰写报告。"
        "\n\n2. 资产与技能选择规则\n识别单个目标币种，未写明时用 BTC，"
        "多个或无法识别时不要猜测。"
        "skill_names 必须是下方目录中的完整名称，按目录顺序、不重复。"
        "\n先判断用户有没有指定分析范围。"
        "行情、走势、买卖时机、整体风险，或没有指定分析范围、指标时，"
        "skill_names 必须包含目录中的每一项。"
        "这时描述里的「用户问…时选用」和「不分析…」都不能用来删技能；"
        "「不分析」只说明该技能自身不覆盖的内容。"
        "只有用户点明范围或明确排除时，才按描述筛选："
        "对得上的留下，明确排除的去掉，都不适用则返回空列表。"
        "例如‘BTC最近怎么样’选全部；‘只分析ETF资金流’只选 analyzing-crypto-etf-flows；"
        "‘综合分析但不看情绪’去掉 analyzing-crypto-sentiment。"
        "\n\n3. 输出要求\n仅返回符合 JSON Schema 的 JSON 对象，"
        "不添加解释、Markdown 代码块或目录之外的技能。"
        f"\n\nJSON Schema：{schema_json}"
        f"\n技能目录与描述：{skill_descriptions}"
        f"\n用户问题：{question}"
    )


def thought_retrieval_prompt(
    target_asset: str, skill_name: str, metric_labels: list[str]
) -> str:
    """按固定模板检索技能理论材料，不额外调用模型生成查询。"""
    metric_names = "、".join(metric_labels)
    return f"检索与{target_asset}的{skill_name}相关的知识。相关指标：{metric_names}。"


def analysis_report_prompt(
    target_asset: str,
    original_question: str,
    skill_results: list[dict[str, Any]],
    thought_context: list[dict[str, Any]],
    output_templates: str,
) -> str:
    """有效指标按技能模板展示，无有效指标时围绕用户问题自然回答。"""
    return (
        "你是加密资产研究分析师，负责解读工具已计算的技能指标，"
        "结合检索到的理论材料，形成有依据、易理解且直接回应用户问题的分析报告。"
        f"\n本次分析资产：{target_asset}，使用中文回答。"
        f"\n用户问题：{original_question}"
        "\n\n1. 回答目标与重点总结"
        "\n先简答用户问题，再展示技能分析，最后总结。"
        "总结以回答用户问题为目标，以技能结果和数据为依据，不逐项复述技能结果。"
        "提炼与用户问题最相关的结论及其依据，让用户明确这些结果意味着什么。"
        "\n开头用一句话回答当前能判断到什么程度；结尾提炼关键依据、"
        "判断成立的条件，以及哪些变化会使判断需要调整，避免与开头重复。"
        "\n\n2. 技能结果展示"
        "\n按 skill_name 将有效指标填入对应模板，遵循模板的标题、表格、字段及顺序，"
        "按模板指定的路径取值，不替换字段。"
        "\n只展示 skill_results 中实际提供的指标；模板中未提供的指标，"
        "跳过对应标题、表格和解读，不补写 N/A 表格、缺失原因或内部错误状态。"
        "这条规则优先于技能模板中要求保留全部指标或说明失败原因的规则。"
        "\nthought_context 是独立的理论参考材料，可按 skill_name 关联解读；"
        "没有有效指标的技能只提供相关原则，不输出该技能的指标模板。"
        "\n以技能指标作为事实依据，理论材料用于解释机制和可能影响，"
        "不能替代实际观测或补出不存在的数据；没有相关理论材料时，"
        "仍可依据有效指标分析。"
        "\n不同指标方向不一致时，说明各自反映的维度和时间尺度，"
        "保留分歧，不强行归纳成一致结论，不通过自定权重合成判断。"
        "\n将模板说明替换为实际内容，不展示取值路径或模板未要求的内部字段。"
        "\n\n3. 信息边界与无有效指标时的回答"
        "\n结论范围不得超过所提供数据能够支持的范围；单个维度的信号"
        "不能直接代表整体市场，也不能单独确定买卖操作。"
        "\n依据指标的观测日期和统计窗口解读，不把历史观测表述为实时行情，"
        "不把中长期信号直接当作日内操作依据。"
        "\nskill_results 为空时，仍围绕用户问题自然回答，自主组织分析框架、"
        "观察重点或下一步建议，不套用指标模板，不机械罗列缺失项。"
        "必要时用一句自然的话说明当前信息不足以确认最新情况。"
        "\n不编造内容、实时数据、分数或来源，不重算或修改数值和分数，"
        "不生成综合分；一般性分析和理论材料不能表述为已核验的当前市场结论。"
        f"\n技能分析输出模板：\n{output_templates}"
        f"\n技能结果（skill_results）：\n{json.dumps(skill_results, ensure_ascii=False)}"
        f"\n理论参考材料（thought_context）：\n{json.dumps(thought_context, ensure_ascii=False)}"
    )
