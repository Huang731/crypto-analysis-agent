"""加密货币分析规划、技能执行和报告撰写提示词。"""

import json
from typing import Any

import yaml

from app.crypto_analysis.crypto_analysis_state import SkillResult


def metric_execution_prompt(skill_config: dict) -> str:
    """提供本轮指标任务与技能配置，通用执行规则由系统提示规定。"""
    config_yaml = yaml.safe_dump(skill_config, allow_unicode=True, sort_keys=False)
    return f"完成以下 Skill 配置中的所有指标。\n\nSkill 配置：\n{config_yaml}"


def metric_agent_system_prompt() -> str:
    """技能工具 Agent 的系统提示。"""
    return (
        "你是加密货币技能研究员，使用工具完成当前 Skill 的取数和指标计算。"
        "\n1. 只执行配置中声明的指标和工具。先取数，再使用返回的真实 evidence_id "
        "调用适用的计算工具，结果由工具写入状态。"
        "\n2. 不编造证据 ID，不自行计算、补造或修改工具数值与分数；"
        "调用参数错误按工具说明纠正。"
        "\n3. 全部指标取得可核验工具结果后结束；无法取得真实证据时停止，"
        "不假装完成，不撰写投资报告。"
    )


def analysis_plan_prompt(skill_descriptions: str, question: str) -> str:
    """根据用户问题和技能目录选择目标币种与所需 Skill。"""
    return (
        "你是加密货币分析规划员，只提交 target_asset、skill_names、refresh。"
        "\n1. target_asset：币种符号转大写，未指定时用 BTC。"
        "\n2. skill_names：按以下优先级确定集合："
        "\n- 泛问整体行情、走势、买卖时机、整体风险：选目录全部技能。"
        "‘分析BTC的走势’选全部；‘走势’本身不是限定趋势技能的条件。"
        "\n- 明确指定具体指标或细分领域：按目录描述选择对应技能，多领域取并集。"
        "‘只看均线、ADX’选趋势技能。"
        "\n- 未指定范围：选全部。不得自行按相关性或币种删减。"
        "\n- 有排除条件：从上述集合删除对应技能。"
        "\n3. 确定集合后，按技能名升序排列（与目录顺序一致），"
        "禁止按问题提及顺序排列。使用完整原名、不重复。"
        "\n4. refresh：只有用户明确要求重新取数、刷新数据、不用缓存"
        "或获取最新数据才为 true，其他一律 false。"
        "‘获取最新资金费率数据’为 true；"
        "‘现在适合买入吗’‘再分析一次’‘重新分析’为 false。"
        "按用户的取数要求判断，不因行情时效性或分析需求推断刷新。"
        f"\n\n技能目录与描述：{skill_descriptions}"
        f"\n用户问题：{question}"
    )


def thought_retrieval_prompt(
    target_asset: str, skill_name: str, metric_labels: list[str]
) -> str:
    """按固定模板检索技能理论材料，不额外调用模型生成查询。"""
    metric_names = "、".join(metric_labels)
    return f"检索与{target_asset}的{skill_name}相关的知识。相关指标：{metric_names}。"


def skill_output_prompt(
    analysis_subject: str,
    skill_name: str,
    indicators: list[dict[str, Any]],
    thought_context: dict[str, Any],
    output_template: str,
) -> str:
    """生成可跨轮复用的技能表格与思想体系解读。"""
    return (
        "根据指标按当前技能模板生成中文指标表格和思想体系解读。"
        f"\n分析对象：{analysis_subject}；技能：{skill_name}。"
        f"\n输出模板：\n{output_template}"
        f"\n成功指标：\n{json.dumps(indicators, ensure_ascii=False)}"
        f"\n思想材料：\n{json.dumps(thought_context, ensure_ascii=False)}"
    )


def analysis_report_prompt(
    target_asset: str,
    original_question: str,
    skill_results: list[SkillResult],
    history_summary: str,
) -> str:
    """仅依据技能内容和此前结论摘要撰写本轮整体结论。"""
    return (
        "根据各技能内容、思想解读和历史摘要，用中文回答本轮问题并撰写整体结论，字数500-600字左右"
        "。直接输出报告正文，不添加 JSON 包装或工具调用参数。"
        f"\n目标资产：{target_asset}；用户问题：{original_question}"
        f"\n各技能内容：\n{json.dumps(skill_results, ensure_ascii=False)}"
        f"\n此前结论摘要：\n{history_summary}"
    )


def history_summary_prompt(
    previous_summary: str,
    report: str,
) -> str:
    """只将本轮问题与结论纳入累计摘要，不重复保存技能表格。"""
    return (
        "合并旧摘要与本轮结论，提炼【值得下次分析复用】的记忆摘要"
        "。只返回摘要正文，不添加 JSON 包装或解释。"
        f"\n旧摘要：{previous_summary}"
        f"\n本轮结论：{report}"
    )
