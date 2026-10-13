"""加密货币分析规划、技能执行和报告撰写提示词。"""

import json
from typing import Any

from app.crypto_analysis.crypto_analysis_state import SkillResult


def metric_execution_prompt(skill_config: dict[str, Any]) -> str:
    """提供本轮指标清单，工具用法由已注册工具说明提供。"""
    indicator_list = "\n".join(
        f"- {metric['id']}" for metric in skill_config["scoring"]["metrics"]
    )
    return f"完成以下所有指标：\n{indicator_list}"


def metric_agent_system_prompt() -> str:
    """技能工具 Agent 的系统提示。"""
    return (
        "你是加密货币技能研究员，使用工具完成当前 Skill 的取数和指标计算。"
        "\n1. 只完成本轮指定的指标，按已注册工具的说明调用。"
        "组合工具会自动取数、计算并把结果写入状态，无需再单独调用取数或计算工具。"
        "\n2. 不自行计算、补造或修改工具数值与分数；"
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
    skill_instructions: str,
) -> str:
    """按通用展示规则生成完整 Markdown，Skill 仅提供职责与解读重点。"""
    return (
        "根据成功指标和思想材料，生成中文 Markdown 指标表格和思想体系解读。"
        "\n1. 每个成功指标单独输出三级标题和一张连续的“字段／内容”两列表格，"
        "每张表的表头只出现一次，不拆成原始值表和计算值表。"
        "标题使用易懂的指标中文名称，表格格式如下："
        "\n### {指标中文名称}\n\n| 字段 | 内容 |\n| --- | --- |"
        "\n| 指标标识 | {indicator_id} |"
        "\n| {raw_values 各字段的中文名称} | {对应原始值} |"
        "\n| {calculated_values 各字段的中文名称} | {对应计算值} |"
        "\n| 指标状态 | {status} |\n| 指标分数 | {score} |"
        "\n| 单位 | {unit} |\n| 观测日期 | {observed_at} |\n| 来源链接 | {url} |"
        "\n2. 模板中的固定行全部保留，即使指标分数为 0 也必须单独展示。"
        "raw_values 和 calculated_values 的占位行各自在原位置展开为全部字段的表格行，"
        "指标状态、指标分数、单位、观测日期和来源链接排在全部原始值及计算值之后。"
        "“字段”列必须使用中文语义名称，禁止直接输出原始英文键名、"
        "带下划线的变量名或字段路径；必要的指标符号可保留，如 OI、MA50。"
        "保留统计窗口和对应单位，不遗漏字段，不重复展示。"
        "数值、单位、日期和分数严格采用输入，不重新计算或修改精度。"
        "0 和 false 正常展示，null 展示为 N/A；未提供的字段不自行补充。"
        "列表和对象完整展示，多条内容用 <br> 分隔；正文中的竖线须转义，避免破坏表格。"
        "来源链接使用 url，不展示 reason 或原始 API 地址 source。"
        "\n3. 最后输出以下思想体系解读表："
        "\n### 思想体系解读\n\n| 字段 | 内容 |\n| --- | --- |"
        "\n| 来源章节 | 实际引用材料的 metadata.section |"
        "\n| 当前解读 | 思想原则、指标与原则的关系、信号分歧和判断限制 |"
        "\n按当前 Skill 的解读重点结合指标分析，只引用实际使用的材料，"
        "来源章节去重，不将文档 ID 当作章节。"
        "没有思想材料时，来源章节为 N/A，仅解释成功指标及限制，说明缺少思想依据；"
        "没有成功指标时，仅解释材料原则及适用限制，不判断当前行情。"
        "\n4. 只使用提供的指标和材料，不补造数据、趋势、来源或因果关系。"
        "只输出完整 Markdown，不输出 JSON、字段路径或工具执行过程。"
        f"\n分析对象：{analysis_subject}；技能：{skill_name}。"
        f"\nSkill 说明：\n{skill_instructions}"
        f"\n成功指标：\n{json.dumps(indicators, ensure_ascii=False)}"
        f"\n思想材料：\n{json.dumps(thought_context, ensure_ascii=False)}"
    )


def analysis_report_prompt(
    target_asset: str,
    original_question: str,
    skill_results: list[SkillResult],
    history_summary: str,
) -> str:
    """结合技能指标与已有思想解读回答本轮问题，历史摘要仅作背景。"""
    return (
        "你是加密资产分析报告撰写者，依据提供的技能内容回答用户问题。"
        "\n\n1. 回答目标"
        "\n先直接回应用户问题；证据充分时给出判断，"
        "不足时说明无法确定的部分及缺少的信息。"
        "综合相关技能，解释支持判断的依据及限制，避免逐项复述指标。"
        "\n\n2. 输入使用"
        "\n以技能表格作为数值依据，保留单位、统计范围、观察窗口和观测日期，"
        "不自行计算或补造数值、分数及外部事实。"
        "技能解读属于分析意见，与表格冲突或超出数据支持范围的表述不作为确定依据。"
        "区分不同时间窗口的信号，不将单个观测值描述为变化趋势，"
        "不将相关性表述为已证实的因果关系。"
        "历史摘要仅用于理解此前讨论；与本轮技能内容冲突时，以本轮内容为准。"
        "缓存数据按其观测日期描述，不宣称已实时更新。"
        "\n\n3. 思想依据"
        "\n使用技能中与问题相关的已有思想解读，按"
        "“思想原则 → 当前指标如何符合或不符合 → 对判断的影响”组织论证。"
        "在对应论述附近沿用已有来源章节，不只罗列书名。"
        "未提供原文时不声称已核验原文，不扩展未提供的理论或伪造原文引语；"
        "缺少相关思想解读时说明局限，不为引用而强套理论。"
        "\n\n4. 操作边界"
        "\n涉及加仓、减仓或买卖时，用已有指标和参考位说明成立及失效条件。"
        "缺少依据或用户个人条件时，不新增价格阈值、仓位比例或操作数量。"
        "不从思想解读中的做空或杠杆表述推定用户的操作需求。"
        "\n\n5. 输出格式"
        "\n用中文输出约500—600字的报告正文，引用随相关论述呈现。"
        "不输出完整指标表格、JSON、内部字段名或工具调用过程。"
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
