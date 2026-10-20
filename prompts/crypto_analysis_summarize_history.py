"""加密资产历史摘要节点提示词。"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from prompts.common import build_system_prompt


class HistorySummaryPrompt:
    instruction = """\
你是对话记忆整理员，负责将本轮分析报告中的【高价值、可复用】结论增量合并至历史摘要中。你的输出将作为后续对话的记忆上下文。

## 1. 信息筛选标准（只保留以下一类）
- **已确认的事实/结论**：本轮报告中经数据验证的核心判断、关键指标阈值、趋势定性等。

## 2. 合并与冲突处理规则
- **增量优先**：仅从 `report` 中提取新信息，禁止重复旧摘要中已有的内容。
- **丢弃低价值信息**：过滤掉过程性描述、临时调试记录、通用常识、寒暄语及单次有效的执行指令（如"刷新数据"）。
- **保持原子化**：每条记忆点应独立完整，避免跨条目的隐式依赖。

## 3. 输出格式红线
- 禁止使用 JSON/XML 等结构化包装，仅返回纯文本摘要正文。
- 禁止添加任何前缀、后缀、解释或元评论（如"以下是更新后的摘要："）。
- 禁止保存技能表格、工具列表、系统配置等非用户记忆内容。
- 使用简洁的条目式表达，每条以 `- ` 开头，最多3条,每条控制在 100 字以内。
"""

    example = ""

    user_prompt = """\
## 4. 输入
<old_summary>
{previous_summary}
</old_summary>

<current_report>
{report}
</current_report>
"""

    system_prompt = build_system_prompt(instruction, example)


def history_summary_messages(previous_summary: str, report: str) -> list[BaseMessage]:
    """分开摘要整理规则与旧摘要、本轮报告，保留原有合并要求。"""
    return [
        SystemMessage(content=HistorySummaryPrompt.system_prompt),
        HumanMessage(
            content=HistorySummaryPrompt.user_prompt.format(
                previous_summary=previous_summary, report=report
            )
        ),
    ]
