"""按 Skill 创建独立分析师，自动加载技能内容作为提示词。"""

from collections.abc import Sequence

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.tools import BaseTool

from app.core.config import settings
from app.crypto_analysis.skill_loader import get_skill_loader
from app.llm.model import make_glm_4_5_air_llm


class SkillAnalyst:
    """每个实例对应一个 Skill，独立执行工具调用并生成分析结果。"""

    def __init__(
        self,
        skill_name: str,
        tools: Sequence[BaseTool],
    ) -> None:
        """按名称加载 Skill 全文，绑定调用方传入的工具。"""
        self.skill_name = skill_name
        system_prompt = get_skill_loader().load_skill(skill_name)
        # 模型工厂已设置单次请求超时为 120 秒，重试仅作用于模型请求。
        self.agent = create_agent(
            model=make_glm_4_5_air_llm(),
            system_prompt=system_prompt,
            tools=list(tools),
            state_schema=AgentState,
            middleware=[
                ModelRetryMiddleware(max_retries=1, on_failure="error"),
                ToolCallLimitMiddleware(
                    run_limit=settings.max_tool_calls_per_skill,
                    exit_behavior="continue",
                ),
            ],
            name=skill_name,
        )

    async def ainvoke(self, state: AgentState) -> AgentState:
        """执行本轮分析，返回包含工具消息和分析结果的完整状态。"""
        return await self.agent.ainvoke(state)
