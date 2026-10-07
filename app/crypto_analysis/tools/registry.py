"""按名称提供应用已实现的工具，供 Skill 配置选择。"""

from langchain_core.tools import BaseTool


def get_tool_registry() -> dict[str, BaseTool]:
    """延迟导入工具，保留现有共享函数的导入关系。"""
    from app.crypto_analysis.tools.derivatives_tools import build_derivatives_tools
    from app.crypto_analysis.tools.etf_tools import build_etf_tools
    from app.crypto_analysis.tools.fed_policy_tools import build_fed_policy_tools
    from app.crypto_analysis.tools.liquidity_tools import build_liquidity_tools
    from app.crypto_analysis.tools.sentiment_tools import build_sentiment_tools
    from app.crypto_analysis.tools.trend_tools import build_trend_tools

    return {
        tool.name: tool
        for tool in (
            *build_liquidity_tools(),
            *build_fed_policy_tools(),
            *build_etf_tools(),
            *build_derivatives_tools(),
            *build_trend_tools(),
            *build_sentiment_tools(),
        )
    }
