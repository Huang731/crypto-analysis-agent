"""使用边界问题测试真实 Router，需要配置 ZHIPUAI_4_API_KEY。

运行：python -m pytest tests/test_router.py -v -s
仅验证路由输出合法，并打印问题与本次路由结果。
"""

from unittest.mock import patch

import pytest

from app.core.config import settings
from app.routing.graph import RoutingState, make_router


class TestRouter:
    """将问题固定为测试参数，直接作为 resolved_question 使用。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resolved_question",
        [
            # "BTC 最近持续上涨，现在应该减仓还是继续持有？",
            # "ETH 的资金费率和合约持仓量是否显示多头过度拥挤？",
            # "结合 BTC 近期的价格趋势和成交量，现在适合加仓吗？",
            # "分析 BTC 近期的资金流向和市场情绪。",
            # "解释比特币工作量证明机制的原理及其安全性。",
            # "解释以太坊的技术架构和交易处理机制。",
            # "研究比特币挖矿公司的商业模式及主要成本来源。",
            # "研究美联储降息政策对全球经济的传导机制。",
            # "黄金最近的走势如何，现在适合买入吗？",
            "美元指数、稳定币市值和美元流动性对比特币（BTC）的影响是偏利好还是偏利空？",
            "比特币ETF近期资金流较此前是改善还是走弱？",
            # # 日常问答与生活场景。
            # "你好，介绍一下你能帮我做什么。",
            # "谢谢你的帮助，今天先聊到这里。",
            # "长沙明天会下雨吗，出门需要带伞吗？",
            # "家里只有鸡蛋、西红柿和面条，晚饭可以怎么做？",
            # "帮我安排一个周末整理房间和阅读的时间表。",
            # # 非加密领域的专题研究。
            # "研究新能源汽车行业的竞争格局和主要增长动力。",
            # "分析苹果公司的商业模式及主要收入来源。",
            # "比较 PostgreSQL 和 MySQL 在事务处理上的差异。",
            # "研究生成式人工智能在企业客服中的应用与局限。",
            # "分析人口老龄化对养老服务行业需求的影响。",
        ],
        ids=lambda value: value.strip(),
    )
    async def test_router_returns_valid_route(self, resolved_question: str) -> None:
        state = RoutingState(
            messages=[],
            scope={},
            resolved_question=resolved_question,
            route="",
            report="",
        )
        # 开启候选采集，保存本次测试的路由输入和结果。
        router = make_router()
        result = await router(state)

        print(f"\n问题：{resolved_question.strip()}\n本次路由：{result['route']}")
        assert set(result) == {"route"}
        assert result["route"] in (
            "crypto_analysis_agent",
            "topic_research_agent",
        )
