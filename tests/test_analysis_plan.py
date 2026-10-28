"""使用固定采集案例测试真实 Plan，需要配置 ZHIPUAI_4_API_KEY。

运行：python -m pytest tests/test_analysis_plan.py -v -s
问题和技能目录固定自采集记录，测试运行时不读取采集文件。
"""

import pytest

from app.crypto_analysis.crypto_analysis_parent_agent import create_analysis_plan
from app.llm.model import make_glm_4_5_air_llm
from prompts.crypto_analysis_plan import AnalysisPlan


class TestAnalysisPlan:
    """覆盖技能范围、币种提取及刷新要求，打印当前提示词的规划结果。"""

    @pytest.fixture
    def skills(self) -> dict[str, str]:
        """固定使用采集记录中的六项技能及其描述。"""
        return {
            "analyzing-crypto-derivatives": (
                "适用于合约资金面、持仓量（OI）、资金费率、杠杆和多空拥挤分析。"
            ),
            "analyzing-crypto-etf-flows": (
                "适用于现货 ETF 资金面、净流入净流出及机构申购赎回分析。"
            ),
            "analyzing-crypto-fed-policy": (
                "适用于美联储加息、降息、政策利率变化及鹰鸽倾向分析。"
            ),
            "analyzing-crypto-liquidity": (
                "适用于美元及加密市场流动性分析，"
                "包括 DXY、M2、WALCL 和 USDT+USDC 稳定币市值。"
            ),
            "analyzing-crypto-sentiment": (
                "适用于恐惧贪婪指数、市场恐慌、贪婪和极端情绪分析。"
            ),
            "analyzing-crypto-trend": (
                "适用于价格走势、趋势方向与强度、均线 MA50/MA200、ADX 和 DI 分析。"
            ),
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "question",
        [
            pytest.param("比特币现在适合加仓吗", id="综合分析"),
            pytest.param("BTC只分析ETF资金流", id="指定单项技能"),
            pytest.param("BTC只看均线ADX和合约资金费率", id="指定多个领域"),
            pytest.param("综合分析BTC，但不看市场情绪", id="排除指定维度"),
            pytest.param("只分析ETF资金流", id="未指定币种"),
            pytest.param("只评估sol的MA50、MA200和ADX", id="小写币种"),
            pytest.param("刷新BTC资金费率数据，这次只分析资金费率", id="明确刷新"),
            pytest.param("BTC只分析ETF资金流，不要用缓存", id="不用缓存"),
            pytest.param("获取ETH最新资金费率数据，只分析资金费率", id="获取最新数据"),
            pytest.param("再分析一次BTC的ETF资金流，只看ETF", id="再次分析不等于刷新"),
        ],
    )
    async def test_analysis_plan_returns_valid_selection(
        self, question: str, skills: dict[str, str]
    ) -> None:
        skill_descriptions = "\n".join(
            ["【可用技能 Skills】"]
            + [f"  - {name}: {description}" for name, description in skills.items()]
        )
        llm = make_glm_4_5_air_llm()
        result = await create_analysis_plan(llm, question, skill_descriptions)
        plan = AnalysisPlan.model_validate(result)

        print(f"\n问题：{question}\n规划结果：{plan.model_dump_json(indent=2)}")
        assert plan.skill_names
        assert set(plan.skill_names) <= set(skills)
        assert plan.skill_names == sorted(set(plan.skill_names))
