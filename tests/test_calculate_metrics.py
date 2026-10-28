"""使用固定采集案例测试真实指标节点，运行时请求模型与数据源。

运行：python -m pytest tests/test_calculate_metrics.py -v -s
需要配置模型及对应数据源的环境变量。固定输入不依赖运行时读取采集文件。
仅校验执行结果结构和工具范围，不验证公式或数值正确性。
"""

import json
from math import isfinite
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.crypto_analysis.crypto_analysis_state import CryptoAnalysisSubGraphState
from app.crypto_analysis.crypto_analysis_sub_agent import make_calculate_metrics
from app.crypto_analysis.skill_loader import LoadedSkill


class TestCalculateMetrics:
    """使用六类技能的采集输入，执行当前提示词和真实注册工具。"""

    @pytest.fixture
    def scope(self) -> dict[str, str]:
        """固定使用选中采集案例的请求范围。"""
        return {
            "as_of": "2026-10-27T13:35:16.179751+00:00",
            "thread_id": "web-f248cb97",
            "trace_id": "cdb03b9e",
            "original_question": "BTC 现在适合加仓吗？",
            "resolved_question": "比特币（BTC）目前是否适合进行加仓操作？",
            "target_asset": "BTC",
        }

    @pytest.mark.asyncio(loop_scope="module")
    @pytest.mark.parametrize(
        "skill_config",
        [
            # 采集时间：2026-10-27T21:35:26.622201+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-trend",
                    "description": "适用于价格走势、趋势方向与强度、均线 MA50/MA200、ADX 和 DI 分析。",
                    "tools": ["analyze_trend"],
                    "scoring": {
                        "name": "趋势与均线",
                        "metrics": [
                            {
                                "id": "Trend",
                                "label": "趋势与均线",
                                "weight": 1.0,
                                "kind": "numeric",
                            }
                        ],
                    },
                },
                id="趋势与均线",
            ),
            # 采集时间：2026-10-27T21:35:27.392831+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-derivatives",
                    "description": "适用于合约资金面、持仓量（OI）、资金费率、杠杆和多空拥挤分析。",
                    "tools": ["analyze_derivatives"],
                    "scoring": {
                        "name": "衍生品数据",
                        "metrics": [
                            {
                                "id": "OI",
                                "label": "未平仓量",
                                "weight": 0.6,
                                "kind": "numeric",
                            },
                            {
                                "id": "Funding",
                                "label": "资金费率",
                                "weight": 0.4,
                                "kind": "numeric",
                            },
                        ],
                    },
                },
                id="持仓量与资金费率",
            ),
            # 采集时间：2026-10-27T21:35:31.794048+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-etf-flows",
                    "description": "适用于现货 ETF 资金面、净流入净流出及机构申购赎回分析。",
                    "tools": ["analyze_etf_flows"],
                    "scoring": {
                        "name": "ETF资金流",
                        "metrics": [
                            {
                                "id": "ETF_Flows",
                                "label": "ETF资金流",
                                "weight": 1.0,
                                "kind": "numeric",
                            }
                        ],
                    },
                },
                id="ETF资金流",
            ),
            # 采集时间：2026-10-27T21:35:29.037860+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-sentiment",
                    "description": "适用于恐惧贪婪指数、市场恐慌、贪婪和极端情绪分析。",
                    "tools": ["analyze_sentiment"],
                    "scoring": {
                        "name": "市场情绪",
                        "metrics": [
                            {
                                "id": "Fear_Greed",
                                "label": "恐惧与贪婪指数",
                                "weight": 1.0,
                                "kind": "numeric",
                            }
                        ],
                    },
                },
                id="市场情绪",
            ),
            # 采集时间：2026-10-27T21:35:27.596965+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-fed-policy",
                    "description": "适用于美联储加息、降息、政策利率变化及鹰鸽倾向分析。",
                    "tools": ["analyze_fed_policy"],
                    "scoring": {
                        "name": "美联储政策与利率倾向",
                        "metrics": [
                            {
                                "id": "Fed_Rate_Change_3M",
                                "label": "三个月政策利率净变化",
                                "weight": 1.0,
                                "kind": "numeric",
                            }
                        ],
                    },
                },
                id="美联储政策利率",
            ),
            # 采集时间：2026-10-27T21:35:30.418902+08:00。
            pytest.param(
                {
                    "name": "analyzing-crypto-liquidity",
                    "description": "适用于美元及加密市场流动性分析，包括 DXY、M2、WALCL 和 USDT+USDC 稳定币市值。",
                    "tools": ["analyze_liquidity"],
                    "scoring": {
                        "name": "美元流动性与加密内部流动性",
                        "metrics": [
                            {
                                "id": "DXY_Trend",
                                "label": "美元指数",
                                "weight": 0.35,
                                "kind": "numeric",
                            },
                            {
                                "id": "Stable_MCap",
                                "label": "稳定币总市值",
                                "weight": 0.3,
                                "kind": "numeric",
                            },
                            {
                                "id": "M2_YoY",
                                "label": "M2 同比变化",
                                "weight": 0.2,
                                "kind": "numeric",
                            },
                            {
                                "id": "Fed_Balance",
                                "label": "美联储资产负债表",
                                "weight": 0.15,
                                "kind": "numeric",
                            },
                        ],
                    },
                },
                id="流动性四项指标",
            ),
        ],
    )
    async def test_calculate_metrics_returns_indicator_results(
        self, skill_config: dict[str, Any], scope: dict[str, str]
    ) -> None:
        skill = LoadedSkill.model_validate(skill_config)
        state = CryptoAnalysisSubGraphState(
            skill_name=skill.name,
            skill=skill,
            scope=scope.copy(),
            cached_memory=None,
            messages=[],
            evidence={},
            indicator_results={},
            metric_result={},
            skill_result={},
            skill_memory={},
        )
        calculate_metrics = make_calculate_metrics()
        result = await calculate_metrics(state)
        indicators = result["indicator_results"]

        print(
            f"\n技能：{skill.name}\n指标结果：\n"
            + json.dumps(indicators, ensure_ascii=False, indent=2)
        )
        assert set(indicators) == {metric.id for metric in skill.scoring.metrics}
        assert result["evidence"]

        tool_calls = [
            call
            for message in result["messages"]
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
        assert tool_calls
        assert {call["name"] for call in tool_calls} <= set(skill.tools)

        for metric_id, indicator in indicators.items():
            assert indicator["indicator_id"] == metric_id
            assert indicator["skill_name"] == skill.name
            assert indicator["status"] in {
                "ok",
                "data_missing",
                "not_applicable",
                "calculation_error",
            }
            if indicator["status"] == "ok":
                score = indicator["score"]
                assert isinstance(score, (int, float)) and not isinstance(score, bool)
                assert isfinite(score) and 0 <= score <= 10
            else:
                assert indicator["reason"]
                assert indicator["score"] is None
