"""使用固定上下文测试真实改写节点，需要配置 ZHIPUAI_4_API_KEY。

运行：python -m pytest tests/test_query_rewriter.py -v -s
使用当前改写提示词，验证切换到工资话题后不再继承历史币种。
"""

import pytest
from langchain_core.messages import HumanMessage

from app.routing.graph import RoutingState, make_rewrite


class TestQueryRewriter:
    """使用采集案例中的历史问题和本轮原话作为固定参数。"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("history_questions", "original_question"),
        [
            pytest.param(
                ["比特币（BTC）目前是否适合进行加仓操作？"],
                "长沙的工资",
                id="切换到长沙工资话题",
            ),
        ],
    )
    async def test_rewrite_does_not_inherit_unrelated_asset(
        self, history_questions: list[str], original_question: str
    ) -> None:
        state = RoutingState(
            messages=[HumanMessage(content=question) for question in history_questions],
            scope={"original_question": original_question},
            resolved_question="",
            route="",
            report="",
        )
        rewrite = make_rewrite()
        result = await rewrite(state)
        resolved_question = result["resolved_question"]

        print(
            f"\n历史问题：{history_questions}\n"
            f"本轮原话：{original_question}\n"
            f"改写结果：{resolved_question}"
        )
        assert isinstance(resolved_question, str)
        assert resolved_question.strip()
        assert "长沙" in resolved_question
        assert "比特币" not in resolved_question
        assert "btc" not in resolved_question.lower()
