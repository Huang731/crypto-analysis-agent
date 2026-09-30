"""分派前用 LLM 把口语化问题改写为意图明确的完整问题。"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel  # 标注模型接口。
from langchain_core.messages import HumanMessage, SystemMessage  # 构造改写对话。

from app.observability import get_logger, log_event

_LOG = get_logger("crypto_analysis.query_rewriter")
_TIMEOUT_SECONDS = 15  # 改写只是前置优化，超时即放弃，不能拖慢主流程。
_MAX_CHARS = 200  # 改写结果超过该长度视为模型跑题。

_SYSTEM_PROMPT = (
    "你是问题改写助手。把用户的口语化、简略或含俚语的问题改写成一句意图明确、表述完整的中文问题，供下游调度员判断问题类型。\n"
    "规则：\n"
    "1. 只改写表述，不改变意图，不补充用户没提到的条件、数据或结论；\n"
    "2. 加密货币俚语改为规范名称，例如：大饼=比特币（BTC），姨太=以太坊（ETH），狗狗=狗狗币（DOGE），"
    "U/稳定币=USDT 等稳定币，上车=买入，梭哈=全仓买入，割肉=止损卖出；"
    "用户直接写的币种代号（如 sol、bnb、pepe）保持原币种，只转为大写代号，绝不替换成其他币种；"
    "不确定含义的词原样保留；\n"
    "3. 问题本身已清晰时原样返回；寒暄、闲聊或日常问题（如天气）只做规范化表述，不要硬改成研究问题；\n"
    "4. 你不是在和用户对话，绝不回答或回应问题，改写结果必须仍是用户视角的原话；\n"
    "5. 只输出改写后的问题本身，不加引号、解释或前缀。\n"
    "示例：\n"
    "大饼还能上车不 → 比特币（BTC）现在还适合买入吗？\n"
    "sol咋样 → SOL 现在行情怎么样？\n"
    "天气咋样 → 今天天气怎么样？\n"
    "你好 → 你好\n"
    "在吗 → 在吗？\n"
    "你是谁 → 你是谁？\n"
    "你能干啥 → 你能做什么？"
)


async def rewrite_question(llm: BaseChatModel, question: str) -> str:
    """返回改写后的问题；模型失败、超时或输出异常时返回原问题。"""
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception:  # noqa: BLE001 - 改写失败不应阻断分派，退回原问题即可。
        _LOG.warning("query.rewrite.failed", extra={"event": "query.rewrite.failed"}, exc_info=True)
        return question

    rewritten = response.text.strip().strip("\"'“”")
    if not rewritten or len(rewritten) > _MAX_CHARS:
        log_event(_LOG, "query.rewrite.discarded", output_chars=len(rewritten))
        return question
    log_event(_LOG, "query.rewrite.done", original=question, rewritten=rewritten)
    return rewritten
