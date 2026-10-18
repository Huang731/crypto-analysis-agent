"""根据检索材料流式生成回答。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.llm.model import make_bailian_smart_llm
from prompts.rag_generate import SAFE_SYSTEM_PROMPT

_SYSTEM_PROMPT = SAFE_SYSTEM_PROMPT


def build_context(docs: list[Document]) -> str:
    """将检索材料编号，并放入系统提示词指定的材料标签内。"""
    parts = [f"【材料{i}】\n{doc.page_content}" for i, doc in enumerate(docs, 1)]
    body = "\n\n".join(parts)
    return f"<begin_retrieved_documents>\n{body}\n</end_retrieved_documents>"


async def stream_answer(
    question: str,
    docs: list[Document],
) -> AsyncIterator[str]:
    """流式生成回答，逐 token yield。"""
    messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
    messages.append(
        HumanMessage(content=f"【材料】\n{build_context(docs)}\n\n【问题】{question}")
    )

    llm = make_bailian_smart_llm()
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield str(chunk.content)
