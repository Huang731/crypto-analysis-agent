from __future__ import annotations

import asyncio
from typing import Any

import jieba
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.core.tracing import trace_generation_call, trace_span
from app.llm.model import make_glm_4_5_air_llm, rerank
from app.rag.ingest import get_vectorstore


def _tokenize_zh(text: str) -> list[str]:
    """jieba 中文分词，过滤空白 token。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def load_corpus(vs: FAISS) -> list[Document]:
    """读取 FAISS 中的全部文档块，供 BM25 在内存中建立索引。"""
    ids = list(vs.index_to_docstore_id.values())
    return vs.get_by_ids(ids)


def build_hybrid_retriever() -> BaseRetriever:
    """构建混合检索器：BM25 + 向量，加权 RRF 融合。

    可选 MultiQueryRetriever 包装（settings.enable_multi_query）：
    用配置的执行模型把问题改写成多个视角再检索，提升口语化提问的召回，
    代价是每次多一轮 LLM 调用，延迟敏感场景默认关。
    """
    with trace_span("rag.retriever.build") as node_span:
        try:
            vs = get_vectorstore()
            corpus = load_corpus(vs)
            if not corpus:
                raise RuntimeError("知识库为空，请先执行 python -m app.cli")

            # 关键词检索使用中文分词，向量检索沿用已有 FAISS 索引配置。
            bm25 = BM25Retriever.from_documents(
                corpus,
                preprocess_func=_tokenize_zh,
                k=settings.retrieve_k,
            )
            vector = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retrieve_k},
            )

            hybrid = EnsembleRetriever(
                retrievers=[bm25, vector],
                weights=[settings.bm25_weight, settings.vector_weight],
                id_key="chunk_idx",
            )

            if settings.enable_multi_query:
                return MultiQueryRetriever.from_llm(
                    retriever=hybrid,
                    llm=make_glm_4_5_air_llm().bind(
                        extra_body={"thinking": {"type": "disabled"}}
                    ),
                )
            return hybrid
        except Exception:
            node_span.metadata["status"] = "error"
            raise


@trace_generation_call(
    name="rag.rerank",
    model=settings.rerank_model,
    input_arg="docs",
    input_format="documents",
    output_format="none",
    metadata=lambda arguments: {"candidate_count": len(arguments.arguments["docs"])},
    generation_arg="generation",
    kind="generation",
)
def _call_rerank(query: str, docs: list[Document], *, generation: Any) -> Any:
    """保留重排返回值检查，用量按候选文档估算，不依赖可选的计费字段。"""
    try:
        resp = rerank(query, [doc.page_content for doc in docs])
    except Exception:
        generation.metadata["status"] = "error"
        raise
    generation.output = {"result_count": len(resp["results"])}
    return resp


def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """使用硅基流动重排候选文档，保留文档 ID 和元数据并附加相关性分数。"""
    with trace_span("rag.rerank.documents") as node_span:
        try:
            top_n = settings.final_k if top_n is None else top_n
            if top_n <= 0:
                raise ValueError("重排返回数量 top_n 必须大于 0")
            if not docs:
                return []
            if len(docs) <= 1:
                return docs[:top_n]

            resp = _call_rerank(query, docs)
            results = resp["results"]
            ordered = sorted(
                results, key=lambda r: (-r["relevance_score"], r["index"])
            )[:top_n]

            # 不可变：不改原 Document，构造新对象带上 rerank_score
            return [
                Document(
                    id=docs[r["index"]].id,
                    page_content=docs[r["index"]].page_content,
                    metadata={
                        **docs[r["index"]].metadata,
                        "rerank_score": r["relevance_score"],
                    },
                )
                for r in ordered
            ]
        except Exception:
            node_span.metadata["status"] = "error"
            raise


class KBRetriever:
    """检索层门面：建一次索引，答多次问题（BM25 建索引有成本，不能每问重建）。

    rerank —— 混合检索 + 硅基流动重排。
    """

    MODES = ("vector", "hybrid", "rerank")

    def __init__(self) -> None:
        self.hybrid = build_hybrid_retriever()
        self._semaphore = asyncio.Semaphore(settings.retrieval_max_concurrency)
        self._pending: set[asyncio.Task[list[Document]]] = set()

    async def aretrieve(self, question: str) -> list[Document]:
        """限制共享实例的检索并发，将同步 SDK 调用移出事件循环。"""
        with trace_span("rag.retrieve.queue"):
            await self._semaphore.acquire()
        task = asyncio.create_task(asyncio.to_thread(self.retrieve, question))
        self._pending.add(task)
        task.add_done_callback(self._retrieval_done)
        # 请求取消不终止同步 SDK；实际结束后再释放名额，避免突破并发上限。
        return await asyncio.shield(task)

    def _retrieval_done(self, task: asyncio.Task[list[Document]]) -> None:
        self._pending.discard(task)
        self._semaphore.release()
        if not task.cancelled():
            task.exception()  # 消费被取消请求遗留的异常，正常调用仍会收到异常。

    async def aclose(self) -> None:
        """服务退出时等待已经开始的同步检索结束。"""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    def retrieve(self, question: str) -> list[Document]:
        with trace_span("rag.retrieve", retrieve_k=settings.retrieve_k) as span:
            with trace_span("rag.retrieve.recall") as recall_span:
                candidates = self.hybrid.invoke(question)[: settings.retrieve_k]
                recall_span.output = {"candidate_count": len(candidates)}
            print(f"\n重排前召回 {len(candidates)} 个候选文档块")
            for i, candidate in enumerate(candidates, start=1):
                print(f"\n候选文档块 {i}")
                print(f"元数据：{candidate.metadata}")
                print(candidate.page_content)
            documents = rerank_documents(question, candidates, top_n=settings.final_k)
            span.output = {
                "candidate_count": len(candidates),
                "document_count": len(documents),
            }
            return documents


def main() -> None:
    """使用候选记录中的六个原始问题，逐题查看检索结果。"""
    retriever = KBRetriever()
    questions = [
        (
            "为什么一个国家即使所有产业的生产效率都低于贸易伙伴，仍能从国际贸易中获益，而国内某些行业和劳动者却可能受损？"
        ),
        # (
        #     "当价格高于50日及200日均线且两条均线较20日前抬升，"
        #     "但负向DI高于正向DI时，如何综合判断趋势与动量？"
        # ),
        # (
        #     "如何分析比特币现货ETF的24小时、7日及30日净流入数据以判断资金流向趋势？"
        #     "当持仓量7日微增但现货价格7日下跌，且资金费率微弱偏多时，"
        #     "如何分析多空分歧与市场状态？"
        # ),
        # (
        #     "如何分析美联储在三个月内目标利率中点从3.625%升至3.875%、"
        #     "净变化25个基点的政策调整对加密市场的影响？"
        # ),
        # (
        #     "当美元指数处于上行趋势且高于50日及200日均线，"
        #     "同时稳定币市值较30日均值下跌但无方向冲突，而M2同比增速连续上升时，"
        #     "如何分析流动性扩张与资产价格的关系？"
        # ),
    ]
    for question_idx, question in enumerate(questions, start=1):
        print(f"\n问题 {question_idx}/{len(questions)}：{question}")
        documents = retriever.retrieve(question)
        print(f"检索到 {len(documents)} 个文档块")
        for i, document in enumerate(documents, start=1):
            print(f"\n文档块 {i}")
            print(f"元数据：{document.metadata}")
            print(document.page_content[:100])


if __name__ == "__main__":
    main()
