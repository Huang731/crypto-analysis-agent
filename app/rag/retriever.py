from __future__ import annotations

from http import HTTPStatus

import jieba
from dashscope import TextReRank
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.ingest import get_vectorstore
from app.config.config import settings
from app.llm.model import make_fast_llm
from app.tracing import trace_generation, trace_span


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
    用 glm-4-flash 把问题改写成多个视角再检索，提升口语化提问的召回，
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
            )

            if settings.enable_multi_query:
                return MultiQueryRetriever.from_llm(
                    retriever=hybrid,
                    llm=make_fast_llm(),
                )
            return hybrid
        except Exception:
            node_span.metadata["status"] = "error"
            raise

def rerank_documents(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """使用百炼重排候选文档，保留文档 ID 和元数据并附加相关性分数。"""
    with trace_span("rag.rerank.documents") as node_span:
        try:
            top_n = settings.final_k if top_n is None else top_n
            if top_n <= 0:
                raise ValueError("重排返回数量 top_n 必须大于 0")
            if not docs:
                return []
            if len(docs) <= 1:
                return docs[:top_n]

            if not settings.dashscope_api_key:
                raise ValueError("百炼重排需要配置 DASHSCOPE_API_KEY")

            with trace_generation(
                "rag.rerank", model=settings.rerank_model, candidate_count=len(docs)
            ) as generation:
                resp = TextReRank.call(
                    api_key=settings.dashscope_api_key,
                    model=settings.rerank_model,
                    query=query,
                    documents=[doc.page_content for doc in docs],
                    top_n=len(docs),  # 获取全部分数，在本地统一排序和截断。
                    return_documents=False,
                )
                if resp.status_code != HTTPStatus.OK:
                    generation.metadata["status"] = "error"
                    raise RuntimeError(f"百炼重排失败：{resp.code}，{resp.message}")
                results = resp.output.results
                # 重排只产生相关性分数，用量采用接口返回的总输入 token。
                generation.usage = {"input": resp.usage.total_tokens, "output": 0}
                generation.output = {"result_count": len(results)}
            ordered = sorted(results, key=lambda r: (-r.relevance_score, r.index))[:top_n]

            # 不可变：不改原 Document，构造新对象带上 rerank_score
            return [
                Document(
                    id=docs[r.index].id,
                    page_content=docs[r.index].page_content,
                    metadata={**docs[r.index].metadata, "rerank_score": r.relevance_score},
                )
                for r in ordered
            ]
        except Exception:
            node_span.metadata["status"] = "error"
            raise

class KBRetriever:
    """检索层门面：建一次索引，答多次问题（BM25 建索引有成本，不能每问重建）。

        rerank —— 混合检索 + 百炼重排。
    """

    MODES = ("vector", "hybrid", "rerank")

    def __init__(self):
        self.hybrid = build_hybrid_retriever()

    def retrieve(self, question: str) -> list[Document]:
        with trace_span("rag.retrieve", retrieve_k=settings.retrieve_k) as span:
            candidates = self.hybrid.invoke(question)[: settings.retrieve_k]
            documents = rerank_documents(question, candidates, top_n=settings.final_k)
            span.output = {"candidate_count": len(candidates), "document_count": len(documents)}
            return documents
