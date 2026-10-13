"""Ingest 管线：文档 → 分块 → 向量化 → FAISS 持久化（增量缓存）。

增量同步：
    新文件      → 全量入库
    文件被修改  → 根据 source 对应的 ID 删除旧块，再入库新块
    文件被删除  → prune 时清掉库里的孤儿块
开启 prune 时，清理目录中已不存在的文件对应的切片。
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss

# 官方已停止维护 langchain-community，但 FAISS 与百炼向量还没有独立集成包。
# 导入时会发 DeprecationWarning；langchain-classic 仍转调这个包，过滤这条提示即可。
warnings.filterwarnings(
    "ignore",
    message="`langchain-community` is being sunset",
    category=DeprecationWarning,
)
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.config.config import settings
from app.rag.loader import file_md5, list_documents, load_and_split
from app.tracing import start_trace, trace_generation_call, trace_span


@contextmanager
def _ignore_ip_normalize_warning() -> Iterator[None]:
    """LangChain 认为 L2 归一化只适用于欧氏距离，因此会警告。

    内积索引配合单位向量时，归一化后的内积就是 cosine，警告不成立，归一化仍会执行。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Normalizing L2 is not applicable for metric type:",
            category=UserWarning,
        )
        yield


def get_embeddings() -> DashScopeEmbeddings:
    """使用 LangChain 的百炼集成，向量维度采用模型默认值。"""
    if not settings.dashscope_api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY（.env 或环境变量）")
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
    )


def get_vectorstore() -> FAISS:
    """加载已有索引，首次入库时创建空库。

    文档向量与查询向量均归一化，内积即 cosine 相似度，越大越相似。
    similarity_search 返回文档，similarity_search_with_score 返回原始 cosine 分数。
    """
    with trace_span("rag.vectorstore.load") as node_span:
        try:
            embeddings = get_embeddings()
            index_dir = Path(settings.faiss_path)
            # 已有任一索引文件时执行加载；文件缺失或损坏直接报错。
            if (index_dir / "index.faiss").exists() or (
                index_dir / "index.pkl"
            ).exists():
                with _ignore_ip_normalize_warning():
                    vs = FAISS.load_local(
                        str(index_dir),
                        embeddings,
                        allow_dangerous_deserialization=True,  # 仅加载本项目生成的可信索引。
                        normalize_L2=True,
                        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                    )
                if vs.index.metric_type != faiss.METRIC_INNER_PRODUCT:
                    raise ValueError(
                        "已有 FAISS 索引不是内积索引，请使用新索引目录重新入库"
                    )
                if vs.index.d != settings.embedding_dimensions:
                    raise ValueError(
                        "已有 FAISS 索引的维度与向量模型不一致，请使用新索引目录重新入库"
                    )
                return vs

            with _ignore_ip_normalize_warning():
                return FAISS(
                    embedding_function=embeddings,
                    index=faiss.IndexFlatIP(settings.embedding_dimensions),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                    normalize_L2=True,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
        except Exception:
            node_span.metadata["status"] = "error"
            raise


@dataclass(frozen=True)
class IngestReport:
    """一次 ingest 的结果统计（不可变，直接进日志/API 响应）。"""

    added_files: tuple[str, ...] = ()
    updated_files: tuple[str, ...] = ()
    skipped_files: tuple[str, ...] = ()
    pruned_files: tuple[str, ...] = ()
    added_chunks: int = 0
    total_chunks: int = 0


def _existing_sources(vs: FAISS) -> tuple[dict[str, list[str]], dict[str, str]]:
    """一次遍历构建文件到切片 ID、文件到内容 hash 的映射。"""
    source_ids: dict[str, list[str]] = {}
    source_hashes: dict[str, str] = {}

    ids = list(vs.index_to_docstore_id.values())
    # get_by_ids 不保证返回顺序，且会忽略缺失文档；按 ID 关联并检查完整性。
    documents = {doc.id: doc for doc in vs.get_by_ids(ids)}
    missing_ids = set(ids) - documents.keys()
    if missing_ids:
        raise ValueError(f"向量索引对应的文档不存在：{sorted(missing_ids)}")

    for doc_id in ids:
        doc = documents[doc_id]
        meta = doc.metadata
        source = meta["source"]
        source_ids.setdefault(source, []).append(doc_id)
        if "src_hash" in meta:
            source_hashes[source] = meta["src_hash"]

    return source_ids, source_hashes


def _delete_source(
    vs: FAISS,
    source: str,
    source_ids: dict[str, list[str]],
    source_hashes: dict[str, str],
) -> None:
    """删除文件对应的全部切片，成功后同步移除两份映射。"""
    vs.delete(ids=source_ids[source])
    del source_ids[source]
    source_hashes.pop(source, None)


def _add_in_batches(vs: FAISS, docs: list[Document], ids: list[str]) -> None:
    """按配置分批入库，百炼 text-embedding-v4 单请求最多 10 条文本。"""
    step = settings.embed_batch_size
    for i in range(0, len(docs), step):
        batch = docs[i : i + step]
        _embed_batch(vs, batch, ids[i : i + step], i // step + 1)


@trace_generation_call(
    name="ingest.embedding",
    model=settings.embedding_model,
    input_arg="batch",
    input_format="documents",
    output_format="none",
    metadata=lambda arguments: {
        "batch_index": arguments.arguments["batch_index"],
        "chunk_count": len(arguments.arguments["batch"]),
    },
    generation_arg="generation",
    unit="TOKENS",
)
def _embed_batch(
    vs: FAISS,
    batch: list[Document],
    ids: list[str],
    batch_index: int,
    *,
    generation: Any,
) -> None:
    """沿用批次入库与异常标记，输入用量由装饰器逐条估算。"""
    try:
        vs.add_documents(batch, ids=ids)
    except Exception as exc:
        generation.metadata.update(status="error", error_type=type(exc).__name__)
        raise
    generation.metadata["status"] = "done"


def ingest_directory(
    docs_dir: str | Path | None = None, prune: bool = True
) -> IngestReport:
    """同步文档目录，并用本地 tracing 记录入库和向量模型调用。"""
    directory = Path(settings.docs_dir if docs_dir is None else docs_dir)
    with start_trace("rag.ingest", docs_dir=str(directory), prune=prune) as trace:
        try:
            report = _ingest_directory(directory, prune=prune)
        except Exception as exc:
            trace.output = {"status": "error", "error_type": type(exc).__name__}
            raise
        trace.output = {"status": "done", **asdict(report)}
        return report


def _ingest_directory(docs_dir: Path, prune: bool) -> IngestReport:
    """把目录下所有文档同步进向量库，返回统计报告。幂等：重复跑不产生变化。"""
    with trace_span("rag.ingest.sync") as node_span:
        try:
            files = list_documents(docs_dir)
            vs = get_vectorstore()
            source_ids, source_hashes = _existing_sources(vs)

            on_disk = {p.name for p in files}

            added: list[str] = []
            updated: list[str] = []
            skipped: list[str] = []
            added_chunks = 0

            for path in files:
                fname = path.name
                fhash = file_md5(path)
                if source_hashes.get(fname) == fhash:
                    skipped.append(fname)
                    continue

                is_update = fname in source_ids
                if is_update:
                    # 文件改过：先清旧块，避免新旧版本混存
                    _delete_source(vs, fname, source_ids, source_hashes)
                # PDF 走版面感知解析；md/txt 走老路
                if path.suffix.lower() == ".pdf":
                    chunks = _load_pdf_as_documents(path)
                else:
                    chunks = load_and_split(path, src_hash=fhash)
                ids = [f"{fname}:{fhash[:8]}:{c.metadata['chunk_idx']}" for c in chunks]
                _add_in_batches(vs, chunks, ids)
                # 入库成功后更新映射，空文件不记录切片 ID。
                if ids:
                    source_ids[fname] = ids
                    source_hashes[fname] = fhash
                added_chunks += len(chunks)
                if is_update:
                    updated.append(fname)
                else:
                    added.append(fname)

            # 目录里已删除的文件，库里的孤儿块一并清掉
            pruned: list[str] = []
            if prune:
                # 先收集待清理文件，避免遍历字典时修改字典。
                pruned = [source for source in source_ids if source not in on_disk]
                for fname in pruned:
                    _delete_source(vs, fname, source_ids, source_hashes)

            # 映射来自文档及 ID 关联，下次加载时重建，无需额外保存 JSON。
            with trace_span("rag.vectorstore.save"):
                vs.save_local(settings.faiss_path)

            return IngestReport(
                added_files=tuple(added),
                updated_files=tuple(updated),
                skipped_files=tuple(skipped),
                pruned_files=tuple(pruned),
                added_chunks=added_chunks,
                total_chunks=vs.index.ntotal,
            )
        except Exception:
            node_span.metadata["status"] = "error"
            raise

def _load_pdf_as_documents(path: Path) -> list[Document]:
    """版面感知加载 PDF（doc-intelligence L01）。

    把 doc_parser.parse_pdf 产出的 Element 流转成 LangChain Document：
        text 元素  → 每个 block 一个 Document（content=文本）
        table 元素 → 整表一个 Document（content 暂存串行文本，L02 升级结构化）
        image 元素 → content 为空时跳过（L03 OCR / L04 描述才入）
    所有 Document 带 page/element_type/bbox metadata（L06 引用溯源用），
    以及和 md/txt 一致的 source/src_hash/chunk_idx（增量缓存/id 用）。
    """
    # 延迟导入：enable_multimodal_ingest=off 时不需要 PyMuPDF，保持现状依赖干净
    from .doc_parser import parse_pdf

    elements = parse_pdf(path)
    fhash = file_md5(path)
    docs: list[Document] = []
    chunk_idx = 0
    for el in elements:
        # 图片元素内容为空时（OCR/描述未启用）不入库，避免空 chunk 污染向量库
        if el.type == "image" and not el.content:
            continue
        docs.append(
            Document(
                page_content=el.content,
                metadata={
                    "source": path.name,
                    "section": f"P{el.page}·{el.type}",  # 引用溯源展示「P3·table」
                    "src_hash": fhash,
                    "chunk_idx": chunk_idx,
                    # 多模态字段（L06 引用溯源用；旧检索/生成不读这些，向后兼容）
                    "page": el.page,
                    "element_type": el.type,
                    "bbox": ",".join(f"{v:.1f}" for v in el.bbox),
                },
            )
        )
        chunk_idx += 1
    return docs