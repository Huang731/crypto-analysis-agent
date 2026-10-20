"""文档加载与结构感知分块。

Markdown 先按标题保留章节信息，再按长度切分；纯文本直接按长度切分。
每个切片带有文件名、章节路径、内容 hash 和块序号，用于引用与增量入库。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings
from app.core.tracing import trace_span
from app.rag.cleaner import CleanProcessor

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def file_md5(path: Path) -> str:
    """计算文件内容 MD5，用于判断文件是否发生变化。"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_documents(docs_dir: str | Path) -> list[Path]:
    """扫描目录下所有受支持的文档，按文件名排序保证幂等。"""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"文档目录不存在：{root}")
    return sorted(p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def _breadcrumb(meta: dict[str, str]) -> str:
    """把标题层级拼成面包屑，如「员工手册 > 考勤管理 > 迟到处理」。"""
    parts = [meta.get(k) for k in ("h1", "h2", "h3")]
    return " > ".join(p for p in parts if p)


def _text_splitter() -> RecursiveCharacterTextSplitter:
    """统一 Markdown 与纯文本的切片长度、重叠和中文分隔符配置。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )


def split_markdown(text: str) -> list[Document]:
    """结构感知分块：先按标题切节，超长节再按字符二次切分。"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # 保留标题在正文里，embedding 能吃到章节语义
    )
    sections = header_splitter.split_text(text)

    return _text_splitter().split_documents(sections)


def split_plain_text(text: str) -> list[Document]:
    """无结构文本（.txt）退化为纯字符分块。"""
    return _text_splitter().create_documents([text])


def load_and_split(path: Path, *, src_hash: str | None = None) -> list[Document]:
    """加载单个文档并分块，为每个 chunk 打上溯源 metadata。

    metadata 约定（检索层/生成层依赖）：
        source    —— 文件名，引用溯源展示用
        section   —— 标题面包屑，精确到条款的出处
        src_hash  —— 文件内容 MD5，增量缓存判断依据
        chunk_idx —— 块序号，稳定 id 生成用

    入库流程可传入已计算的 src_hash，避免重复读取文件计算 hash。
    """
    with trace_span("rag.document.load_split", source=path.name) as node_span:
        try:
            text = path.read_text(encoding="utf-8")
            # 在结构感知分块前清洗，章节与切片均基于同一份正文。
            text = CleanProcessor.clean(text)
            if path.suffix.lower() == ".md":
                chunks = split_markdown(text)
            else:
                chunks = split_plain_text(text)

            fhash = file_md5(path) if src_hash is None else src_hash
            return [
                Document(
                    page_content=c.page_content,
                    metadata={
                        "source": path.name,
                        "section": _breadcrumb(c.metadata),
                        "src_hash": fhash,
                        "chunk_idx": f"{fhash}:{i}",
                    },
                )
                for i, c in enumerate(chunks)
            ]
        except Exception:
            node_span.metadata["status"] = "error"
            raise
