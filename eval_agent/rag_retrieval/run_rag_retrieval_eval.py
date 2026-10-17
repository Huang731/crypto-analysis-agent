"""固定查询直接调用生产检索器，按原文标注评分并保存 Langfuse 实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import truststore
from langchain_core.documents import Document
from langfuse import Evaluation, Langfuse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config.config import PROJECT_ROOT, settings
from app.crypto_analysis.metric_service import _log_reason
from app.crypto_analysis.skill_loader import get_skill_loader
from app.rag.ingest import get_vectorstore
from app.rag.retriever import KBRetriever, load_corpus
from eval_agent.planning.run_planning_eval import CaseMetadata, EvalSettings
from eval_agent.skill_execution.run_skill_execution_eval import duration_summary
from prompts.crypto_analysis_skill_content import thought_retrieval_prompt


class RetrievalInput(BaseModel):
    """固定检索查询；Skill 案例保留生产模板所需的参数。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1)
    skill_name: str | None = None
    target_asset: str | None = None

    @model_validator(mode="after")
    def complete_skill_scope(self) -> RetrievalInput:
        if (self.skill_name is None) != (self.target_asset is None):
            raise ValueError("Skill 查询必须同时提供技能名和分析对象")
        return self


class RelevantChunk(BaseModel):
    """阅读索引原文后标注的相关切片，不从检索输出反推标签。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    section: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=10)


class RetrievalExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_status: Literal["labeled", "knowledge_gap"]
    relevant_chunks: list[RelevantChunk]

    @model_validator(mode="after")
    def valid_references(self) -> RetrievalExpected:
        if (self.reference_status == "labeled") != bool(self.relevant_chunks):
            raise ValueError("有依据案例必须标注相关切片；知识缺口案例不标注相关切片")
        ids = [chunk.document_id for chunk in self.relevant_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("预期相关切片重复")
        return self


class RetrievalMetadata(CaseMetadata):
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: RetrievalInput
    expected_output: RetrievalExpected
    metadata: RetrievalMetadata


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str | None
    content: str
    metadata: dict[str, Any]


class RetrievalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_status: Literal["success", "error"]
    documents: list[RetrievedChunk] = Field(default_factory=list)
    duration_ms: float = Field(ge=0)
    source_metadata_valid: bool | None = None
    error_type: str | None = None
    error_message: str | None = None


def corpus_fingerprint(documents: list[Document]) -> str:
    """对实际索引的 ID、正文与元数据取哈希，索引变化后要求重新检查标签。"""
    payload = [
        {"id": doc.id, "content": doc.page_content, "metadata": doc.metadata}
        for doc in sorted(documents, key=lambda item: item.id or "")
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_cases(path: Path, corpus: list[Document]) -> list[RetrievalCase]:
    """离线校验索引快照、原文引用、固定查询及案例唯一性。"""
    checksum = corpus_fingerprint(corpus)
    documents = {doc.id: doc for doc in corpus}
    if not corpus or None in documents or len(documents) != len(corpus):
        raise ValueError("知识库为空或包含缺失、重复的切片 ID")
    loader = get_skill_loader()
    cases: list[RetrievalCase] = []
    identifiers: set[str] = set()
    queries: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = RetrievalCase.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"{path}:{number} 案例格式错误："
                f"{exc.errors(include_input=False, include_url=False)}"
            ) from None
        if case.metadata.case_id in identifiers or case.input.query in queries:
            raise ValueError(f"{path}:{number} 案例标识或查询重复")
        if case.metadata.corpus_sha256 != checksum:
            raise ValueError(f"{path}:{number} 知识库已改变，请重新核对原文标注")
        for expected in case.expected_output.relevant_chunks:
            doc = documents.get(expected.document_id)
            if doc is None or (
                doc.metadata.get("source") != expected.source
                or doc.metadata.get("section") != expected.section
                or expected.evidence_quote not in doc.page_content
            ):
                raise ValueError(f"{path}:{number} 标注的切片、来源、章节或原文不匹配")
        if case.input.skill_name is not None:
            skill = loader.load_definition(case.input.skill_name)
            query = thought_retrieval_prompt(
                case.input.target_asset,
                skill.scoring.name,
                [metric.label for metric in skill.scoring.metrics],
            )
            if query != case.input.query:
                raise ValueError(f"{path}:{number} Skill 查询与当前生产模板不一致")
        identifiers.add(case.metadata.case_id)
        queries.add(case.input.query)
        cases.append(case)
    if not cases:
        raise ValueError("评测案例不能为空")
    return cases


async def evaluate_retrieval(
    retriever: KBRetriever,
    client: Langfuse,
    question: RetrievalInput,
    corpus: dict[str, Document],
) -> RetrievalOutput:
    """直接执行生产入口，保留错误；不更换检索模式、不补充材料。"""
    started = time.perf_counter()
    with client.start_as_current_observation(
        as_type="span", name="rag_retrieval.invoke", input=question.model_dump()
    ) as observation:
        try:
            documents = await retriever.aretrieve(question.query)
            ids = [doc.id for doc in documents]
            valid = len(ids) == len(set(ids)) and len(ids) <= settings.final_k
            for doc in documents:
                original = corpus.get(doc.id)
                valid = valid and (
                    original is not None
                    and doc.page_content == original.page_content
                    and all(
                        doc.metadata.get(key) == value
                        for key, value in original.metadata.items()
                    )
                    and bool(doc.metadata.get("source"))
                    and bool(doc.metadata.get("section"))
                )
            output = RetrievalOutput(
                execution_status="success",
                documents=[
                    RetrievedChunk(
                        document_id=doc.id,
                        content=doc.page_content,
                        metadata=doc.metadata,
                    )
                    for doc in documents
                ],
                source_metadata_valid=valid if documents else None,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 保留单项真实失败，继续后续案例。
            output = RetrievalOutput(
                execution_status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
                error_message=_log_reason(str(exc))[:400],
            )
        observation.update(
            output=output.model_dump(mode="json"),
            level="DEFAULT" if output.execution_status == "success" else "ERROR",
        )
    return output


def compare_retrieval(
    actual: RetrievalOutput, expected: RetrievalExpected, top_k: int
) -> dict[str, Any]:
    """只对已标注正例计算命中、排名和召回；未标注材料不直接判为无关。"""
    ids = [doc.document_id for doc in actual.documents[:top_k]]
    gold = {chunk.document_id for chunk in expected.relevant_chunks}
    ranks = [rank for rank, doc_id in enumerate(ids, 1) if doc_id in gold]
    labeled = expected.reference_status == "labeled"
    success = actual.execution_status == "success"
    hit = bool(ranks) and success
    return {
        "execution_success": success,
        "source_metadata_valid": actual.source_metadata_valid,
        "hit_at_k": int(hit) if labeled else None,
        "mrr_at_k": (1 / ranks[0] if hit else 0.0) if labeled else None,
        "annotated_recall_at_k": (len(set(ids) & gold) / len(gold) if success else 0.0)
        if labeled
        else None,
        "matched_document_ids": [doc_id for doc_id in ids if doc_id in gold],
        "missing_annotated_document_ids": sorted(gold - set(ids)),
        "unannotated_document_ids": [doc_id for doc_id in ids if doc_id not in gold],
        "knowledge_gap_returned_count": len(ids) if not labeled and success else None,
    }


def score_retrieval(
    *, output: Any, expected_output: Any, **kwargs: Any
) -> list[Evaluation]:
    """评分由固定标注与代码确定，不调用额外的裁判模型。"""
    comparison = compare_retrieval(
        RetrievalOutput.model_validate(output),
        RetrievalExpected.model_validate(expected_output),
        settings.final_k,
    )
    comment = json.dumps(comparison, ensure_ascii=False)
    return [
        Evaluation(name=name, value=float(comparison[name]), comment=comment)
        for name in (
            "execution_success",
            "source_metadata_valid",
            "hit_at_k",
            "mrr_at_k",
            "annotated_recall_at_k",
            "knowledge_gap_returned_count",
        )
        if comparison[name] is not None
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """知识缺口与执行失败单独计数，失败正例仍进入命中率分母。"""
    labeled = [row for row in rows if row["comparison"]["hit_at_k"] is not None]
    gaps = [row for row in rows if row["comparison"]["hit_at_k"] is None]
    metadata_scores = [
        row["comparison"]["source_metadata_valid"]
        for row in rows
        if row["comparison"]["source_metadata_valid"] is not None
    ]
    return {
        "completed_cases": len(rows),
        "labeled_cases": len(labeled),
        "knowledge_gap_cases": len(gaps),
        "execution_errors": sum(
            row["output"]["execution_status"] == "error" for row in rows
        ),
        "mean_scores": {
            name: sum(row["comparison"][name] for row in labeled) / len(labeled)
            if labeled
            else None
            for name in ("hit_at_k", "mrr_at_k", "annotated_recall_at_k")
        },
        "source_metadata_accuracy": sum(metadata_scores) / len(metadata_scores)
        if metadata_scores
        else None,
        "hit_rate_by_category": {
            category: sum(
                row["comparison"]["hit_at_k"]
                for row in labeled
                if row["metadata"]["category"] == category
            )
            / sum(row["metadata"]["category"] == category for row in labeled)
            for category in sorted({row["metadata"]["category"] for row in labeled})
        },
        "knowledge_gap_cases_returning_documents": sum(
            bool(row["comparison"]["knowledge_gap_returned_count"]) for row in gaps
        ),
        "duration": duration_summary([row["output"]["duration_ms"] for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="固定案例的生产 RAG 检索评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("rag_retrieval_cases.jsonl"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--system-certificates", action="store_true")
    parser.add_argument("--dataset", default="crypto-agent-rag-retrieval")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rag_retrieval_eval")
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("案例数量必须大于零")
    if settings.retrieve_k < 1 or settings.final_k < 1:
        parser.error("检索与重排数量必须大于零")
    corpus = load_corpus(get_vectorstore())
    cases = load_cases(args.cases, corpus)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.validate_only:
        print(
            f"已校验 {len(cases)} 例、{len(corpus)} 个切片："
            f"{dict(Counter(case.expected_output.reference_status for case in cases))}"
        )
        return
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "rag-%Y%m%dT%H%M%SZ"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    output_path = args.output_dir / f"{run_name}.json"
    if output_path.exists():
        parser.error(f"结果文件已存在：{output_path}")
    if args.system_certificates:
        truststore.inject_into_ssl()
    config = EvalSettings()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
        timeout=30,
    )
    retriever = KBRetriever()
    corpus_by_id = {doc.id: doc for doc in corpus}
    metadata = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "evaluator_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "corpus_sha256": corpus_fingerprint(corpus),
        "retriever_code_sha256": hashlib.sha256(
            (PROJECT_ROOT / "app/rag/retriever.py").read_bytes()
        ).hexdigest(),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "rerank_model": settings.rerank_model,
        "retrieve_k": settings.retrieve_k,
        "final_k": settings.final_k,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "enable_multi_query": settings.enable_multi_query,
        "multi_query_model": settings.bailian_fast_model
        if settings.enable_multi_query
        else None,
        "model_timeout_seconds": 120.0,
        "max_attempts": 2,
        "annotation_scope": "non_exhaustive_positive_chunks",
        "metric_boundary": "召回率仅针对已标注切片；知识缺口不计算命中率，不评测拒答能力",
    }
    rows: list[dict[str, Any]] = []
    run_status = "failed"

    async def task(*, item: Any) -> dict[str, Any]:
        question = RetrievalInput.model_validate(item.input)
        try:
            output = await evaluate_retrieval(retriever, client, question, corpus_by_id)
        finally:
            await retriever.aclose()
        comparison = compare_retrieval(
            output,
            RetrievalExpected.model_validate(item.expected_output),
            settings.final_k,
        )
        rows.append(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
                "output": output.model_dump(mode="json"),
                "comparison": comparison,
            }
        )
        print(
            f"{item.metadata['case_id']}：{output.execution_status}，"
            f"hit@{settings.final_k}={comparison['hit_at_k']}，"
            f"{output.duration_ms:.0f} ms",
            flush=True,
        )
        return output.model_dump(mode="json")

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败")
        client.create_dataset(
            name=args.dataset,
            description="原文人工标注的固定查询检索评测；知识缺口单独记录",
            input_schema=RetrievalInput.model_json_schema(),
            expected_output_schema=RetrievalExpected.model_json_schema(),
        )
        items = [
            client.create_dataset_item(
                dataset_name=args.dataset,
                # 标注更新后创建独立条目，保留历史实验所引用的预期材料。
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{args.dataset}:{case.metadata.case_id}:"
                        f"{metadata['cases_sha256']}",
                    )
                ),
                input=case.input.model_dump(),
                expected_output=case.expected_output.model_dump(),
                metadata=case.metadata.model_dump(),
            )
            for case in cases
        ]
        experiment = client.run_experiment(
            name=args.dataset,
            run_name=run_name,
            data=items,
            task=task,
            evaluators=[score_retrieval],
            max_concurrency=1,
            metadata=metadata,
        )
        for row, result in zip(rows, experiment.item_results, strict=True):
            row["scores"] = [score.__dict__ for score in result.evaluations]
            row["langfuse_trace_id"] = result.trace_id
        if len(rows) != len(cases):
            raise RuntimeError("评测记录不完整")
        run_status = "completed"
        print(experiment.format())
    finally:
        if rows:
            report = {
                "run_name": run_name,
                "run_status": run_status,
                "planned_cases": len(cases),
                "metadata": metadata,
                "summary": summarize(rows),
                "cases": rows,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            print(f"本地结果：{output_path.resolve()}")
        client.shutdown()


if __name__ == "__main__":
    main()
