"""RAGAS 评测：固定问答集 → 检索与回答 → 四指标 → 结果落盘。

在项目根目录运行：
    .venv-eval/bin/python -m eval_agent.rag_retrieval.run_eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.metrics.collections import ContextPrecision, ContextRecall

from app.config.config import PROJECT_ROOT, settings
from app.llm.model import model_session
from app.rag.generate import stream_answer
from app.rag.retriever import KBRetriever
from eval_agent.writer.run_writer_eval import create_metrics

GOLDEN_PATH = Path(__file__).with_name("golden_set.json")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_CONCURRENCY = 4


async def _answer(question: str, docs: list[Document]) -> str:
    parts = [tok async for tok in stream_answer(question, docs)]
    return "".join(parts)


async def build_samples(golden: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对固定问答集逐题执行检索和生成，产出 RAGAS 样本。"""
    kb = KBRetriever()
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            docs = await kb.aretrieve(item["question"])
            answer = await _answer(item["question"], docs)
        return {
            "user_input": item["question"],
            "retrieved_contexts": [doc.page_content for doc in docs],
            "response": answer,
            "reference": item["ground_truth"],
        }

    try:
        return list(await asyncio.gather(*(one(item) for item in golden)))
    finally:
        await kb.aclose()


async def score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """使用 RAGAS 实验入口执行四项指标评分。"""
    async with model_session() as http:
        metrics = create_metrics(settings.bailian_smart_model, http=http)
        # 保留 RAG 评测原先生成三个问题的相关性评分参数。
        metrics["answer_relevancy"].strictness = 1
        judge = metrics["faithfulness"].llm
        metrics["context_precision"] = ContextPrecision(llm=judge)
        metrics["context_recall"] = ContextRecall(llm=judge)
        sem = asyncio.Semaphore(_CONCURRENCY)

        @experiment()
        async def run_evaluation(sample: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                faithfulness = await metrics["faithfulness"].ascore(
                    user_input=sample["user_input"],
                    response=sample["response"],
                    retrieved_contexts=sample["retrieved_contexts"],
                )
                answer_relevancy = await metrics["answer_relevancy"].ascore(
                    user_input=sample["user_input"],
                    response=sample["response"],
                )
                context_precision = await metrics["context_precision"].ascore(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                )
                context_recall = await metrics["context_recall"].ascore(
                    user_input=sample["user_input"],
                    reference=sample["reference"],
                    retrieved_contexts=sample["retrieved_contexts"],
                )
                return {
                    "question": sample["user_input"],
                    "faithfulness": faithfulness.value,
                    "answer_relevancy": answer_relevancy.value,
                    "context_precision": context_precision.value,
                    "context_recall": context_recall.value,
                }

        dataset = Dataset("rag_evaluation", backend=InMemoryBackend(), data=samples)
        result = await run_evaluation.arun(dataset)
        if len(result) != len(samples):
            raise RuntimeError("存在未完成的评测案例，请检查 RAGAS 实验错误")
        return list(result)


def run_eval(golden: list[dict[str, Any]]) -> dict[str, Any]:
    """生成生产检索器的样本，执行四指标评测并汇总结果。"""
    started = time.perf_counter()
    print(f"\n生成 {len(golden)} 题的检索与回答…", flush=True)
    samples = asyncio.run(build_samples(golden))

    print("RAGAS 四指标评测中…", flush=True)
    rows = asyncio.run(score_samples(samples))

    def _valid(values: list[float | None]) -> list[float]:
        """聚合时排除裁判调用失败产生的 None 或 NaN。"""
        return [
            value for value in values if value is not None and not math.isnan(value)
        ]

    aggregate = {}
    for name in METRICS:
        values = _valid([row.get(name) for row in rows])
        aggregate[name] = round(sum(values) / len(values), 4) if values else None
    print(f"评测完成（{time.perf_counter() - started:.0f}s）：{aggregate}")
    return {
        "aggregate": aggregate,
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 四指标评测")
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "outputs/rag_eval/results.json")
    )
    args = parser.parse_args()

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"固定问答集：{len(golden)} 题")
    result = run_eval(golden)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"n_questions": len(golden), "results": [result]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {args.out}")

    print(f"\n{'指标':<20}{'分数':>10}")
    for name in METRICS:
        print(f"{name:<20}{result['aggregate'][name]!s:>10}")


if __name__ == "__main__":
    main()
