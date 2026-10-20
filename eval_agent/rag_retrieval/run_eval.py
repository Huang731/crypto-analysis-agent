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
from typing import Any, TypeVar

import httpx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.config import PROJECT_ROOT, _env_files, settings
from app.llm.model import (
    ainvoke_structured,
    get_embeddings,
    make_glm_5_3_flash_llm,
    model_session,
)
from app.rag.generate import stream_answer
from app.rag.retriever import KBRetriever

GOLDEN_PATH = Path(__file__).with_name("golden_set.jsonl")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_CONCURRENCY = 4
_Schema = TypeVar("_Schema", bound=BaseModel)


class EvalSettings(BaseSettings):
    """RAG 评测独立读取 Langfuse 环境配置。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(), env_file_encoding="utf-8", extra="ignore"
    )
    langfuse_public_key: str = Field(min_length=1)
    langfuse_secret_key: str = Field(min_length=1)
    langfuse_base_url: str = "https://cloud.langfuse.com"


def create_metrics(judge_model: str, *, http: httpx.AsyncClient) -> dict[str, Any]:
    """独立构建 RAG 四项指标，共用 GLM 裁判和项目向量模型。"""
    if judge_model != settings.glm_5_3_flash_model:
        raise ValueError(f"未配置的 RAG 裁判模型：{judge_model}")
    model = (
        make_glm_5_3_flash_llm(async_client=http)
        .bind(reasoning_effort="low")
        .with_config(callbacks=[CallbackHandler()])
    )

    class RAGJudge(InstructorBaseRagasLLM):
        """将 GLM 的异步结构化调用接入 RAGAS。"""

        def generate(self, prompt: str, response_model: type[_Schema]) -> _Schema:
            raise NotImplementedError("评测只使用异步 agenerate")

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def agenerate(
            self, prompt: str, response_model: type[_Schema]
        ) -> _Schema:
            result = await ainvoke_structured(
                model,
                [HumanMessage(content=prompt)],
                schema=response_model,
                timeout=120,
            )
            return response_model.model_validate(result)

    embeddings = get_embeddings()

    class RAGEmbeddings(BaseRagasEmbedding):
        """保留 RAG 相关性评分所需的向量接口、超时与重试。"""

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embeddings.embed_query(text)

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.wait_for(embeddings.aembed_query(text), timeout=120)

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def aembed_texts(
            self, texts: list[str], **kwargs: Any
        ) -> list[list[float]]:
            return await asyncio.wait_for(
                embeddings.aembed_documents(texts), timeout=120
            )

    judge = RAGJudge()
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge, embeddings=RAGEmbeddings(), strictness=1
        ),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


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
    # 先注册项目客户端，评分模型的 Langfuse 回调才能复用 .env 中的配置。
    config = EvalSettings()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    try:
        async with model_session() as http:
            metrics = create_metrics(settings.glm_5_3_flash_model, http=http)
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
    finally:
        client.flush()


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
        "--out", default=str(PROJECT_ROOT / "eval_agent/outputs/rag_eval/results.json")
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
