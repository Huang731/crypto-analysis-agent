"""RAGAS 路由评测：固定问题 → 生产路由 → 路由正确率 → 结果落盘。"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.metrics import numeric_metric

from app.core.config import PROJECT_ROOT
from app.llm.model import make_glm_4_flash_llm, model_session
from app.routing.graph import _invoke_route_decision
from prompts.router import routing_messages


@numeric_metric(name="route_correct", allowed_values=(0, 1))
def route_correct(actual: str, expected: str) -> int:
    """实际路由与人工标注完全一致时记一分。"""
    return int(actual == expected)


async def build_samples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """直接调用生产路由函数，沿用其名称校验、重试和超时。"""
    samples = []
    async with model_session() as http:
        llm = make_glm_4_flash_llm(async_client=http)
        for case in cases:
            question = case["input"]["resolved_question"]
            actual = await _invoke_route_decision(llm, routing_messages(question))
            samples.append(
                {
                    "user_input": question,
                    "response": actual,
                    "reference": case["expected_output"]["route"],
                    "metadata": case["metadata"],
                }
            )
            print(f"已生成路由：{question} → {actual}", flush=True)
    return samples


async def score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """通过 RAGAS 实验执行路由规则评分，不调用裁判模型。"""

    @experiment()
    async def run_evaluation(sample: dict[str, Any]) -> dict[str, Any]:
        score = await route_correct.ascore(
            actual=sample["response"],
            expected=sample["reference"],
        )
        if score.value is None:
            raise ValueError(f"route_correct 评分失败：{score.reason}")
        return {**sample, "route_correct": score.value}

    dataset = Dataset("routing_evaluation", backend=InMemoryBackend(), data=samples)
    result = await run_evaluation.arun(dataset)
    if len(result) != len(samples):
        raise RuntimeError("存在未完成的评测案例，请检查 RAGAS 实验错误")
    return list(result)


def run_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """生成路由，执行规则评测并汇总平均正确率。"""
    if not cases:
        raise ValueError("没有路由评测案例")
    started = time.perf_counter()
    print(f"生成 {len(cases)} 题的路由...", flush=True)
    samples = asyncio.run(build_samples(cases))
    print("RAGAS 路由正确率评测中...", flush=True)
    rows = asyncio.run(score_samples(samples))
    aggregate = {
        "route_correct": round(sum(row["route_correct"] for row in rows) / len(rows), 4)
    }
    print(f"评测完成（{time.perf_counter() - started:.0f}s）：{aggregate}")
    return {"aggregate": aggregate, "per_question": rows}


def main() -> None:
    cases_path = Path(__file__).with_name("routing_cases.jsonl")
    output_path = PROJECT_ROOT / "eval_agent/outputs/routing_eval/results.json"

    cases = [
        json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()
    ]
    result = run_eval(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"n_questions": len(cases), "results": [result]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"结果已写入 {output_path}")


if __name__ == "__main__":
    main()
