"""RAGAS 规划评测：固定问题 → 生产规划 → 三项规则指标 → 结果落盘。"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.metrics import numeric_metric

from app.core.config import PROJECT_ROOT
from app.crypto_analysis.crypto_analysis_parent_agent import create_analysis_plan
from app.crypto_analysis.skill_loader import get_skill_loader
from app.llm.model import make_glm_4_5_air_llm, model_session
from prompts.crypto_analysis_plan import AnalysisPlan

METRICS = ("target_asset_correct", "skill_selection_correct", "refresh_correct")


@numeric_metric(name="target_asset_correct", allowed_values=(0, 1))
def target_asset_correct(actual: str, expected: str) -> int:
    """目标资产与人工标注完全一致时记一分。"""
    return int(actual == expected)


@numeric_metric(name="skill_selection_correct", allowed_values=(0, 1))
def skill_selection_correct(actual: list[str], expected: list[str]) -> int:
    """技能列表去重后逐项比较，漏选、多选或顺序错误均记零分。"""
    return int(list(dict.fromkeys(actual)) == list(dict.fromkeys(expected)))


@numeric_metric(name="refresh_correct", allowed_values=(0, 1))
def refresh_correct(actual: bool, expected: bool) -> int:
    """刷新标志与人工标注一致时记一分。"""
    return int(actual == expected)


async def build_samples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """直接调用生产规划函数，沿用其结构化输出、重试和超时。"""
    skill_descriptions = get_skill_loader().format_skill_descriptions()
    samples = []
    async with model_session() as http:
        llm = make_glm_4_5_air_llm(async_client=http)
        for case in cases:
            question = case["input"]["resolved_question"]
            expected = AnalysisPlan.model_validate(case["expected_output"])
            actual = await create_analysis_plan(llm, question, skill_descriptions)
            samples.append(
                {
                    "user_input": question,
                    "response": AnalysisPlan.model_validate(actual).model_dump(),
                    "reference": expected.model_dump(),
                    "metadata": case["metadata"],
                }
            )
            print(f"已生成规划：{question}", flush=True)
    return samples


async def score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """通过 RAGAS 实验执行三个自定义规则指标，不调用裁判模型。"""

    @experiment()
    async def run_evaluation(sample: dict[str, Any]) -> dict[str, Any]:
        row = dict(sample)
        for metric, field in (
            (target_asset_correct, "target_asset"),
            (skill_selection_correct, "skill_names"),
            (refresh_correct, "refresh"),
        ):
            score = await metric.ascore(
                actual=sample["response"][field],
                expected=sample["reference"][field],
            )
            if score.value is None:
                raise ValueError(f"{metric.name} 评分失败：{score.reason}")
            row[metric.name] = score.value
        return row

    dataset = Dataset("planning_evaluation", backend=InMemoryBackend(), data=samples)
    result = await run_evaluation.arun(dataset)
    if len(result) != len(samples):
        raise RuntimeError("存在未完成的评测案例，请检查 RAGAS 实验错误")
    return list(result)


def run_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """生成规划，执行三项指标评测并汇总平均正确率。"""
    if not cases:
        raise ValueError("没有规划评测案例")
    started = time.perf_counter()
    print(f"生成 {len(cases)} 题的规划...", flush=True)
    samples = asyncio.run(build_samples(cases))
    print("RAGAS 三项规则指标评测中...", flush=True)
    rows = asyncio.run(score_samples(samples))
    aggregate = {
        name: round(sum(row[name] for row in rows) / len(rows), 4) for name in METRICS
    }
    print(f"评测完成（{time.perf_counter() - started:.0f}s）：{aggregate}")
    return {"aggregate": aggregate, "per_question": rows}


def main() -> None:
    cases_path = Path(__file__).with_name("planning_cases.jsonl")
    output_path = PROJECT_ROOT / "eval_agent/outputs/planning_eval/results.json"

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
