"""RAGAS 路由评测：历史路由输出 → 人工预期比对 → 结果落盘。"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import messages_from_dict
from pydantic import BaseModel, ConfigDict, Field
from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.metrics import numeric_metric

from app.core.config import PROJECT_ROOT

Route = Literal["crypto_analysis_agent", "topic_research_agent"]


class ExpectedRoute(BaseModel):
    """独立标注预期路由和判断依据，不从历史输出推导答案。"""

    model_config = ConfigDict(extra="forbid")
    route: Route
    label_reason: str = Field(min_length=1)


class RoutingCase(BaseModel):
    """保留真实路由消息、历史输出和采集时间，并记录来源行号。"""

    model_config = ConfigDict(extra="forbid")
    node: Literal["router"]
    ts: datetime
    input: list[dict[str, Any]] = Field(min_length=2, max_length=2)
    actual_output: Route
    expected_output: ExpectedRoute
    source_line: int = Field(ge=1)


def load_cases(path: Path) -> list[RoutingCase]:
    """读取带人工标注的历史记录，缺少预期路由时直接报告错误。"""
    cases = [
        RoutingCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("没有路由评测案例")
    return cases


@numeric_metric(name="route_correct", allowed_values=(0, 1))
def route_correct(actual: str, expected: str) -> int:
    """实际路由与人工标注完全一致时记一分。"""
    return int(actual == expected)


def build_samples(cases: list[RoutingCase]) -> list[dict[str, Any]]:
    """直接使用已保存的路由输出，不重新调用模型或改写原始输入。"""
    samples = []
    for case in cases:
        messages = messages_from_dict(case.input)
        if [message.type for message in messages] != ["system", "human"]:
            raise ValueError(f"第 {case.source_line} 行必须保留一条 system 和 human 消息")
        question = messages[1].content
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"第 {case.source_line} 行的用户问题必须为非空文本")
        samples.append(
            {
                "user_input": question,
                "response": case.actual_output,
                "reference": case.expected_output.route,
                "metadata": {
                    "source_line": case.source_line,
                    "ts": case.ts.isoformat(),
                    "label_reason": case.expected_output.label_reason,
                },
            }
        )
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


def run_eval(cases: list[RoutingCase]) -> dict[str, Any]:
    """对历史路由执行规则评测并汇总平均正确率，重复记录逐条计分。"""
    if not cases:
        raise ValueError("没有路由评测案例")
    started = time.perf_counter()
    print(f"读取 {len(cases)} 条历史路由输出...", flush=True)
    samples = build_samples(cases)
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

    cases = load_cases(cases_path)
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
