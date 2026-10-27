"""RAGAS 规划评测：历史规划输出 → 预期标注比对 → 三项规则评分。"""

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
from prompts.crypto_analysis_plan import AnalysisPlan

METRICS = ("target_asset_correct", "skill_selection_correct", "refresh_correct")


class ExpectedPlan(AnalysisPlan):
    """标注三项规划字段并记录判断依据，与历史输出分别保存。"""

    label_reason: str = Field(min_length=1)


class PlanningCase(BaseModel):
    """保留用户消息、技能目录、采集时间与历史规划，不保存系统提示词。"""

    model_config = ConfigDict(extra="forbid")
    node: Literal["plan"]
    ts: datetime
    input: list[dict[str, Any]] = Field(min_length=1, max_length=1)
    actual_output: AnalysisPlan
    expected_output: ExpectedPlan
    source_line: int = Field(ge=1)


def load_cases(path: Path) -> list[PlanningCase]:
    """读取带预期标注的历史案例，缺少标注或字段不合法时直接报错。"""
    cases = [
        PlanningCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("没有规划评测案例")
    return cases


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


def build_samples(cases: list[PlanningCase]) -> list[dict[str, Any]]:
    """使用历史规划构建评分样本，技能范围按案例中的目录校验。"""
    samples = []
    for case in cases:
        messages = messages_from_dict(case.input)
        if [message.type for message in messages] != ["human"]:
            raise ValueError(f"第 {case.source_line} 行必须只保留一条 human 消息")
        content = messages[0].content
        if not isinstance(content, str):
            raise ValueError(f"第 {case.source_line} 行的用户消息必须为文本")
        catalog, separator, question = content.partition("用户问题：")
        if not separator or not question.strip():
            raise ValueError(f"第 {case.source_line} 行缺少用户问题")
        available_skills = {
            line.strip()[2:].partition(":")[0]
            for line in catalog.splitlines()
            if line.strip().startswith("- ")
        }
        expected = case.expected_output
        if (
            not expected.skill_names
            or expected.skill_names != sorted(set(expected.skill_names))
            or not set(expected.skill_names) <= available_skills
        ):
            raise ValueError(f"第 {case.source_line} 行的预期技能必须来自目录并排序去重")
        samples.append(
            {
                "user_input": question.strip(),
                "response": case.actual_output.model_dump(),
                "reference": expected.model_dump(exclude={"label_reason"}),
                "metadata": {
                    "source_line": case.source_line,
                    "ts": case.ts.isoformat(),
                    "label_reason": expected.label_reason,
                },
            }
        )
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


def run_eval(cases: list[PlanningCase]) -> dict[str, Any]:
    """对历史规划执行三项规则评分，重复记录逐条计分。"""
    if not cases:
        raise ValueError("没有规划评测案例")
    started = time.perf_counter()
    print(f"读取 {len(cases)} 条历史规划输出...", flush=True)
    samples = build_samples(cases)
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
