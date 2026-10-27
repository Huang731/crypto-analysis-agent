"""对同一问题审核前后的报告评分，比较忠实度与答案相关性变化。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import PROJECT_ROOT, settings
from eval_agent.writer.run_eval import (
    WriterCase,
    WriterInput,
)
from eval_agent.writer.run_eval import (
    build_evidence as build_writer_evidence,
)
from eval_agent.writer.run_eval import (
    evaluate as evaluate_report,
)

METRICS = ("faithfulness", "answer_relevancy")


class ReviewerCase(BaseModel):
    """两份报告共用同一份 Writer 输入，确保问题与事实证据一致。"""

    model_config = ConfigDict(extra="forbid")
    input: WriterInput
    before_report: str = Field(min_length=1)
    after_report: str = Field(min_length=1)
    source_line: int = Field(ge=1)


def as_writer_case(case: ReviewerCase, report: str) -> WriterCase:
    """复用 Writer 的证据提取和评分口径，不重新生成或审核报告。"""
    return WriterCase(
        node="writer",
        input=case.input,
        actual_output=report,
        source_line=case.source_line,
    )


def load_cases(path: Path) -> list[ReviewerCase]:
    """读取同一问题的报告对，输入沿用 Writer 案例中的消息格式。"""
    cases = [
        ReviewerCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("审核评测案例文件为空，请补充同一问题的审核前后报告")
    for case in cases:
        if not case.before_report.strip() or not case.after_report.strip():
            raise ValueError(f"第 {case.source_line} 行的审核前后报告必须为非空正文")
        build_writer_evidence(as_writer_case(case, case.before_report))
    return cases


async def evaluate(case: ReviewerCase, judge_model: list[str]) -> dict[str, Any]:
    """用同一裁判配置分别评分报告正文，仅对有效配对计算变化。"""
    before = await evaluate_report(
        as_writer_case(case, case.before_report), judge_model
    )
    after = await evaluate_report(as_writer_case(case, case.after_report), judge_model)
    delta = {
        name: (
            after[name] - before[name]
            if before[name] is not None and after[name] is not None
            else None
        )
        for name in METRICS
    }
    return {
        "ragas_input": before["ragas_input"],
        "before": {name: before[name] for name in (*METRICS, "errors")},
        "after": {name: after[name] for name in (*METRICS, "errors")},
        "delta": delta,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """均分使用相同的有效配对，避免单侧失败造成虚假的提升。"""
    summary: dict[str, Any] = {"cases": len(rows)}
    for name in METRICS:
        pairs = [
            (
                row["output"]["before"][name],
                row["output"]["after"][name],
            )
            for row in rows
            if row["output"]["before"][name] is not None
            and row["output"]["after"][name] is not None
        ]
        deltas = [after - before for before, after in pairs]
        count = len(pairs)
        summary[name] = {
            "paired_cases": count,
            "before_mean": sum(before for before, _ in pairs) / count
            if count
            else None,
            "after_mean": sum(after for _, after in pairs) / count if count else None,
            "mean_delta": sum(deltas) / count if count else None,
        }
    return summary


async def score_samples(cases: list[ReviewerCase]) -> list[dict[str, Any]]:
    """沿用 Writer 的三案例并发，每例分别评审核前与修改后的正文。"""
    semaphore = asyncio.Semaphore(3)

    async def one(case: ReviewerCase) -> dict[str, Any]:
        async with semaphore:
            output = await evaluate(case, settings.qwen3_max_model)
        print(
            f"案例第 {case.source_line} 行：审核前={output['before']}，"
            f"审核后={output['after']}，变化={output['delta']}",
            flush=True,
        )
        return {**case.model_dump(mode="json"), "output": output}

    return list(await asyncio.gather(*(one(case) for case in cases)))


def run_eval(cases: list[ReviewerCase]) -> dict[str, Any]:
    """对已保存的报告对评分，输出两项指标前后均分与平均变化。"""
    if not cases:
        raise ValueError("没有审核前后报告配对案例")
    print(f"开始 {len(cases)} 条审核前后报告对比评测...", flush=True)
    rows = asyncio.run(score_samples(cases))
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "cases": rows}


def main() -> None:
    cases_path = Path(__file__).with_name("reviewer_cases.jsonl")
    output_path = PROJECT_ROOT / "eval_agent/outputs/reviewer_eval/results.json"
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"

    cases = load_cases(cases_path)
    result = run_eval(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"结果已写入 {output_path}")


if __name__ == "__main__":
    main()
