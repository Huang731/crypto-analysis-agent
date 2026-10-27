"""RAGAS 指标节点评测：历史工具输出 → 预期契约比对 → 四项规则评分。"""

import asyncio
import json
import time
from collections import Counter
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import messages_from_dict
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend
from ragas.metrics import numeric_metric

from app.core.config import PROJECT_ROOT
from app.crypto_analysis.skill_loader import LoadedSkill

METRICS = (
    "tool_selection_correct",
    "tool_arguments_correct",
    "evidence_valid",
    "indicator_results_valid",
)

METRIC_DATA_KINDS = {
    "Fear_Greed": "fear_greed",
    "Trend": "trend",
    "OI": "derivatives",
    "Funding": "derivatives",
    "Fed_Rate_Change_3M": "fed_target_range",
    "DXY_Trend": "dxy",
    "Stable_MCap": "stablecoin",
    "M2_YoY": "m2",
    "Fed_Balance": "walcl",
    "ETF_Flows": "etf",
}


class MetricSnapshot(BaseModel):
    """保留历史消息、证据及指标结果，具体输出合法性由评分规则检查。"""

    model_config = ConfigDict(extra="forbid")
    messages: list[dict[str, Any]]
    evidence: dict[str, dict[str, Any]]
    indicator_results: dict[str, dict[str, Any]]


class MetricInput(MetricSnapshot):
    """节点输入使用采集时的技能定义和分析范围。"""

    skill: LoadedSkill
    scope: dict[str, str]


class ExpectedToolCall(BaseModel):
    """独立标注应调用的工具及显式参数，不包含运行时注入的上下文。"""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    args: dict[str, Any]


class ExpectedEvidence(BaseModel):
    """标注证据类型、来源动作及取数参数，不复制历史原始数据。"""

    model_config = ConfigDict(extra="forbid")
    data_kind: str = Field(min_length=1)
    source_action: str = Field(min_length=1)
    arguments: dict[str, Any]


class ExpectedIndicator(BaseModel):
    """标注成功结果的元数据契约，不将历史计算数值作为标准答案。"""

    model_config = ConfigDict(extra="forbid")
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    url: str = Field(min_length=1)


class ExpectedMetrics(BaseModel):
    """四项规则评分所需的预期契约及标注依据。"""

    model_config = ConfigDict(extra="forbid")
    tool_calls: list[ExpectedToolCall]
    evidence: dict[str, ExpectedEvidence]
    indicator_results: dict[str, ExpectedIndicator]
    label_reason: str = Field(min_length=1)


class MetricCase(BaseModel):
    """保存历史指标节点记录、独立标注和案例来源行号。"""

    model_config = ConfigDict(extra="forbid")
    node: Literal["calculate_metrics"]
    ts: datetime
    input: MetricInput
    actual_output: MetricSnapshot
    expected_output: ExpectedMetrics
    source_line: int = Field(ge=1)


def load_cases(path: Path) -> list[MetricCase]:
    """读取已标注的历史案例，缺少预期或字段不合法时直接报错。"""
    cases = [
        MetricCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("没有指标节点评测案例")
    return cases


class IndicatorResult(BaseModel):
    """生产指标结果的字段契约，同时用于人工标注预期值。"""

    model_config = ConfigDict(extra="forbid")
    skill_name: str
    indicator_id: str
    status: Literal["ok", "data_missing", "not_applicable", "calculation_error"]
    reason: str | None
    raw_values: dict[str, Any]
    calculated_values: dict[str, Any]
    unit: str | None
    observed_at: str | None
    source: str | None
    url: str | None
    score: float | None = Field(ge=0, le=10, allow_inf_nan=False, strict=True)


def values_equal(actual: Any, expected: Any) -> bool:
    """递归比较完整结果；浮点只允许 1e-6 绝对误差，不执行计算器生成答案。"""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isclose(actual, expected, rel_tol=0, abs_tol=1e-6)
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(values_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                values_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def tool_calls(output: dict[str, Any]) -> list[dict[str, Any]]:
    """仅提取本轮模型实际请求的工具名称和参数。"""
    return [
        {"name": call["name"], "args": call["args"]}
        for message in messages_from_dict(output["messages"])
        if message.type == "ai"
        for call in message.tool_calls
    ]


@numeric_metric(name="tool_selection_correct", allowed_values=(0, 1))
def tool_selection_correct(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> int:
    """工具名称和调用次数必须一致，不要求并行调用顺序一致。"""
    return int(
        Counter(call["name"] for call in actual)
        == Counter(call["name"] for call in expected)
    )


@numeric_metric(name="tool_arguments_correct", allowed_values=(0, 1))
def tool_arguments_correct(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> int:
    """按名称匹配调用，参数必须完整一致。"""
    return int(
        sorted(
            (call["name"], json.dumps(call["args"], sort_keys=True)) for call in actual
        )
        == sorted(
            (call["name"], json.dumps(call["args"], sort_keys=True))
            for call in expected
        )
    )


@numeric_metric(name="evidence_valid", allowed_values=(0, 1))
def evidence_valid(
    actual: dict[str, Any], inputs: dict[str, Any], expected: dict[str, Any]
) -> int:
    """检查证据来源、取数参数与指标关联，允许调用 ID 每次变化。"""
    try:
        evidence = actual["evidence"]
        original = inputs["evidence"]
        if not all(
            key in evidence and values_equal(evidence[key], record)
            for key, record in original.items()
        ):
            return 0
        added = {key: value for key, value in evidence.items() if key not in original}
        by_kind = {record["data_kind"]: record for record in added.values()}
        expected_added = {
            key: record
            for key, record in expected["evidence"].items()
            if key not in original
        }
        specs = {record["data_kind"]: record for record in expected_added.values()}
        if (
            len(by_kind) != len(added)
            or len(specs) != len(expected_added)
            or set(by_kind) != set(specs)
        ):
            return 0
        for kind, spec in specs.items():
            record = by_kind[kind]
            if record["source_action"] != spec["source_action"] or not values_equal(
                record["arguments"], spec["arguments"]
            ):
                return 0
            if record["status"] == "ok":
                if (
                    not isinstance(record["raw"], dict)
                    or not isinstance(record["raw"]["source"], str)
                    or not record["raw"]["source"].strip()
                ):
                    return 0
            elif record["status"] == "data_missing":
                if (
                    not isinstance(record["reason"], str)
                    or not record["reason"].strip()
                ):
                    return 0
            else:
                return 0
        pending = {
            metric["id"] for metric in inputs["skill"]["scoring"]["metrics"]
        } - inputs["indicator_results"].keys()
        metric_kinds = {
            metric_id: METRIC_DATA_KINDS[metric_id] for metric_id in pending
        }
        if set(metric_kinds.values()) != set(specs):
            return 0

        messages = messages_from_dict(actual["messages"])
        calls: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.type == "ai":
                if message.invalid_tool_calls:
                    return 0
                for call in message.tool_calls:
                    if not call["id"] or call["id"] in calls:
                        return 0
                    calls[call["id"]] = call
        responded: set[str] = set()
        linked: set[str] = set()
        recorded: set[str] = set()
        for message in messages:
            if message.type != "tool":
                continue
            call_id = message.tool_call_id
            if (
                message.status != "success"
                or call_id not in calls
                or call_id in responded
                or message.name != calls[call_id]["name"]
            ):
                return 0
            responded.add(call_id)
            summaries = json.loads(message.content)["indicators"]
            references = message.artifact["indicator_evidence_ids"]
            ids = [item["indicator_id"] for item in summaries]
            if len(ids) != len(set(ids)) or set(ids) != set(references):
                return 0
            for item in summaries:
                metric_id = item["indicator_id"]
                if metric_id in recorded or metric_id not in metric_kinds:
                    return 0
                recorded.add(metric_id)
                evidence_id = references[metric_id]
                kind = metric_kinds[metric_id]
                if (
                    evidence_id != f"{call_id}:{kind}"
                    or evidence_id not in added
                    or added[evidence_id]["data_kind"] != kind
                ):
                    return 0
                linked.add(evidence_id)
                result = actual["indicator_results"][metric_id]
                if item["status"] != result["status"]:
                    return 0
                if result["status"] != "ok" and item["reason"] != result["reason"]:
                    return 0
        return int(
            responded == set(calls) and linked == set(added) and recorded == pending
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        # 无效输出应记零分，不能被 RAGAS 当作未完成评分而跳过。
        return 0


@numeric_metric(name="indicator_results_valid", allowed_values=(0, 1))
def indicator_results_valid(
    actual: dict[str, Any], inputs: dict[str, Any], expected: dict[str, Any]
) -> int:
    """只校验结果契约与元数据，不将字段合法性当作数值计算正确性。"""
    try:
        results = actual["indicator_results"]
        metric_ids = {metric["id"] for metric in inputs["skill"]["scoring"]["metrics"]}
        if (
            set(results) != metric_ids
            or set(expected["indicator_results"]) != metric_ids
        ):
            return 0
        if not all(
            key in results and values_equal(results[key], value)
            for key, value in inputs["indicator_results"].items()
        ):
            return 0
        for metric_id, raw in results.items():
            result = IndicatorResult.model_validate(raw)
            if (
                result.indicator_id != metric_id
                or result.skill_name != inputs["skill"]["name"]
            ):
                return 0
            if result.status == "ok":
                if (
                    result.score is None
                    or result.reason is not None
                    or not result.observed_at
                    or not result.raw_values
                    or not result.calculated_values
                ):
                    return 0
                spec = expected["indicator_results"][metric_id]
                if any(
                    getattr(result, field) != spec[field]
                    for field in ("unit", "source", "url")
                ):
                    return 0
            elif (
                result.score is not None
                or not result.reason
                or not result.reason.strip()
                or result.raw_values
                or result.calculated_values
                or any(
                    getattr(result, field) is not None
                    for field in ("unit", "observed_at", "source", "url")
                )
            ):
                return 0
        return 1
    except (KeyError, TypeError, ValueError, ValidationError, AttributeError):
        return 0


def build_samples(cases: list[MetricCase]) -> list[dict[str, Any]]:
    """从历史输出构建样本，仅校验标注范围，不调用模型、工具或数据源。"""
    samples = []
    for case in cases:
        if any(
            message["type"] == "system"
            for snapshot in (case.input, case.actual_output)
            for message in snapshot.messages
        ):
            raise ValueError(f"第 {case.source_line} 行应删除系统提示词消息")
        inputs = case.input.model_dump(mode="json")
        expected = case.expected_output
        metric_ids = {metric.id for metric in case.input.skill.scoring.metrics}
        pending = metric_ids - case.input.indicator_results.keys()
        if set(expected.indicator_results) != metric_ids:
            raise ValueError(f"第 {case.source_line} 行的预期指标与技能定义不一致")
        if any(call.name not in case.input.skill.tools for call in expected.tool_calls):
            raise ValueError(f"第 {case.source_line} 行的预期工具不属于当前技能")
        kinds = [record.data_kind for record in expected.evidence.values()]
        if len(kinds) != len(set(kinds)) or set(kinds) != {
            METRIC_DATA_KINDS[metric_id] for metric_id in pending
        }:
            raise ValueError(f"第 {case.source_line} 行的预期证据与待计算指标不一致")
        samples.append(
            {
                "input": inputs,
                "response": case.actual_output.model_dump(mode="json"),
                "reference": expected.model_dump(
                    mode="json", exclude={"label_reason"}
                ),
                "metadata": {
                    "source_line": case.source_line,
                    "ts": case.ts.isoformat(),
                    "label_reason": expected.label_reason,
                },
            }
        )
    return samples


async def score_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """执行四项 RAGAS 自定义规则，不使用裁判模型。"""

    @experiment()
    async def run_evaluation(sample: dict[str, Any]) -> dict[str, Any]:
        row = dict(sample)
        actual = sample["response"]
        expected = sample["reference"]
        for metric in (tool_selection_correct, tool_arguments_correct):
            score = await metric.ascore(
                actual=tool_calls(actual), expected=expected["tool_calls"]
            )
            if score.value is None:
                raise ValueError(f"{metric.name} 评分失败：{score.reason}")
            row[metric.name] = score.value
        for metric in (evidence_valid, indicator_results_valid):
            score = await metric.ascore(
                actual=actual, inputs=sample["input"], expected=expected
            )
            if score.value is None:
                raise ValueError(f"{metric.name} 评分失败：{score.reason}")
            row[metric.name] = score.value
        return row

    dataset = Dataset(
        "calculate_metrics_evaluation", backend=InMemoryBackend(), data=samples
    )
    result = await run_evaluation.arun(dataset)
    if len(result) != len(samples):
        raise RuntimeError("存在未完成的评测案例，请检查 RAGAS 实验错误")
    return list(result)


def run_eval(cases: list[MetricCase]) -> dict[str, Any]:
    """对历史输出执行四项规则评分，保留原有契约校验范围。"""
    if not cases:
        raise ValueError("没有指标节点评测案例")
    started = time.perf_counter()
    print(f"读取 {len(cases)} 条历史指标输出...", flush=True)
    samples = build_samples(cases)
    print("RAGAS 四项规则指标评测中...", flush=True)
    rows = asyncio.run(score_samples(samples))
    aggregate = {
        name: round(sum(row[name] for row in rows) / len(rows), 4) for name in METRICS
    }
    print(f"评测完成（{time.perf_counter() - started:.0f}s）：{aggregate}")
    return {"aggregate": aggregate, "per_question": rows}


def main() -> None:
    cases_path = Path(__file__).with_name("calculate_metrics_cases.jsonl")
    output_path = (
        PROJECT_ROOT / "eval_agent/outputs/calculate_metrics_eval/results.json"
    )
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
