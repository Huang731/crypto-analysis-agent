"""固定取数响应，直接调用生产 execute_skill，保存 Langfuse 与本地结果。"""

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, isclose
from pathlib import Path
from statistics import median
from typing import Any, Literal
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import httpx
import truststore
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langfuse import Evaluation, Langfuse, get_client
from langfuse.api import DatasetItem
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config.config import PROJECT_ROOT, settings
from app.crypto_analysis.crypto_analysis_agent import build_metric_agent, execute_skill
from app.crypto_analysis.metric_service import _log_reason
from app.crypto_analysis.skill_loader import LoadedSkill
from app.llm.model import _make_model
from eval_agent.planning.run_planning_eval import CaseMetadata, EvalSettings
from prompts.crypto_analysis import metric_agent_system_prompt, metric_execution_prompt

# 只替换工具模块引用的数据源函数，取数证据处理、计算器与聚合保持原样。
SOURCE_TARGETS = {
    "fear_greed": "app.crypto_analysis.tools.sentiment_tools.get_fear_and_greed",
    "derivatives": "app.crypto_analysis.tools.derivatives_tools.get_binance_derivatives",
    "etf": "app.crypto_analysis.tools.etf_tools.get_crypto_etf_flows",
    "trend": "app.crypto_analysis.tools.trend_tools.get_binance_trend_klines",
    "fed_target_range": "app.crypto_analysis.tools.fed_policy_tools.get_fed_target_range",
    "dxy": "app.crypto_analysis.tools.liquidity_tools.query_dxy_from_official_mcp",
    "stablecoin": "app.crypto_analysis.tools.liquidity_tools.get_stablecoin_historical_market_cap",
    "fred_observations": "app.crypto_analysis.tools.liquidity_tools.get_m2_yoy_change",
}

# 各指标必须使用对应类型的原始证据，共用数据的 OI 与 Funding 指向同一类型。
METRIC_DATA_KINDS = {
    "OI": "derivatives",
    "Funding": "derivatives",
    "ETF_Flows": "etf",
    "Trend": "trend",
    "Fed_Rate_Change_3M": "fed_target_range",
    "Fear_Greed": "fear_greed",
    "DXY_Trend": "dxy",
    "Stable_MCap": "stablecoin",
    "M2_YoY": "m2",
    "Fed_Balance": "walcl",
}


class SourceResponse(BaseModel):
    """按数据源与参数匹配固定返回值，或模拟明确的取数异常。"""

    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    arguments: dict[str, Any]
    raw: dict[str, Any] | None = None
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def valid_response(self) -> "SourceResponse":
        if self.source not in SOURCE_TARGETS:
            raise ValueError("回放数据源未注册")
        if (self.raw is None) == (self.error is None):
            raise ValueError("raw 与 error 必须且只能提供一个")
        return self


class SkillExecutionScope(BaseModel):
    """保留真实技能节点收到的请求范围；as_of 仅为原有展示时间。"""

    model_config = ConfigDict(extra="forbid")
    as_of: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    resolved_question: str = Field(min_length=1)
    target_asset: str = Field(pattern=r"^[A-Z0-9]+$", strict=True)
    thread_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)


class MetricAgentConfig(BaseModel):
    """Agent 对象不能序列化，记录重建生产 Agent 所需的模型配置。"""

    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)


class SkillExecutionInput(BaseModel):
    """逐项对应 execute_skill 的 metric_agent、skill、scope 三个入参。"""

    model_config = ConfigDict(extra="forbid")
    metric_agent: MetricAgentConfig
    skill: LoadedSkill
    scope: SkillExecutionScope


class SourceReplayConfig(BaseModel):
    """固定取数响应属于评测环境配置，不混入生产函数入参。"""

    model_config = ConfigDict(extra="forbid")
    source_responses: list[SourceResponse] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_responses(self) -> "SourceReplayConfig":
        identifiers = [
            (item.source, json.dumps(item.arguments, sort_keys=True))
            for item in self.source_responses
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("同一数据源和参数不能配置多个回放响应")
        return self


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


class SkillExecutionResult(BaseModel):
    """生产函数返回的完整技能结果，不以模型正文代替指标结果。"""

    model_config = ConfigDict(extra="forbid")
    skill_name: str
    name: str
    status: Literal[
        "ok", "partial", "data_missing", "not_applicable", "calculation_error"
    ]
    reason: str | None
    score: float | None = Field(ge=0, le=10, allow_inf_nan=False, strict=True)
    indicators: list[IndicatorResult] = Field(min_length=1)


class SkillExecutionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: SkillExecutionInput
    expected_output: SkillExecutionResult
    replay: SourceReplayConfig
    metadata: CaseMetadata


class SourceCall(BaseModel):
    source: str
    arguments: dict[str, Any]
    matched: bool


class SkillExecutionOutput(BaseModel):
    """错误保留独立状态；耗时包含 execute_skill 内全部模型与工具步骤。"""

    model_config = ConfigDict(extra="forbid")
    result: SkillExecutionResult | None = None
    raw_result: Any = None
    execution_status: Literal["success", "error"]
    duration_ms: float = Field(ge=0)
    error_type: str | None = None
    error_stage: str | None = None
    error_message: str | None = None
    replay_valid: bool | None
    source_calls: list[SourceCall]
    tool_calls: list[dict[str, Any]]
    tool_error_count: int = Field(ge=0)
    model_attempt_count: int = Field(ge=0)
    evidence_chain_valid: bool
    token_usage: list[dict[str, Any]]


@dataclass
class ReplayState(AsyncCallbackHandler):
    """一次案例持有自己的固定响应与调用记录；评测必须串行运行。"""

    responses: list[SourceResponse]
    source_calls: list[SourceCall] = field(default_factory=list)
    agent_state: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_error_count: int = 0
    model_attempt_count: int = 0
    token_usage: list[dict[str, Any]] = field(default_factory=list)

    async def on_chat_model_start(
        self, serialized: Any, messages: Any, **kwargs: Any
    ) -> None:
        self.model_attempt_count += 1

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                self.tool_calls.extend(
                    dict(call) for call in generation.message.tool_calls
                )
                if generation.message.usage_metadata:
                    self.token_usage.append(dict(generation.message.usage_metadata))

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        messages = (
            output.update.get("messages", [])
            if isinstance(output, Command)
            else [output]
        )
        self.tool_error_count += sum(
            getattr(message, "status", None) == "error" for message in messages
        )

    async def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self.tool_error_count += 1

    def source_function(self, source: str) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def replay(**arguments: Any) -> dict[str, Any]:
            matches = [
                item
                for item in self.responses
                if item.source == source and item.arguments == arguments
            ]
            self.source_calls.append(
                SourceCall(source=source, arguments=arguments, matched=bool(matches))
            )
            if not matches:
                # 未配置的调用只能失败，禁止访问真实行情或补造默认数据。
                raise RuntimeError("回放案例未提供匹配的数据源响应")
            response = matches[0]
            if response.error is not None:
                raise RuntimeError(response.error)
            assert response.raw is not None
            return deepcopy(response.raw)

        return replay


@dataclass
class RecordedMetricAgent:
    """保留生产 Agent 最终状态供检查，调用参数完全由 execute_skill 提供。"""

    agent: Any
    replay: ReplayState

    async def ainvoke(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        config = kwargs["config"]
        kwargs["config"] = {**config, "callbacks": [self.replay]}
        state = await self.agent.ainvoke(inputs, **kwargs)
        self.replay.agent_state = state
        return state


@dataclass
class EvaluationRunState:
    """当前评测持有运行记录和诊断，不把它们混入生产出参。"""

    rows: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, SkillExecutionOutput] = field(default_factory=dict)


def load_cases(path: Path) -> list[SkillExecutionCase]:
    """校验案例、唯一标识、技能归属、指标顺序及缺失分数，禁止静默跳过。"""
    cases: list[SkillExecutionCase] = []
    identifiers: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = SkillExecutionCase.model_validate_json(line)
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False)
            raise ValueError(f"{path}:{number} 案例格式错误：{errors}") from None
        if case.metadata.case_id in identifiers:
            raise ValueError(f"{path}:{number} 案例标识重复：{case.metadata.case_id}")
        skill = case.input.skill
        expected = case.expected_output
        if expected.skill_name != skill.name or expected.name != skill.scoring.name:
            raise ValueError(f"{path}:{number} 预期结果与技能不一致")
        if [item.indicator_id for item in expected.indicators] != [
            metric.id for metric in skill.scoring.metrics
        ]:
            raise ValueError(f"{path}:{number} 预期指标与技能配置顺序不一致")
        for item in expected.indicators:
            if item.skill_name != skill.name:
                raise ValueError(f"{path}:{number} 指标所属技能不一致")
            if item.status == "ok" and item.score is None:
                raise ValueError(f"{path}:{number} 成功指标缺少分数")
            if item.status != "ok" and (item.score is not None or not item.reason):
                raise ValueError(f"{path}:{number} 缺失指标必须无分数且有原因")
        identifiers.add(case.metadata.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} 没有评测案例")
    return cases


def tool_trajectory(
    state: dict[str, Any] | None, skill: LoadedSkill
) -> tuple[list[dict[str, Any]], int, bool]:
    """核对组合工具调用、各指标状态与完整证据关联。"""
    if state is None:
        return [], 0, False
    calls: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    responded: set[str] = set()
    errors = 0
    valid = True
    for message in state["messages"]:
        if message.type == "ai":
            for call in message.tool_calls:
                calls.append(dict(call))
                calls_by_id[call["id"]] = call
                if call["name"] not in skill.tools:
                    valid = False
                valid &= call["args"] == {}
        elif message.type == "tool":
            if message.status == "error":
                # 框架校验错误是普通文本，保留失败事实，不当作取数证据解析。
                errors += 1
                valid = False
                continue
            payload = json.loads(message.content)
            if "indicators" not in payload or not isinstance(message.artifact, dict):
                valid = False
                continue
            references = message.artifact.get("indicator_evidence_ids")
            if not isinstance(references, dict):
                valid = False
                continue
            valid &= message.tool_call_id in calls_by_id
            valid &= message.tool_call_id not in responded
            responded.add(message.tool_call_id)
            summaries = payload["indicators"]
            metric_ids = [item["indicator_id"] for item in summaries]
            valid &= len(metric_ids) == len(set(metric_ids))
            valid &= set(metric_ids) == {metric.id for metric in skill.scoring.metrics}
            valid &= set(references) == set(metric_ids)
            call_evidence = {
                evidence_id
                for evidence_id in state["evidence"]
                if evidence_id.startswith(f"{message.tool_call_id}:")
            }
            valid &= bool(call_evidence) and set(references.values()) == call_evidence
            for item in summaries:
                evidence_id = references.get(item["indicator_id"])
                evidence = state["evidence"].get(evidence_id)
                valid &= (
                    evidence is not None
                    and evidence["data_kind"]
                    == (METRIC_DATA_KINDS[item["indicator_id"]])
                )
                result = state["indicator_results"].get(item["indicator_id"])
                valid &= result is not None and item["status"] == result["status"]
                if result is not None and result["status"] != "ok":
                    valid &= item.get("reason") == result["reason"]
    return (
        calls,
        errors,
        (
            valid
            and bool(calls)
            and len(calls_by_id) == len(calls)
            and responded == calls_by_id.keys()
        ),
    )


async def evaluate_skill(
    client: Langfuse,
    inputs: SkillExecutionInput,
    replay_config: SourceReplayConfig,
    timeout_seconds: float,
) -> SkillExecutionOutput:
    """真实模型选择工具，只替换取数返回，沿用生产计算器、重试与单次超时。"""
    skill = inputs.skill
    replay = ReplayState(responses=replay_config.source_responses)
    # 每例持有独立异步连接，避免关闭框架共享连接后影响下一例。
    http_client = httpx.AsyncClient(timeout=120)
    try:
        llm = _make_model(inputs.metric_agent.model, http_async_client=http_client)
        with ExitStack() as stack:
            for source, target in SOURCE_TARGETS.items():
                stack.enter_context(patch(target, new=replay.source_function(source)))
            agent = RecordedMetricAgent(build_metric_agent(llm, skill), replay)
            with client.start_as_current_observation(
                as_type="span",
                name="skill_execution.invoke",
                input=inputs.model_dump(mode="json"),
                metadata={
                    "model": inputs.metric_agent.model,
                    "data_mode": "fixed_replay",
                },
            ) as observation:
                started = time.perf_counter()
                stage = "agent"
                raw_result: Any = None
                try:
                    raw_result = await asyncio.wait_for(
                        execute_skill(agent, skill, inputs.scope.model_dump()),
                        timeout=timeout_seconds,
                    )
                    duration_ms = (time.perf_counter() - started) * 1000
                    stage = "result_validation"
                    actual = SkillExecutionResult.model_validate(raw_result)
                    status, error_type, error_stage, error_message = (
                        "success",
                        None,
                        None,
                        None,
                    )
                except Exception as exc:  # noqa: BLE001 - 保留单项失败，继续其他案例。
                    duration_ms = (time.perf_counter() - started) * 1000
                    actual, status, error_type = None, "error", type(exc).__name__
                    error_stage = stage
                    error_message = _log_reason(
                        str(exc).replace(settings.zhipuai_api_key, "[已隐藏]")
                    )[:1000]
                _, errors, chain_valid = tool_trajectory(replay.agent_state, skill)
                configured = Counter(
                    (item.source, json.dumps(item.arguments, sort_keys=True))
                    for item in replay.responses
                )
                requested = Counter(
                    (call.source, json.dumps(call.arguments, sort_keys=True))
                    for call in replay.source_calls
                )
                output = SkillExecutionOutput(
                    result=actual,
                    raw_result=raw_result,
                    execution_status=status,
                    duration_ms=duration_ms,
                    error_type=error_type,
                    error_stage=error_stage,
                    error_message=error_message,
                    replay_valid=(
                        configured == requested
                        and all(call.matched for call in replay.source_calls)
                        if status == "success" or replay.source_calls
                        else None
                    ),
                    source_calls=replay.source_calls,
                    tool_calls=replay.tool_calls,
                    tool_error_count=(
                        errors
                        if replay.agent_state is not None
                        else replay.tool_error_count
                    ),
                    model_attempt_count=replay.model_attempt_count,
                    evidence_chain_valid=chain_valid,
                    token_usage=replay.token_usage,
                )
                observation.update(
                    output=raw_result,
                    metadata=output.model_dump(
                        mode="json", exclude={"result", "raw_result"}
                    ),
                    level="DEFAULT" if status == "success" else "ERROR",
                )
                return output
    finally:
        await http_client.aclose()


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


def result_differences(
    actual: Any, expected: Any, path: str = "$"
) -> list[dict[str, Any]]:
    """记录具体字段差异，沿用完整结构比较及浮点容差。"""
    if values_equal(actual, expected):
        return []
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(actual.keys() | expected.keys()):
            field_path = f"{path}.{key}"
            if key not in actual or key not in expected:
                differences.append(
                    {
                        "path": field_path,
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                        "expected_present": key in expected,
                        "actual_present": key in actual,
                    }
                )
            else:
                differences.extend(
                    result_differences(actual[key], expected[key], field_path)
                )
        return differences
    if isinstance(actual, list) and isinstance(expected, list):
        differences = []
        if len(actual) != len(expected):
            differences.append(
                {
                    "path": f"{path}.length",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        for index, (left, right) in enumerate(zip(actual, expected)):
            differences.extend(result_differences(left, right, f"{path}[{index}]"))
        return differences
    return [{"path": path, "expected": expected, "actual": actual}]


def compare_skill(
    actual: SkillExecutionOutput, expected: SkillExecutionResult, skill: LoadedSkill
) -> dict[str, bool | None]:
    """分开检查调度、计算、状态和聚合；无结果时计算正确性保持未知。"""
    success = actual.execution_status == "success" and actual.result is not None
    scores: dict[str, bool | None] = {
        "execution_success": success,
        "replay_valid": actual.replay_valid,
        "tool_selection_correct": None,
        "indicator_calculation_correct": None,
        "status_handling_correct": None,
        "aggregate_score_correct": None,
    }
    if success or actual.error_stage == "result_validation":
        scores["tool_selection_correct"] = (
            Counter(call["name"] for call in actual.tool_calls) == Counter(skill.tools)
            and all(call["args"] == {} for call in actual.tool_calls)
            and actual.evidence_chain_valid
            and actual.tool_error_count == 0
        )
        result = actual.raw_result
        if isinstance(result, dict):
            indicators = result.get("indicators")
            valid_indicators = isinstance(indicators, list) and all(
                isinstance(item, dict) for item in indicators
            )
            scores["indicator_calculation_correct"] = valid_indicators and values_equal(
                [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"status", "reason"}
                    }
                    for item in indicators
                ],
                [
                    item.model_dump(exclude={"status", "reason"})
                    for item in expected.indicators
                ],
            )
            scores["status_handling_correct"] = valid_indicators and values_equal(
                {
                    **{
                        key: value
                        for key, value in result.items()
                        if key in {"status", "reason"}
                    },
                    "indicators": [
                        {
                            key: value
                            for key, value in item.items()
                            if key in {"indicator_id", "status", "reason"}
                        }
                        for item in indicators
                    ],
                },
                {
                    "status": expected.status,
                    "reason": expected.reason,
                    "indicators": [
                        item.model_dump(include={"indicator_id", "status", "reason"})
                        for item in expected.indicators
                    ],
                },
            )
            scores["aggregate_score_correct"] = values_equal(
                {
                    key: value
                    for key, value in result.items()
                    if key in {"skill_name", "name", "score"}
                },
                expected.model_dump(include={"skill_name", "name", "score"}),
            )
        else:
            scores.update(
                indicator_calculation_correct=False,
                status_handling_correct=False,
                aggregate_score_correct=False,
            )
    scores["skill_execution_correct"] = all(value is True for value in scores.values())
    return scores


def score_skill(
    *,
    output: Any,
    expected_output: Any,
    input: Any,
    diagnostics: SkillExecutionOutput,
) -> list[Evaluation]:
    """案例出参就是生产结果；执行诊断单独保存，评分不调用裁判模型。"""
    expected = SkillExecutionResult.model_validate(expected_output)
    skill = LoadedSkill.model_validate(input["skill"])
    recorded = diagnostics.raw_result
    if output is not recorded and not values_equal(output, recorded):
        raise ValueError("案例出参与记录的 execute_skill 返回结果不一致")
    details = {
        "execution_status": diagnostics.execution_status,
        "error_type": diagnostics.error_type,
        "error_stage": diagnostics.error_stage,
        "error_message": diagnostics.error_message,
        "differences": result_differences(output, expected.model_dump())
        if output is not None
        else [],
        "tool_call_count": len(diagnostics.tool_calls),
        "tool_error_count": diagnostics.tool_error_count,
        "model_attempt_count": diagnostics.model_attempt_count,
    }
    comment = json.dumps(details, ensure_ascii=False)
    return [
        *[
            Evaluation(name=name, value=int(correct), comment=comment)
            for name, correct in compare_skill(diagnostics, expected, skill).items()
            if correct is not None
        ],
        Evaluation(
            name="duration_ms",
            value=diagnostics.duration_ms,
            comment="execute_skill 全程耗时，包含模型、重试、工具和聚合，单位毫秒",
        ),
    ]


def duration_summary(durations: list[float]) -> dict[str, float | None]:
    """P50 使用中位数，P95 使用最近秩；无案例时明确返回未知。"""
    if not durations:
        return {
            "mean_duration_ms": None,
            "p50_duration_ms": None,
            "p95_duration_ms": None,
        }
    ordered = sorted(durations)
    return {
        "mean_duration_ms": sum(durations) / len(durations),
        "p50_duration_ms": median(durations),
        "p95_duration_ms": ordered[ceil(len(durations) * 0.95) - 1],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: Counter[str] = Counter()
    evaluated: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    durations: dict[str, list[float]] = {"success": [], "error": []}
    tool_calls = 0
    tool_errors = 0
    model_attempts = 0
    token_usage: list[dict[str, Any]] = []
    for row in rows:
        actual = SkillExecutionOutput.model_validate(
            {
                "result": row["output"]
                if row["diagnostics"]["execution_status"] == "success"
                else None,
                "raw_result": row["output"],
                **row["diagnostics"],
            }
        )
        expected = SkillExecutionResult.model_validate(row["expected_output"])
        skill = LoadedSkill.model_validate(row["input"]["skill"])
        comparison = compare_skill(actual, expected, skill)
        for name, correct in comparison.items():
            if correct is not None:
                scores[name] += int(correct)
                evaluated[name] += 1
        statuses[actual.execution_status] += 1
        durations[actual.execution_status].append(actual.duration_ms)
        tool_calls += len(actual.tool_calls)
        tool_errors += actual.tool_error_count
        model_attempts += actual.model_attempt_count
        token_usage.extend(actual.token_usage)
        category = categories.setdefault(row["metadata"]["category"], Counter())
        category["runs"] += 1
        category["correct"] += int(comparison["skill_execution_correct"])
    total = len(rows)
    return {
        "planned_runs": total,
        "correct_runs": scores["skill_execution_correct"],
        "overall_success_rate": scores["skill_execution_correct"] / total,
        "accuracy_by_metric": {
            name: scores[name] / evaluated[name] if evaluated[name] else None
            for name in comparison
        },
        "evaluated_count_by_metric": {name: evaluated[name] for name in comparison},
        "unknown_count_by_metric": {
            name: total - evaluated[name] for name in comparison
        },
        "execution_status_counts": dict(statuses),
        "accuracy_by_category": {
            name: counts["correct"] / counts["runs"]
            for name, counts in categories.items()
        },
        **duration_summary(durations["success"] + durations["error"]),
        "duration_by_execution_status": {
            status: duration_summary(values) for status, values in durations.items()
        },
        "tool_call_count": tool_calls,
        "mean_tool_call_count": tool_calls / total,
        "tool_error_count": tool_errors,
        "model_attempts": model_attempts,
        "token_usage": {
            name: sum(usage[name] for usage in token_usage)
            if token_usage and all(name in usage for usage in token_usage)
            else None
            for name in ("input_tokens", "output_tokens", "total_tokens")
        },
        "model_calls_with_token_usage": len(token_usage),
        "cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="固定数据回放的独立技能执行评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("skill_execution_cases.jsonl"),
    )
    parser.add_argument("--dataset", default="crypto-agent-skill-execution-result")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--system-certificates", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/skill_execution_eval")
    )
    args = parser.parse_args()
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        parser.error("重复次数和案例数量必须大于零")
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("单案例超时必须是有限正数")
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.validate_only:
        print(
            f"已校验 {len(cases)} 个案例：{dict(Counter(c.metadata.category for c in cases))}"
        )
        return
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "skill-execution-%Y%m%dT%H%M%SZ"
    )
    output_path = args.output_dir / f"{run_name}.json"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    if output_path.exists():
        parser.error(f"结果文件已存在，请更换实验名称：{output_path}")
    if not settings.zhipuai_api_key.strip() or settings.zhipuai_api_key.startswith(
        "xxxx"
    ):
        parser.error("未配置 ZHIPUAI_API_KEY")
    if args.system_certificates:
        truststore.inject_into_ssl()
    config = EvalSettings()
    hash_paths = [
        PROJECT_ROOT / "app/crypto_analysis/crypto_analysis_agent.py",
        PROJECT_ROOT / "app/crypto_analysis/metric_service.py",
        Path(__file__),
        *sorted((PROJECT_ROOT / "app/crypto_analysis/calculators").glob("*.py")),
        *sorted((PROJECT_ROOT / "app/crypto_analysis/tools").glob("*.py")),
    ]
    metadata = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            (
                metric_agent_system_prompt()
                + "\n"
                + "\n".join(
                    metric_execution_prompt(case.input.skill.model_dump())
                    for case in cases
                )
            ).encode("utf-8")
        ).hexdigest(),
        "skill_configs_sha256": hashlib.sha256(
            json.dumps(
                {
                    case.metadata.case_id: case.input.skill.model_dump(mode="json")
                    for case in cases
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "working_tree_sha256": hashlib.sha256(
            b"".join(
                path.relative_to(PROJECT_ROOT).as_posix().encode()
                + b"\0"
                + path.read_bytes()
                + b"\0"
                for path in hash_paths
            )
        ).hexdigest(),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "models": sorted({case.input.metric_agent.model for case in cases}),
        "scoring_version": "execute_skill_result_v2",
        "input_contract": "metric_agent配置、skill、scope对应生产函数三个参数",
        "output_contract": "案例expected_output与output均为execute_skill完整result",
        "data_mode": "fixed_replay",
        "repetitions": str(args.repetitions),
        "case_count": str(len(cases)),
        "model_timeout_seconds": "120",
        "max_attempts": "2",
        "case_timeout_seconds": str(args.timeout_seconds),
        "system_certificates": str(args.system_certificates),
    }
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    client = get_client(public_key=config.langfuse_public_key)
    run_state = EvaluationRunState()

    async def task(*, item: DatasetItem) -> dict[str, Any] | None:
        inputs = SkillExecutionInput.model_validate(item.input)
        replay_config = SourceReplayConfig.model_validate(item.metadata["replay"])
        output = await evaluate_skill(
            client, inputs, replay_config, args.timeout_seconds
        )
        result = output.raw_result
        run_state.diagnostics[item.metadata["case_id"]] = output
        run_state.rows.append(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "replay": replay_config.model_dump(mode="json"),
                "metadata": {k: v for k, v in item.metadata.items() if k != "replay"},
                "output": result,
                "diagnostics": output.model_dump(
                    mode="json", exclude={"result", "raw_result"}
                ),
                "differences": result_differences(result, item.expected_output)
                if result is not None
                else [],
            }
        )
        print(
            f"{item.metadata['case_id']}：{output.execution_status}，{output.duration_ms:.0f} ms",
            flush=True,
        )
        return result

    def evaluate_item(
        *,
        output: Any,
        expected_output: Any,
        input: Any,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> list[Evaluation]:
        return score_skill(
            output=output,
            expected_output=expected_output,
            input=input,
            diagnostics=run_state.diagnostics[metadata["case_id"]],
        )

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败，请检查密钥和服务地址")
        client.create_dataset(
            name=args.dataset,
            description="固定数据源响应与人工预期结果的独立 execute_skill 评测",
            input_schema=SkillExecutionInput.model_json_schema(),
            expected_output_schema=SkillExecutionResult.model_json_schema(),
        )
        items = [
            client.create_dataset_item(
                dataset_name=args.dataset,
                id=str(uuid5(NAMESPACE_URL, f"{args.dataset}:{case.metadata.case_id}")),
                input=case.input.model_dump(mode="json"),
                expected_output=case.expected_output.model_dump(mode="json"),
                metadata={
                    **case.metadata.model_dump(),
                    "replay": case.replay.model_dump(mode="json"),
                },
            )
            for case in cases
        ]
        print(f"开始 {len(items) * args.repetitions} 次技能执行评测", flush=True)
        for repetition in range(1, args.repetitions + 1):
            row_start = len(run_state.rows)
            result = client.run_experiment(
                name=args.dataset,
                run_name=f"{run_name}-r{repetition}",
                data=items,
                task=task,
                evaluators=[evaluate_item],
                max_concurrency=1,
                metadata=metadata,
            )
            for row, item_result in zip(
                run_state.rows[row_start:], result.item_results, strict=True
            ):
                names = {score.name for score in item_result.evaluations}
                if (
                    not {"execution_success", "skill_execution_correct", "duration_ms"}
                    <= names
                ):
                    raise RuntimeError("评分记录不完整，不能输出完整实验结果")
                row["repetition"] = repetition
                row["scores"] = [score.__dict__ for score in item_result.evaluations]
                row["langfuse_trace_id"] = item_result.trace_id
            print(result.format())
        if len(run_state.rows) != len(items) * args.repetitions:
            raise RuntimeError("执行记录不完整，不能输出完整成功率")
        report = {
            "run_name": run_name,
            "metadata": metadata,
            "summary": summarize(run_state.rows),
            "cases": run_state.rows,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"本地结果：{output_path.resolve()}")
    finally:
        client.flush()


if __name__ == "__main__":
    main()
