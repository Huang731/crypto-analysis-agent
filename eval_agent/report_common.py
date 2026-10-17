"""写作和审核评测共用的候选消息、人工标注、裁判与实验保存。"""

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import truststore
from langchain_core.messages import BaseMessage, messages_from_dict
from langfuse import Evaluation, Langfuse
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config.config import PROJECT_ROOT, settings
from app.crypto_analysis.crypto_analysis_parent_agent import (
    ReportReview,
    _invoke_report_review,
    _invoke_report_text,
)
from app.crypto_analysis.metric_service import _log_reason
from app.llm.model import (
    ainvoke_structured,
    make_bailian_fast_llm,
    make_bailian_smart_llm,
    make_glm_smart_llm,
    model_session,
)
from eval_agent.planning.run_planning_eval import EvalSettings
from eval_agent.prompts.report_eval import report_judge_messages
from eval_agent.skill_execution.run_skill_execution_eval import duration_summary
from eval_agent.skill_generation.run_skill_generation_eval import CallCounter


def content_sha256(value: Any) -> str:
    """对解码后的 JSON 内容计算哈希，候选文件追加不影响已有行身份。"""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class MessageData(BaseModel):
    """保留 LangChain 候选消息的所有字段。"""

    model_config = ConfigDict(extra="allow")
    content: str = Field(min_length=1)


class CapturedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["system", "human"]
    data: MessageData


class ReportInput(RootModel[list[CapturedMessage]]):
    """与 candidates.input 相同，直接使用完整的消息数组。"""

    @model_validator(mode="after")
    def validate_roles(self) -> "ReportInput":
        if [message.type for message in self.root] != ["system", "human"]:
            raise ValueError("报告案例必须保留一条 system 和一条 human 消息")
        return self

    def to_messages(self) -> list[BaseMessage]:
        return messages_from_dict(self.model_dump(mode="json"))


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    evidence_quotes: list[str] = Field(min_length=1)


class MessageEdit(BaseModel):
    """逐项记录相对候选输入的定向替换，不隐式改写其他证据。"""

    model_config = ConfigDict(extra="forbid")
    message_index: Literal[0, 1]
    before: str = Field(min_length=1)
    after: str


class ReportMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    label_reason: str = Field(min_length=1)
    source_candidate_file: str = Field(min_length=1)
    source_candidate_line: int = Field(ge=1)
    source_candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    edits: list[MessageEdit]
    criteria: list[Criterion] = Field(min_length=1)
    mode: Literal["generation", "judge_calibration"] = "generation"
    calibration_output: str | ReportReview | None = None
    assessment_should_pass: bool = True
    expected_failed_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_annotation(self) -> "ReportMetadata":
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("人工检查项 ID 重复")
        if (self.mode == "judge_calibration") != (self.calibration_output is not None):
            raise ValueError("只有裁判校准案例必须提供固定待评输出")
        if self.mode == "generation" and not self.assessment_should_pass:
            raise ValueError("真实生成案例必须以通过人工标准为目标")
        if not set(self.expected_failed_criteria) <= set(ids):
            raise ValueError("裁判校准标注了不存在的失败检查项")
        if not self.assessment_should_pass and not self.expected_failed_criteria:
            raise ValueError("负向裁判校准必须指定应识别的错误检查项")
        if self.assessment_should_pass and self.expected_failed_criteria:
            raise ValueError("预期通过的案例不能标注应失败的检查项")
        return self


class ReportCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: ReportInput
    expected_output: str | ReportReview
    metadata: ReportMetadata

    @model_validator(mode="after")
    def validate_evidence(self) -> "ReportCase":
        if (
            content_sha256(self.input.model_dump(mode="json"))
            != self.metadata.input_sha256
        ):
            raise ValueError("案例输入与记录的快照哈希不一致")
        # 表格在消息里以 JSON 字符串保存，解码只用于逐字证据定位，不重构业务输入。
        text = "\n".join(message.data.content for message in self.input.root)
        text = text.replace('\\"', '"').replace("\\n", "\n")
        for criterion in self.metadata.criteria:
            if any(
                not quote.strip() or quote not in text
                for quote in criterion.evidence_quotes
            ):
                raise ValueError(f"检查项 {criterion.id} 的证据引文不在固定输入中")
        return self


def load_cases(
    path: Path, case_type: type[ReportCase], check_sources: bool
) -> list[ReportCase]:
    """离线集自带完整输入，可显式要求逐行核验本地候选来源。"""
    cases = [
        case_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.metadata.case_id for case in cases}) != len(cases):
        raise ValueError("案例为空或案例 ID 重复")
    if check_sources:
        for case in cases:
            metadata = case.metadata
            path = PROJECT_ROOT / metadata.source_candidate_file
            lines = path.read_text(encoding="utf-8").splitlines()
            original = json.loads(lines[metadata.source_candidate_line - 1])
            if content_sha256(original) != metadata.source_candidate_sha256:
                raise ValueError(f"{metadata.case_id} 的候选来源行发生变化")
            messages = original["input"]
            for edit in metadata.edits:
                content = messages[edit.message_index]["data"]["content"]
                if content.count(edit.before) != 1:
                    raise ValueError(f"{metadata.case_id} 的来源替换不能唯一定位")
                messages[edit.message_index]["data"]["content"] = content.replace(
                    edit.before, edit.after, 1
                )
            if messages != case.input.model_dump(mode="json"):
                raise ValueError(f"{metadata.case_id} 的输入不符合记录的来源和替换")
    return cases


class CriterionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    passed: bool = Field(strict=True)
    output_quote: str
    evidence_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ExtraError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReportJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: list[CriterionJudgment]
    extra_errors: list[ExtraError]


def output_text(value: str | dict[str, Any]) -> str:
    """审核字段分别保留原文，避免 JSON 转义干扰逐字引文核验。"""
    if isinstance(value, str):
        return value
    return "\n".join(item for item in value.values() if isinstance(item, str))


def validate_judgment(
    judgment: ReportJudgment, criteria: list[Criterion], actual: str | dict[str, Any]
) -> None:
    """检查裁判没有漏项、增项或编造引文，不替代语义裁判。"""
    required = {criterion.id: criterion for criterion in criteria}
    ids = [item.id for item in judgment.criteria]
    if len(ids) != len(set(ids)) or set(ids) != set(required):
        raise ValueError("裁判必须且只能逐一评估全部人工检查项")
    text = output_text(actual)
    for item in judgment.criteria:
        if item.evidence_quote not in required[item.id].evidence_quotes:
            raise ValueError(f"裁判检查项 {item.id} 未使用人工标注的证据引文")
        if item.output_quote and item.output_quote not in text:
            raise ValueError(f"裁判检查项 {item.id} 编造了待评输出引文")
        if item.passed and not item.output_quote.strip():
            raise ValueError(f"裁判检查项 {item.id} 判为通过却未定位待评输出原文")
    if any(item.output_quote not in text for item in judgment.extra_errors):
        raise ValueError("裁判额外错误没有引用待评输出原文")


async def judge_report(
    judge: Any, node: str, case: ReportCase, actual: str | dict[str, Any]
) -> ReportJudgment:
    expected = case.expected_output
    messages = report_judge_messages(
        node,
        case.input.model_dump(mode="json"),
        actual,
        expected.model_dump() if isinstance(expected, ReportReview) else expected,
        [criterion.model_dump() for criterion in case.metadata.criteria],
    )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(min=1, max=2),
        reraise=True,
    )
    async def invoke() -> ReportJudgment:
        """仅重试裁判请求和校验，不重跑已经完成的业务输出。"""
        result = await ainvoke_structured(
            judge, messages, schema=ReportJudgment, timeout=120
        )
        validate_judgment(result, case.metadata.criteria, actual)
        return result

    return await asyncio.wait_for(invoke(), timeout=250)


def comparison(
    case: ReportCase,
    actual: str | dict[str, Any],
    judgment: ReportJudgment,
    node: str,
) -> dict[str, float | bool]:
    """判断值由代码比较，语义检查逐项计分，不冒充 RAGAS 忠实度。"""
    checks = [item.passed for item in judgment.criteria]
    semantic_pass = all(checks) and not judgment.extra_errors
    metrics: dict[str, float | bool] = {
        "criteria_pass_rate": sum(checks) / len(checks),
        "no_extra_errors": not judgment.extra_errors,
    }
    for item in judgment.criteria:
        metrics[f"criterion_{item.id}"] = item.passed
    if node == "reviewer":
        expected = ReportReview.model_validate(case.expected_output)
        actual_review = ReportReview.model_validate(actual)
        metrics["decision_correct"] = actual_review.passed == expected.passed
        semantic_pass &= metrics["decision_correct"]
    metrics["assessment_pass"] = semantic_pass
    failures = {item.id for item in judgment.criteria if not item.passed}
    metrics["case_pass"] = (
        semantic_pass == case.metadata.assessment_should_pass
        and set(case.metadata.expected_failed_criteria) <= failures
    )
    return metrics


def summarize(rows: list[dict[str, Any]], node: str) -> dict[str, Any]:
    """真实生成与裁判校准分开统计，执行失败和裁判失败保留未知评分。"""
    generated = [row for row in rows if row["metadata"]["mode"] == "generation"]
    calibrated = [row for row in rows if row["metadata"]["mode"] == "judge_calibration"]
    scored = [
        row for row in generated if row["output"]["execution_status"] == "success"
    ]
    result: dict[str, Any] = {
        "generation_cases": len(generated),
        "generation_passes": sum(
            row["output"]["metrics"].get("case_pass", False) for row in generated
        ),
        "judge_calibration_cases": len(calibrated),
        "judge_calibration_passes": sum(
            row["output"]["metrics"].get("case_pass", False) for row in calibrated
        ),
        "judge_calibration_passed": bool(calibrated)
        and all(row["output"]["metrics"].get("case_pass", False) for row in calibrated),
        "execution_errors": sum(
            row["output"]["error_stage"] == "target" for row in rows
        ),
        "judge_errors": sum(row["output"]["error_stage"] == "judge" for row in rows),
        "scored_generation_cases": len(scored),
        "accuracy_by_category": {
            category: sum(
                row["output"]["metrics"].get("case_pass", False)
                for row in generated
                if row["metadata"]["category"] == category
            )
            / sum(row["metadata"]["category"] == category for row in generated)
            for category in sorted({row["metadata"]["category"] for row in generated})
        },
        "target_duration": duration_summary(
            [
                row["output"]["target_duration_ms"]
                for row in generated
                if row["output"]["target_duration_ms"] is not None
            ]
        ),
        "judge_duration": duration_summary(
            [
                row["output"]["judge_duration_ms"]
                for row in rows
                if row["output"]["judge_duration_ms"] is not None
            ]
        ),
        "cost": None,
    }
    if node == "reviewer":
        decided = [
            row for row in generated if isinstance(row["output"]["actual_output"], dict)
        ]
        result["decision_evaluated_count"] = len(decided)
        result["decision_accuracy"] = (
            sum(
                row["output"]["actual_output"]["passed"]
                == row["expected_output"]["passed"]
                for row in decided
            )
            / len(decided)
            if decided
            else None
        )
        result["false_accept_count"] = sum(
            row["output"]["actual_output"]["passed"]
            and not row["expected_output"]["passed"]
            for row in decided
        )
        result["false_reject_count"] = sum(
            not row["output"]["actual_output"]["passed"]
            and row["expected_output"]["passed"]
            for row in decided
        )
    return result


def run_evaluation(
    node: Literal["writer", "reviewer"],
    case_type: type[ReportCase],
    evaluate: Callable[[Langfuse, ReportCase, str], Awaitable[dict[str, Any]]],
    expected_schema: dict[str, Any],
) -> None:
    """沿用独立 JSONL、生产函数、Langfuse 实验与本地结果的现有入口约定。"""
    parser = argparse.ArgumentParser(description=f"固定候选输入的 {node} 离线评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / f"eval_agent/{node}/{node}_cases.jsonl",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-certificates", action="store_true")
    parser.add_argument(
        "--judge-model",
        choices=(settings.bailian_fast_model, settings.bailian_smart_model),
        default=settings.bailian_smart_model,
    )
    args = parser.parse_args()
    cases = load_cases(args.cases, case_type, args.check_sources)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("案例数量必须大于零")
        cases = cases[: args.limit]
    if args.validate_only:
        print(
            f"已校验 {len(cases)} 个 {node} 案例：{dict(Counter(case.metadata.mode for case in cases))}"
        )
        return
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        f"{node}-%Y%m%dT%H%M%SZ"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    path = PROJECT_ROOT / f"outputs/{node}_eval/{run_name}.json"
    if path.exists():
        parser.error("结果文件已存在，请使用新的实验名称")
    if args.system_certificates:
        truststore.inject_into_ssl()
    config = EvalSettings()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    metadata = {
        "target_model": settings.glm_smart_model,
        "judge_model": args.judge_model,
        "writer_thinking": "enabled" if node == "writer" else None,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "production_code_sha256": hashlib.sha256(
            (
                PROJECT_ROOT / "app/crypto_analysis/crypto_analysis_parent_agent.py"
            ).read_bytes()
        ).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(
            (PROJECT_ROOT / "eval_agent/prompts/report_eval.py").read_bytes()
        ).hexdigest(),
        "evaluator_code_sha256": content_sha256(
            {
                name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                for name in [
                    "eval_agent/report_common.py",
                    f"eval_agent/{node}/run_{node}_eval.py",
                ]
            }
        ),
        "scoring_version": "candidate_report_v1",
        "input_contract": "candidates.input完整system/human消息数组",
        "data_mode": "fixed_candidate_messages",
        "model_timeout_seconds": 120,
        "max_attempts": 2,
        "boundary": "评估固定消息下的生产模型调用；不运行取数、缓存、重写循环或审核上限分支",
    }
    rows: list[dict[str, Any]] = []
    run_status = "failed"

    async def task(*, item: Any) -> dict[str, Any]:
        case = case_type.model_validate(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
            }
        )
        output = await evaluate(client, case, args.judge_model)
        rows.append(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
                "output": output,
            }
        )
        print(
            f"{case.metadata.case_id}：{output['execution_status']}，通过={output['metrics'].get('case_pass')}",
            flush=True,
        )
        return output

    def score_case(*, output: Any, metadata: Any, **kwargs: Any) -> list[Evaluation]:
        prefix = "judge_calibration_" if metadata["mode"] == "judge_calibration" else ""
        scores = [
            Evaluation(
                name=f"{prefix}evaluation_success",
                value=int(output["execution_status"] == "success"),
            )
        ]
        scores.extend(
            Evaluation(name=f"{prefix}{name}", value=float(value))
            for name, value in output["metrics"].items()
        )
        for stage in ("target", "judge"):
            if stage == "target" and metadata["mode"] == "judge_calibration":
                continue
            scores.append(
                Evaluation(
                    name=f"{prefix}{stage}_execution_success",
                    value=int(output[f"{stage}_status"] == "success"),
                )
            )
            if output[f"{stage}_duration_ms"] is not None:
                scores.append(
                    Evaluation(
                        name=f"{prefix}{stage}_duration_ms",
                        value=output[f"{stage}_duration_ms"],
                    )
                )
        return scores

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败")
        dataset = f"crypto-agent-{node}"
        client.create_dataset(
            name=dataset,
            description=f"候选消息快照、人工标注与独立裁判的{node}评测",
            input_schema=ReportInput.model_json_schema(),
            expected_output_schema=expected_schema,
        )
        items = []
        for case in cases:
            expected = case.expected_output
            items.append(
                client.create_dataset_item(
                    dataset_name=dataset,
                    id=str(uuid5(NAMESPACE_URL, f"{dataset}:{case.metadata.case_id}")),
                    input=case.input.model_dump(mode="json"),
                    expected_output=expected.model_dump()
                    if isinstance(expected, ReportReview)
                    else expected,
                    metadata=case.metadata.model_dump(mode="json"),
                )
            )
        experiment = client.run_experiment(
            name=dataset,
            run_name=run_name,
            data=items,
            task=task,
            evaluators=[score_case],
            max_concurrency=1,
            metadata=metadata,
        )
        for row, evaluated in zip(rows, experiment.item_results, strict=True):
            row["scores"] = [score.__dict__ for score in evaluated.evaluations]
            row["langfuse_trace_id"] = evaluated.trace_id
        if len(rows) != len(cases):
            raise RuntimeError("评测记录不完整")
        run_status = "completed"
        print(experiment.format())
    finally:
        report = {
            "run_name": run_name,
            "run_status": run_status,
            "planned_cases": len(cases),
            "metadata": metadata,
            "summary": summarize(rows, node),
            "cases": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        client.flush()
        print(f"本地结果：{path}")


def evaluation_output() -> dict[str, Any]:
    """未知分数保持空，模型调用失败不会被记作回答质量零分。"""
    return {
        "execution_status": "error",
        "target_status": "pending",
        "judge_status": "pending",
        "actual_output": None,
        "judgment": None,
        "metrics": {},
        "error_stage": None,
        "error_type": None,
        "target_duration_ms": None,
        "judge_duration_ms": None,
    }


async def evaluate_case(
    client: Langfuse,
    case: ReportCase,
    judge_model: str,
    node: Literal["writer", "reviewer"],
) -> dict[str, Any]:
    """直接复用生产模型调用；裁判与被测节点各自持有连接和调用统计。"""
    output = evaluation_output()
    target_counter = CallCounter()
    judge_counter = CallCounter()
    stage = "target"
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    with client.start_as_current_observation(
        as_type="span",
        name=f"{node}.evaluate",
        input=case.input.model_dump(mode="json"),
    ) as observation:
        try:
            async with AsyncExitStack() as stack:
                if case.metadata.mode == "judge_calibration":
                    actual = case.metadata.calibration_output
                    output["target_status"] = "skipped"
                else:
                    http = await stack.enter_async_context(model_session())
                    target = make_glm_smart_llm(async_client=http).with_config(
                        callbacks=[target_counter]
                    )
                    if node == "writer":
                        target = target.bind(
                            extra_body={"thinking": {"type": "enabled"}}
                        )
                        actual = await asyncio.wait_for(
                            _invoke_report_text(
                                target, case.input.to_messages(), events.append
                            ),
                            timeout=250,
                        )
                    else:
                        actual = await asyncio.wait_for(
                            _invoke_report_review(target, case.input.to_messages()),
                            timeout=250,
                        )
                    output["target_duration_ms"] = (
                        time.perf_counter() - started
                    ) * 1000
                    output["target_status"] = "success"
                actual = (
                    actual.model_dump() if isinstance(actual, ReportReview) else actual
                )
                output["actual_output"] = actual
                if node == "reviewer":
                    expected = ReportReview.model_validate(case.expected_output)
                    output["metrics"]["decision_correct"] = (
                        actual["passed"] == expected.passed
                    )
                stage = "judge"
                started = time.perf_counter()
                http = await stack.enter_async_context(model_session())
                factory = (
                    make_bailian_fast_llm
                    if judge_model == settings.bailian_fast_model
                    else make_bailian_smart_llm
                )
                judge = (
                    factory(async_client=http)
                    .bind(enable_thinking=False)
                    .with_config(callbacks=[judge_counter])
                )
                with client.start_as_current_observation(
                    as_type="generation",
                    name=f"{node}.independent_judge",
                    model=judge_model,
                    input={
                        "actual_output": actual,
                        "criteria": [
                            item.model_dump() for item in case.metadata.criteria
                        ],
                    },
                ) as judge_observation:
                    judgment = await judge_report(judge, node, case, actual)
                    judge_observation.update(output=judgment.model_dump())
                output["judge_duration_ms"] = (time.perf_counter() - started) * 1000
                output["judge_status"] = "success"
                output.update(
                    execution_status="success",
                    judgment=judgment.model_dump(),
                    metrics=comparison(case, actual, judgment, node),
                )
        except Exception as exc:  # noqa: BLE001 - 保留单项失败，继续后续案例。
            output[f"{stage}_status"] = "error"
            output[f"{stage}_duration_ms"] = (time.perf_counter() - started) * 1000
            record_error(
                output,
                exc,
                stage,
                judge_counter if stage == "judge" else target_counter,
            )
        output["target_model_attempts"] = target_counter.attempts
        output["judge_model_attempts"] = judge_counter.attempts
        output["token_usage"] = {
            "target": target_counter.usage,
            "judge": judge_counter.usage,
        }
        output["events"] = events
        observation.update(
            output=output,
            level="DEFAULT" if output["execution_status"] == "success" else "ERROR",
        )
    return output


def record_error(
    output: dict[str, Any], exc: Exception, stage: str, counter: CallCounter
) -> None:
    """保留失败阶段、脱敏异常与实际响应，便于后续核验。"""
    message = str(exc)
    raw = json.dumps(counter.last_response, ensure_ascii=False)
    for key in (settings.dashscope_api_key, settings.zhipuai_api_key):
        if key:
            message = message.replace(key, "[已隐藏]")
            raw = raw.replace(key, "[已隐藏]")
    output.update(
        error_stage=stage,
        error_type=type(exc).__name__,
        error_message=_log_reason(message)[:600],
        error_response=raw[:4000],
    )
