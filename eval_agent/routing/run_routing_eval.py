"""读取案例、调用路由并评分，保存 Langfuse 实验和本地结果。"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import truststore
from langchain_core.language_models.chat_models import BaseChatModel
from langfuse import Evaluation, Langfuse, get_client
from langfuse.api import DatasetItem
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.config import PROJECT_ROOT, _env_files, settings
from app.llm.model import make_fast_llm, model_session
from app.orchestration.routing import (
    RouteCallState,
    RouteDecision,
    RouteTarget,
    decide_route,
)
from prompts.routing import routing_prompt


class RoutingInput(BaseModel):
    """已改写的固定问题，不调用问题改写环节。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    resolved_question: str = Field(min_length=1)


class RoutingExpected(BaseModel):
    """人工标注的唯一预期分支。"""

    model_config = ConfigDict(extra="forbid")
    route: RouteTarget


class CaseMetadata(BaseModel):
    """案例标识、分类和人工标注理由。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    label_reason: str = Field(min_length=1)


class RoutingCase(BaseModel):
    """一行 JSONL 对应一个完整案例。"""

    model_config = ConfigDict(extra="forbid")
    input: RoutingInput
    expected_output: RoutingExpected
    metadata: CaseMetadata


class RoutingOutput(BaseModel):
    """执行失败保留状态，不伪装成有效路由决定。"""

    model_config = ConfigDict(extra="forbid")
    decision: RouteDecision | None = None
    execution_status: Literal["success", "invalid_model_output", "error"]
    duration_ms: float = Field(ge=0)
    error_type: str | None = None


async def evaluate_route(
    llm: BaseChatModel, client: Langfuse, question: RoutingInput
) -> RoutingOutput:
    """记录包含重试的路由操作；未知 Token 与费用不填写。"""
    call_state = RouteCallState()
    started = time.perf_counter()
    with client.start_as_current_observation(
        as_type="span",
        name="routing.invoke",
        input={
            "system_prompt": routing_prompt(),
            "resolved_question": question.resolved_question,
        },
        metadata={"model": settings.fast_model},
    ) as observation:
        try:
            decision = await decide_route(
                llm, question.resolved_question, call_state=call_state
            )
            output = RoutingOutput(
                decision=decision,
                execution_status=(
                    "invalid_model_output"
                    if call_state.invalid_model_output
                    else "success"
                ),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 评测需要记录所有单项执行失败，继续其他案例。
            # 记录失败后继续评其他案例，不重发整个案例，不上传异常中的敏感原文。
            output = RoutingOutput(
                execution_status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
            )
        observation.update(
            output=output.model_dump(mode="json"),
            level="DEFAULT" if output.execution_status == "success" else "ERROR",
        )
    return output


def execution_success(*, output: Any, **kwargs: Any) -> Evaluation:
    """无效模型输出与执行异常都计入执行失败。"""
    actual = RoutingOutput.model_validate(output)
    return Evaluation(
        name="execution_success",
        value=int(actual.execution_status == "success"),
        comment=actual.execution_status,
    )


def route_correct(*, output: Any, expected_output: Any, **kwargs: Any) -> Evaluation:
    """非成功案例计零，并单独保留失败原因以免被误当作普通误路由。"""
    actual = RoutingOutput.model_validate(output)
    expected = RoutingExpected.model_validate(expected_output)
    correct = (
        actual.execution_status == "success"
        and actual.decision is not None
        and actual.decision.route == expected.route
    )
    actual_route = (
        actual.decision.route if actual.decision is not None else "无有效决定"
    )
    return Evaluation(
        name="route_correct",
        value=int(correct),
        comment=(
            f"预期：{expected.route}；实际：{actual_route}；"
            f"状态：{actual.execution_status}"
        ),
    )


def decision_contract_valid(*, output: Any, **kwargs: Any) -> Evaluation:
    """RouteDecision 已执行契约校验，无效模型输出不能借原特殊分支通过。"""
    actual = RoutingOutput.model_validate(output)
    return Evaluation(
        name="decision_contract_valid",
        value=int(actual.execution_status == "success" and actual.decision is not None),
        comment="检查有效结构化结果及直接回复字段约束",
    )


class EvalSettings(BaseSettings):
    """沿用项目 .env 读取规则，评测缺少密钥时明确报错。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(), env_file_encoding="utf-8", extra="ignore"
    )
    langfuse_public_key: str = Field(min_length=1)
    langfuse_secret_key: str = Field(min_length=1)
    langfuse_base_url: str = "https://cloud.langfuse.com"


def load_cases(path: Path) -> list[RoutingCase]:
    """逐行校验，错误定位行号，禁止重复案例标识和静默跳过空行。"""
    cases: list[RoutingCase] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            case = RoutingCase.model_validate_json(line)
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False)
            raise ValueError(f"{path}:{line_number} 案例格式错误：{errors}") from None
        if case.metadata.case_id in identifiers:
            raise ValueError(
                f"{path}:{line_number} 案例标识重复：{case.metadata.case_id}"
            )
        identifiers.add(case.metadata.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} 没有评测案例")
    return cases


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """分别统计整体成功率、有效决定准确率、分类召回与执行失败。"""
    matrix: dict[str, Counter[str]] = {}
    statuses: Counter[str] = Counter()
    total_correct = 0
    durations: list[float] = []
    for row in rows:
        actual = RoutingOutput.model_validate(row["output"])
        expected = row["expected_output"]["route"]
        statuses[actual.execution_status] += 1
        durations.append(actual.duration_ms)
        label = (
            actual.decision.route
            if actual.execution_status == "success" and actual.decision is not None
            else actual.execution_status
        )
        matrix.setdefault(expected, Counter())[label] += 1
        total_correct += int(label == expected)
    successful = statuses["success"]
    return {
        "planned_runs": len(rows),
        "correct_runs": total_correct,
        "overall_success_rate": total_correct / len(rows),
        "execution_status_counts": dict(statuses),
        "valid_decision_accuracy": total_correct / successful if successful else None,
        "recall_by_route": {
            route: counts[route] / sum(counts.values())
            for route, counts in matrix.items()
        },
        "confusion_matrix": {route: dict(counts) for route, counts in matrix.items()},
        "mean_duration_ms": sum(durations) / len(durations),
        "token_usage": None,
        "cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="只运行路由模型的独立评测")
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("routing_cases.jsonl")
    )
    parser.add_argument("--dataset", default="crypto-agent-routing")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--system-certificates",
        action="store_true",
        help="使用操作系统证书信任，仅影响当前评测进程，仍然校验 HTTPS 证书",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/routing_eval"))
    args = parser.parse_args()
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        parser.error("重复次数和案例数量必须大于零")
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.validate_only:
        counts = Counter(case.expected_output.route for case in cases)
        print(f"已校验 {len(cases)} 个案例：{dict(counts)}")
        return

    if args.system_certificates:
        # 本机 Python 的证书集合不完整时，由用户显式选择系统信任，禁止关闭校验。
        truststore.inject_into_ssl()

    config = EvalSettings()
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    client = get_client(public_key=config.langfuse_public_key)
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "routing-%Y%m%dT%H%M%SZ"
    )
    output_path = args.output_dir / f"{run_name}.json"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    if output_path.exists():
        parser.error(f"结果文件已存在，请更换实验名称：{output_path}")
    if not settings.dashscope_api_key.strip():
        parser.error("未配置 DASHSCOPE_API_KEY")
    metadata = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(routing_prompt().encode("utf-8")).hexdigest(),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "working_tree_sha256": hashlib.sha256(
            (PROJECT_ROOT / "app" / "orchestration" / "routing.py").read_bytes()
        ).hexdigest(),
        "model": settings.fast_model,
        "repetitions": str(args.repetitions),
        "case_count": str(len(cases)),
        "model_timeout_seconds": "120",
        "max_attempts": "2",
        "system_certificates": str(args.system_certificates),
    }
    rows: list[dict[str, Any]] = []

    async def task(*, item: DatasetItem) -> dict[str, Any]:
        """每个实验项仅运行路由，全部明细由本次执行局部持有。"""
        question = RoutingInput.model_validate(item.input)
        # 使用显式独立连接，关闭时不影响 LangChain 缓存的业务客户端。
        async with model_session() as transport:
            llm = make_fast_llm(async_client=transport)
            output = await evaluate_route(llm, client, question)
        rows.append(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
                "output": output.model_dump(mode="json"),
            }
        )
        print(f"{item.metadata['case_id']}：{output.execution_status}", flush=True)
        return output.model_dump(mode="json")

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败，请检查密钥和服务地址")
        client.create_dataset(
            name=args.dataset,
            description="已改写问题的三分支独立路由评测",
            input_schema=RoutingInput.model_json_schema(),
        )
        items = [
            client.create_dataset_item(
                dataset_name=args.dataset,
                id=str(uuid5(NAMESPACE_URL, f"{args.dataset}:{case.metadata.case_id}")),
                input=case.input.model_dump(),
                expected_output=case.expected_output.model_dump(),
                metadata=case.metadata.model_dump(),
            )
            for case in cases
        ]
        print(
            f"开始 {len(items) * args.repetitions} 次路由评测，最多两次模型尝试/项",
            flush=True,
        )
        for repetition in range(1, args.repetitions + 1):
            row_start = len(rows)
            result = client.run_experiment(
                name="crypto-agent-routing",
                run_name=f"{run_name}-r{repetition}",
                data=items,
                task=task,
                evaluators=[execution_success, route_correct, decision_contract_valid],
                max_concurrency=1,
                metadata=metadata,
            )
            for row in rows[row_start:]:
                row["repetition"] = repetition
            for row, item_result in zip(
                rows[row_start:], result.item_results, strict=True
            ):
                if len(item_result.evaluations) != 3:
                    raise RuntimeError("评分记录不完整，不能输出完整实验结果")
                row["scores"] = [score.__dict__ for score in item_result.evaluations]
                row["langfuse_trace_id"] = item_result.trace_id
            print(result.format())
        if len(rows) != len(items) * args.repetitions:
            raise RuntimeError("实验执行记录不完整，不能输出完整成功率")
        report = {
            "run_name": run_name,
            "metadata": metadata,
            "summary": summarize(rows),
            "cases": rows,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"本地结果：{output_path.resolve()}")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
