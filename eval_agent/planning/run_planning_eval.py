"""独立调用业务规划函数评分，并保存 Langfuse 实验。"""

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
from langchain_core.messages import messages_to_dict
from langfuse import Evaluation, Langfuse, get_client
from langfuse.api import DatasetItem
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.config import PROJECT_ROOT, _env_files, settings
from app.crypto_analysis.crypto_analysis_parent_agent import (
    AnalysisPlan,
    create_analysis_plan,
)
from app.crypto_analysis.skill_loader import get_skill_loader
from app.llm.model import make_glm_4_flash_llm, model_session
from prompts.crypto_analysis_plan import analysis_plan_messages


class PlanningInput(BaseModel):
    """固定已改写问题，不调用路由或历史问题改写。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    resolved_question: str = Field(min_length=1)


class CaseMetadata(BaseModel):
    """人工标注案例的标识、分类和判定理由。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    label_reason: str = Field(min_length=1)


class PlanningCase(BaseModel):
    """一行 JSONL 包含问题、人工计划和标注信息。"""

    model_config = ConfigDict(extra="forbid")
    input: PlanningInput
    expected_output: AnalysisPlan
    metadata: CaseMetadata


class PlanningOutput(BaseModel):
    """保留规划失败，不补造默认资产或技能。"""

    model_config = ConfigDict(extra="forbid")
    plan: AnalysisPlan | None = None
    execution_status: Literal["success", "error"]
    duration_ms: float = Field(ge=0)
    error_type: str | None = None


class EvalSettings(BaseSettings):
    """沿用项目环境变量，未配置 Langfuse 密钥时明确报错。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(), env_file_encoding="utf-8", extra="ignore"
    )
    langfuse_public_key: str = Field(min_length=1)
    langfuse_secret_key: str = Field(min_length=1)
    langfuse_base_url: str = "https://cloud.langfuse.com"


def load_cases(path: Path, catalog_names: list[str]) -> list[PlanningCase]:
    """校验 JSONL，并拒绝未知技能、错误目录顺序和重复案例标识。"""
    cases: list[PlanningCase] = []
    identifiers: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            case = PlanningCase.model_validate_json(line)
        except ValidationError as exc:
            errors = exc.errors(include_input=False, include_url=False)
            raise ValueError(f"{path}:{number} 案例格式错误：{errors}") from None
        if case.metadata.case_id in identifiers:
            raise ValueError(f"{path}:{number} 案例标识重复：{case.metadata.case_id}")
        names = case.expected_output.skill_names
        unknown = set(names) - set(catalog_names)
        if unknown:
            raise ValueError(f"{path}:{number} 标注了未知技能：{sorted(unknown)}")
        if names != [name for name in catalog_names if name in names]:
            raise ValueError(f"{path}:{number} 预期技能没有按目录顺序排列")
        identifiers.add(case.metadata.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} 没有评测案例")
    return cases


async def evaluate_plan(
    llm: BaseChatModel,
    client: Langfuse,
    question: PlanningInput,
    skill_descriptions: str,
) -> PlanningOutput:
    """直接调用生产规划函数，沿用其重试和模型工厂的超时。"""
    started = time.perf_counter()
    with client.start_as_current_observation(
        as_type="span",
        name="planning.invoke",
        input={
            "resolved_question": question.resolved_question,
            "messages": messages_to_dict(
                analysis_plan_messages(skill_descriptions, question.resolved_question)
            ),
        },
        metadata={"model": settings.glm_4_flash_model},
    ) as observation:
        try:
            result = await create_analysis_plan(
                llm, question.resolved_question, skill_descriptions
            )
            output = PlanningOutput(
                plan=AnalysisPlan.model_validate(result),
                execution_status="success",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - 单项失败需保留记录，继续其他评测项。
            output = PlanningOutput(
                execution_status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
            )
        observation.update(
            output=output.model_dump(mode="json"),
            level="DEFAULT" if output.execution_status == "success" else "ERROR",
        )
    return output


def compare_plan(actual: PlanningOutput, expected: AnalysisPlan) -> dict[str, bool]:
    """技能按完整列表比较，漏选、多选和顺序错误均不通过。"""
    success = actual.execution_status == "success" and actual.plan is not None
    scores = {
        "execution_success": success,
        "target_asset_correct": success
        and actual.plan.target_asset == expected.target_asset,
        "skill_selection_correct": success
        and actual.plan.skill_names == expected.skill_names,
        "refresh_correct": success and actual.plan.refresh == expected.refresh,
    }
    scores["plan_correct"] = all(scores.values())
    return scores


def score_plan(*, output: Any, expected_output: Any, **kwargs: Any) -> list[Evaluation]:
    """一次生成五项确定性评分，不调用裁判模型。"""
    actual = PlanningOutput.model_validate(output)
    expected = AnalysisPlan.model_validate(expected_output)
    actual_names = actual.plan.skill_names if actual.plan is not None else []
    details = {
        "execution_status": actual.execution_status,
        "expected": expected.model_dump(),
        "actual": actual.plan.model_dump() if actual.plan is not None else None,
        "missing_skills": [
            name for name in expected.skill_names if name not in actual_names
        ],
        "extra_skills": [
            name for name in actual_names if name not in expected.skill_names
        ],
    }
    comment = json.dumps(details, ensure_ascii=False)
    return [
        Evaluation(name=name, value=int(correct), comment=comment)
        for name, correct in compare_plan(actual, expected).items()
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """整体正确率包含失败项，同时保留字段表现和分类结果。"""
    scores: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    categories: dict[str, Counter[str]] = {}
    durations: list[float] = []
    for row in rows:
        actual = PlanningOutput.model_validate(row["output"])
        expected = AnalysisPlan.model_validate(row["expected_output"])
        comparison = compare_plan(actual, expected)
        scores.update({name: int(correct) for name, correct in comparison.items()})
        statuses[actual.execution_status] += 1
        counts = categories.setdefault(row["metadata"]["category"], Counter())
        counts["runs"] += 1
        counts["correct"] += int(comparison["plan_correct"])
        durations.append(actual.duration_ms)
    total = len(rows)
    successful = statuses["success"]
    return {
        "planned_runs": total,
        "correct_runs": scores["plan_correct"],
        "overall_success_rate": scores["plan_correct"] / total,
        "accuracy_by_metric": {name: count / total for name, count in scores.items()},
        "execution_status_counts": dict(statuses),
        "valid_plan_accuracy": (
            scores["plan_correct"] / successful if successful else None
        ),
        "accuracy_by_category": {
            category: counts["correct"] / counts["runs"]
            for category, counts in categories.items()
        },
        "mean_duration_ms": sum(durations) / total,
        "token_usage": None,
        "cost": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="只运行技能规划模型的独立评测")
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("planning_cases.jsonl")
    )
    parser.add_argument("--dataset", default="crypto-agent-planning")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--system-certificates",
        action="store_true",
        help="使用操作系统证书信任，仅影响当前进程，仍然校验 HTTPS 证书",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/planning_eval")
    )
    args = parser.parse_args()
    if args.repetitions < 1 or (args.limit is not None and args.limit < 1):
        parser.error("重复次数和案例数量必须大于零")
    loader = get_skill_loader()
    catalog_names = [skill.name for skill in loader.list_skills()]
    skill_descriptions = loader.format_skill_descriptions()
    cases = load_cases(args.cases, catalog_names)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.validate_only:
        counts = Counter(case.metadata.category for case in cases)
        print(
            f"已校验 {len(cases)} 个案例、{len(catalog_names)} 个技能：{dict(counts)}"
        )
        return

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "planning-%Y%m%dT%H%M%SZ"
    )
    output_path = args.output_dir / f"{run_name}.json"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    if output_path.exists():
        parser.error(f"结果文件已存在，请更换实验名称：{output_path}")
    if not settings.dashscope_api_key.strip() or settings.dashscope_api_key.startswith(
        "xxxx"
    ):
        parser.error("未配置 DASHSCOPE_API_KEY")
    if args.system_certificates:
        truststore.inject_into_ssl()
    config = EvalSettings()
    metadata = {
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            json.dumps(
                messages_to_dict(
                    analysis_plan_messages(skill_descriptions, "{resolved_question}")
                ),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "skill_catalog_sha256": hashlib.sha256(
            skill_descriptions.encode("utf-8")
        ).hexdigest(),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "working_tree_sha256": hashlib.sha256(
            (
                PROJECT_ROOT / "app/crypto_analysis/crypto_analysis_parent_agent.py"
            ).read_bytes()
        ).hexdigest(),
        "model": settings.glm_4_flash_model,
        "repetitions": str(args.repetitions),
        "case_count": str(len(cases)),
        "model_timeout_seconds": "120",
        "max_attempts": "2",
        "system_certificates": str(args.system_certificates),
    }
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    client = get_client(public_key=config.langfuse_public_key)
    rows: list[dict[str, Any]] = []

    async def task(*, item: DatasetItem) -> dict[str, Any]:
        """每次只调用生产规划函数，模型连接由本项独立持有。"""
        question = PlanningInput.model_validate(item.input)
        async with model_session() as transport:
            llm = make_glm_4_flash_llm(async_client=transport)
            output = await evaluate_plan(llm, client, question, skill_descriptions)
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
            description="固定问题与技能目录的独立规划评测",
            input_schema=PlanningInput.model_json_schema(),
            expected_output_schema=AnalysisPlan.model_json_schema(),
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
        print(f"开始 {len(items) * args.repetitions} 次技能规划评测", flush=True)
        for repetition in range(1, args.repetitions + 1):
            row_start = len(rows)
            result = client.run_experiment(
                name="crypto-agent-planning",
                run_name=f"{run_name}-r{repetition}",
                data=items,
                task=task,
                evaluators=[score_plan],
                max_concurrency=1,
                metadata=metadata,
            )
            for row, item_result in zip(
                rows[row_start:], result.item_results, strict=True
            ):
                if len(item_result.evaluations) != 5:
                    raise RuntimeError("评分记录不完整，不能输出完整实验结果")
                row["repetition"] = repetition
                row["scores"] = [score.__dict__ for score in item_result.evaluations]
                row["langfuse_trace_id"] = item_result.trace_id
            print(result.format())
        if len(rows) != len(items) * args.repetitions:
            raise RuntimeError("执行记录不完整，不能输出完整成功率")
        report = {
            "run_name": run_name,
            "metadata": metadata,
            "skill_descriptions": skill_descriptions,
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
