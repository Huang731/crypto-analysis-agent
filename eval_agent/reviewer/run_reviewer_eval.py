"""固定候选消息调用生产 reviewer，仅评测 RAGAS 忠实度和答案相关性。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import truststore
from langfuse import Langfuse
from pydantic import BaseModel, ConfigDict, model_validator

from app.config.config import PROJECT_ROOT, settings
from app.crypto_analysis.crypto_analysis_parent_agent import (
    ReportReview,
    _invoke_report_review,
)
from app.llm.model import make_glm_smart_llm, model_session
from eval_agent.prompts.report_eval import reviewer_evaluation_question
from eval_agent.writer.run_writer_eval import (
    EvalSettings,
    WriterInput,
    WriterMetadata,
    content_sha256,
    create_metrics,
    decode_section,
    error_message,
    score_case,
    summarize,
)


class ReviewerCase(BaseModel):
    """保留候选输入、人工参考和来源；参考结果不参与两项指标评分。"""

    model_config = ConfigDict(extra="forbid")
    input: WriterInput
    expected_output: ReportReview
    metadata: WriterMetadata

    @model_validator(mode="after")
    def validate_snapshot(self) -> ReviewerCase:
        if (
            content_sha256(self.input.model_dump(mode="json"))
            != self.metadata.input_sha256
        ):
            raise ValueError("案例输入与快照哈希不一致")
        return self


def build_evidence(case: ReviewerCase) -> dict[str, Any]:
    """审核原文与本轮证据分别标明，不使用历史摘要或人工参考。"""
    human = case.input.root[1].data.content
    scope, payload = human.split("## 任务上下文", 1)[1].split(
        "## 本轮证据（技能表格及对应思想原文）", 1
    )
    asset, question = scope.split("- **目标资产**：", 1)[1].split("- **用户问题**：", 1)
    question = question.split("- **此前结论摘要**：", 1)[0].strip()
    materials, report = payload.split("## 待审核报告", 1)
    skills = decode_section(materials)
    contexts = [
        json.dumps(
            {
                "用户问题": question,
                "待核查的报告原文（不是行情事实依据）": report.strip(),
            },
            ensure_ascii=False,
        )
    ]
    for skill in skills:
        thought = skill["thought_context"]
        contexts.append(
            json.dumps(
                {
                    "skill_name": skill["skill_name"],
                    "本轮指标证据": skill["content"],
                    "思想检索状态": thought["status"],
                    "思想原文": [
                        {
                            "content": doc["content"],
                            "source": doc["metadata"].get("source"),
                            "section": doc["metadata"].get("section"),
                        }
                        for doc in thought["documents"]
                    ],
                },
                ensure_ascii=False,
            )
        )
    if any("USDT与USDC合计市值" in skill["content"] for skill in skills):
        contexts.append("本项目中，稳定币总市值是USDT与USDC合计市值的简称。")
    return {
        "user_input": reviewer_evaluation_question(
            asset.strip(), question, report.strip()
        ),
        "retrieved_contexts": contexts,
    }


def load_cases(path: Path, check_sources: bool = False) -> list[ReviewerCase]:
    cases = [
        ReviewerCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.metadata.case_id for case in cases}) != len(cases):
        raise ValueError("案例为空或案例 ID 重复")
    for case in cases:
        build_evidence(case)
        if not check_sources:
            continue
        metadata = case.metadata
        source = json.loads(
            (PROJECT_ROOT / metadata.source_candidate_file)
            .read_text(encoding="utf-8")
            .splitlines()[metadata.source_candidate_line - 1]
        )
        if content_sha256(source) != metadata.source_candidate_sha256:
            raise ValueError(f"{metadata.case_id} 的候选来源行发生变化")
        messages = source["input"]
        for edit in metadata.edits:
            text = messages[edit.message_index]["data"]["content"]
            if text.count(edit.before) != 1:
                raise ValueError(f"{metadata.case_id} 的来源替换不能唯一定位")
            messages[edit.message_index]["data"]["content"] = text.replace(
                edit.before, edit.after, 1
            )
        if messages != case.input.model_dump(mode="json"):
            raise ValueError(f"{metadata.case_id} 的输入不符合来源与替换记录")
    return cases


async def evaluate(
    client: Langfuse, case: ReviewerCase, judge_model: str
) -> dict[str, Any]:
    """模型或指标失败时保留空分数，不再生成额外评分项。"""
    evidence = build_evidence(case)
    output: dict[str, Any] = {
        "actual_output": None,
        "ragas_input": evidence,
        "faithfulness": None,
        "answer_relevancy": None,
        "errors": {},
    }
    with client.start_as_current_observation(
        as_type="span",
        name="reviewer.evaluate",
        input=case.input.model_dump(mode="json"),
    ) as observation:
        try:
            async with model_session() as http:
                review = await asyncio.wait_for(
                    _invoke_report_review(
                        make_glm_smart_llm(async_client=http), case.input.to_messages()
                    ),
                    timeout=250,
                )
            output["actual_output"] = review.model_dump()
        except Exception as exc:  # noqa: BLE001 - 单个案例失败不阻断其余案例。
            output["errors"]["target"] = error_message(exc)
            observation.update(output=output, level="ERROR")
            return output

        async with model_session() as http:
            for name, metric in create_metrics(judge_model, http=http).items():
                arguments = {
                    "user_input": evidence["user_input"],
                    "response": review.model_dump_json(),
                }
                if name == "faithfulness":
                    arguments["retrieved_contexts"] = evidence["retrieved_contexts"]
                with client.start_as_current_observation(
                    as_type="span", name=f"reviewer.ragas.{name}", input=arguments
                ) as metric_observation:
                    try:
                        result = await asyncio.wait_for(
                            metric.ascore(**arguments), timeout=600
                        )
                        if math.isfinite(result.value):
                            output[name] = result.value
                        metric_observation.update(output={"value": output[name]})
                    except Exception as exc:  # noqa: BLE001 - 保留另一项已完成的得分。
                        output["errors"][name] = error_message(exc)
                        metric_observation.update(
                            output={"error": output["errors"][name]}, level="ERROR"
                        )
        observation.update(
            output=output, level="ERROR" if output["errors"] else "DEFAULT"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="固定候选输入的 reviewer RAGAS 评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval_agent/reviewer/reviewer_cases.jsonl",
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
    cases = load_cases(args.cases, args.check_sources)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("案例数量必须大于零")
        cases = cases[: args.limit]
    if args.validate_only:
        print(f"已校验 {len(cases)} 个 reviewer 案例及 RAGAS 输入")
        return
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "reviewer-ragas-%Y%m%dT%H%M%SZ"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    path = PROJECT_ROOT / "outputs/reviewer_eval" / f"{run_name}.json"
    if path.exists():
        parser.error("结果文件已存在，请使用新的实验名称")
    if not settings.dashscope_api_key.strip():
        parser.error("未配置 DASHSCOPE_API_KEY")
    if args.system_certificates:
        truststore.inject_into_ssl()
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    import_module("ragas.metrics.collections")
    config = EvalSettings()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    rows: list[dict[str, Any]] = []
    run_status = "failed"

    async def task(*, item: Any) -> dict[str, Any]:
        case = ReviewerCase.model_validate(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
            }
        )
        output = await evaluate(client, case, args.judge_model)
        rows.append({**case.model_dump(mode="json"), "output": output})
        print(
            f"{case.metadata.case_id}：faithfulness={output['faithfulness']}，answer_relevancy={output['answer_relevancy']}",
            flush=True,
        )
        return output

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败")
        dataset = "crypto-agent-reviewer"
        client.create_dataset(
            name=dataset,
            description="固定候选输入的 reviewer RAGAS 忠实度与答案相关性评测",
            input_schema=WriterInput.model_json_schema(),
            expected_output_schema=ReportReview.model_json_schema(),
        )
        items = [
            client.create_dataset_item(
                dataset_name=dataset,
                id=str(uuid5(NAMESPACE_URL, f"{dataset}:{case.metadata.case_id}")),
                input=case.input.model_dump(mode="json"),
                expected_output=case.expected_output.model_dump(),
                metadata=case.metadata.model_dump(mode="json"),
            )
            for case in cases
        ]
        experiment = client.run_experiment(
            name=dataset,
            run_name=run_name,
            data=items,
            task=task,
            evaluators=[score_case],
            max_concurrency=1,
            metadata={
                "metrics": ["faithfulness", "answer_relevancy"],
                "target_model": settings.glm_smart_model,
                "judge_model": args.judge_model,
                "embedding_model": settings.embedding_model,
                "ragas_version": version("ragas"),
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            },
        )
        for row, result in zip(rows, experiment.item_results, strict=True):
            row["langfuse_trace_id"] = result.trace_id
        if len(rows) != len(cases):
            raise RuntimeError("评测记录不完整")
        run_status = "completed"
        print(experiment.format())
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "run_status": run_status,
                    "summary": summarize(rows),
                    "cases": rows,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        client.flush()
        print(f"本地结果：{path}")


if __name__ == "__main__":
    main()
