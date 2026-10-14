"""直接调用生产正文生成函数，使用代码及独立 LLM 裁判评分。"""

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import httpx
import truststore
from langchain_core.callbacks import AsyncCallbackHandler
from langfuse import Evaluation, Langfuse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config.config import PROJECT_ROOT, settings
from app.crypto_analysis.crypto_analysis_agent import generate_skill
from app.crypto_analysis.metric_service import _log_reason, metric_field_labels
from app.crypto_analysis.skill_loader import SkillLoader
from app.llm.model import _make_model
from eval_agent.planning.run_planning_eval import CaseMetadata, EvalSettings
from eval_agent.skill_execution.run_skill_execution_eval import (
    SkillExecutionResult,
    duration_summary,
    values_equal,
)
from prompts.skill_generation_eval import skill_generation_judge_prompt


class DocumentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any]


class ThoughtContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    status: Literal["ok", "no_results", "error"]
    documents: list[DocumentEvidence]

    @model_validator(mode="after")
    def valid_documents(self) -> "ThoughtContext":
        if (self.status == "ok") != bool(self.documents):
            raise ValueError("检索状态与固定材料不一致")
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("思想材料 ID 重复")
        if any(not doc.metadata.get("section") for doc in self.documents):
            raise ValueError("思想材料必须有来源章节")
        return self


class GenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["generation", "response_replay", "judge_calibration"] = "generation"
    skill_source: str
    skill_name: str
    analysis_subject: str
    user_input: str
    result: SkillExecutionResult
    thought_context: ThoughtContext
    fixed_response: str | None = None
    source_generation_duration_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_input(self) -> "GenerationInput":
        loader = SkillLoader.from_snapshot({self.skill_name: self.skill_source})
        skill = loader.load_definition(self.skill_name)
        if self.result.skill_name != self.skill_name:
            raise ValueError("指标所属技能不一致")
        if [item.indicator_id for item in self.result.indicators] != [
            metric.id for metric in skill.scoring.metrics
        ]:
            raise ValueError("固定指标与 Skill 配置不一致")
        if (self.mode != "generation") != (self.fixed_response is not None):
            raise ValueError("固定正文回放与校准必须提供回答，直接生成不能提供固定回答")
        if (
            self.mode == "response_replay"
            and self.source_generation_duration_ms is None
        ):
            raise ValueError("正文回放必须保留原生成耗时")
        return self


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skipped: bool = False
    faithfulness_min: float = Field(default=1, ge=0, le=1)
    faithfulness_max: float = Field(default=1, ge=0, le=1)
    answer_relevancy_min: float = Field(default=0.75, ge=0, le=1)
    answer_relevancy_max: float = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def valid_ranges(self) -> "Expected":
        if self.faithfulness_min > self.faithfulness_max:
            raise ValueError("忠实度区间无效")
        if self.answer_relevancy_min > self.answer_relevancy_max:
            raise ValueError("相关性区间无效")
        return self


class GenerationCaseMetadata(CaseMetadata):
    """保留计算评测的真实输出来源及固定思想材料的来源方式。"""

    source_execution_report: str | None = None
    source_execution_case_id: str | None = None
    source_execution_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    thought_context_origin: (
        Literal["fixed_local_knowledge_chunk", "no_documents"] | None
    ) = None

    @model_validator(mode="after")
    def complete_source(self) -> "GenerationCaseMetadata":
        source = (
            self.source_execution_report,
            self.source_execution_case_id,
            self.source_execution_sha256,
        )
        if any(source) and not all(source):
            raise ValueError("计算输出来源必须同时提供文件、案例标识和文件哈希")
        return self


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: GenerationInput
    expected_output: Expected
    metadata: GenerationCaseMetadata


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_id: str = Field(min_length=1)
    supported: bool
    evidence_ids: list[str]
    reason: str = Field(min_length=1)


class Relevance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_match: int = Field(ge=0, le=2)
    coverage: int = Field(ge=0, le=2)
    interpretation_focus: int = Field(ge=0, le=2)
    scope: int = Field(ge=0, le=2)


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_mappings: dict[str, dict[str, str]]
    claims: list[Claim] = Field(min_length=1)
    relevance: Relevance
    relevance_reason: str


class CallCounter(AsyncCallbackHandler):
    """每次案例独立记录真实调用及供应商返回的 Token 用量。"""

    def __init__(self) -> None:
        self.attempts = 0
        self.usage: list[dict[str, Any]] = []
        self.last_response: Any = None

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.attempts += 1

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for batch in response.generations:
            for generation in batch:
                self.last_response = generation.message.content
                usage = generation.message.usage_metadata
                if usage:
                    self.usage.append(dict(usage))


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not cases or len({c.metadata.case_id for c in cases}) != len(cases):
        raise ValueError("案例为空或 ID 重复")
    reports: dict[str, tuple[str, dict[str, Any]]] = {}
    for case in cases:
        source = case.metadata
        if source.source_execution_report is None:
            continue
        if source.source_execution_report not in reports:
            report_path = Path(source.source_execution_report)
            if not report_path.is_absolute():
                report_path = PROJECT_ROOT / report_path
            data = report_path.read_bytes()
            report = json.loads(data)
            reports[source.source_execution_report] = (
                hashlib.sha256(data).hexdigest(),
                {row["metadata"]["case_id"]: row for row in report["cases"]},
            )
        checksum, rows = reports[source.source_execution_report]
        if checksum != source.source_execution_sha256:
            raise ValueError("计算评测文件已改变，案例来源哈希不匹配")
        if source.source_execution_case_id not in rows:
            raise ValueError("计算评测文件中不存在标注的来源案例")
        row = rows[source.source_execution_case_id]
        if row["diagnostics"]["execution_status"] != "success" or not values_equal(
            case.input.result.model_dump(mode="json"), row["output"]
        ):
            raise ValueError("生成案例必须使用计算评测实际成功返回的完整output")
        definition = SkillLoader.from_snapshot(
            {case.input.skill_name: case.input.skill_source}
        ).load_definition(case.input.skill_name)
        if not values_equal(definition.model_dump(mode="json"), row["input"]["skill"]):
            raise ValueError("生成案例Skill配置与来源计算案例不一致")
    return cases


def split_cells(line: str) -> list[str]:
    """按 Markdown 转义规则拆分两列表格，不用正则猜测模型输出。"""
    cells: list[str] = []
    cell = ""
    escaped = False
    for char in line.strip().strip("|"):
        if escaped:
            cell += char
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(cell.strip())
            cell = ""
        else:
            cell += char
    cells.append(cell.strip())
    return cells


def parse_tables(content: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """依据固定指标标识行识别表格，拒绝重复行和拆分后的重复指标表。"""
    tables: dict[str, dict[str, str]] = {}
    thought: dict[str, str] = {}
    rows: dict[str, str] = {}

    def finish() -> None:
        if not rows:
            return
        if "指标标识" in rows:
            metric_id = rows["指标标识"].strip("`* ")
            if metric_id in tables:
                raise ValueError(f"指标表重复：{metric_id}")
            tables[metric_id] = dict(rows)
        elif "来源章节" in rows or "当前解读" in rows:
            if thought:
                raise ValueError("思想解读表重复")
            thought.update(rows)
        else:
            raise ValueError("出现未声明的额外表格")
        rows.clear()

    for line in content.splitlines():
        if not line.strip().startswith("|"):
            finish()
            continue
        cells = split_cells(line)
        if len(cells) != 2:
            raise ValueError("指标表必须是两列，内容中的竖线应转义")
        name, value = cells
        if name == "字段" or set(name) <= set("-: "):
            continue
        if name in rows:
            raise ValueError(f"表格字段重复：{name}")
        rows[name] = value
    finish()
    return tables, thought


def value_matches(actual: str, expected: Any, *, allow_percent: bool = False) -> bool:
    """保留日期、精度与有效零分，支持完整列表和 Markdown 链接展示。"""
    actual = actual.strip("`* ")
    if expected is None:
        return actual == "N/A"
    if isinstance(expected, bool):
        return actual.lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            if allow_percent:
                actual = actual.removesuffix("%")
            return float(actual.replace(",", "")) == expected
        except ValueError:
            return False
    if isinstance(expected, (list, dict)):
        try:
            return json.loads(actual) == expected
        except json.JSONDecodeError:
            if isinstance(expected, list):
                values = actual.replace("<br/>", "<br>").replace("<br />", "<br>")
                parts = values.split("<br>")
                return len(parts) == len(expected) and all(
                    value_matches(a, e) for a, e in zip(parts, expected, strict=True)
                )
            return False
    return actual == expected or (
        str(expected).startswith("https://") and actual.endswith(f"]({expected})")
    )


def response_statements(content: str, thought: dict[str, str]) -> dict[str, str]:
    """固定待评正文句子，避免裁判把输入证据或改写内容当成回答。"""
    passages = [thought.get("当前解读", "")]
    passages.extend(
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith(("|", "#"))
    )
    statements: list[str] = []
    for passage in passages:
        passage = passage.replace("<br/>", "\n").replace("<br />", "\n")
        passage = passage.replace("<br>", "\n")
        current = ""
        for character in passage:
            current += character
            if character in "。！？；\n":
                if current.strip():
                    statements.append(current.strip())
                current = ""
        if current.strip():
            statements.append(current.strip())
    return {f"R{index}": value for index, value in enumerate(statements, start=1)}


def required_fields(inputs: GenerationInput) -> dict[str, dict[str, Any]]:
    if inputs.mode == "judge_calibration":
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in inputs.result.indicators:
        if item.status != "ok":
            continue
        values = item.model_dump()
        fields = {
            key: values[key]
            for key in ("indicator_id", "status", "score", "unit", "observed_at", "url")
        }
        for group in ("raw_values", "calculated_values"):
            fields.update(
                {f"{group}.{key}": value for key, value in values[group].items()}
            )
        result[item.indicator_id] = fields
    return result


def indicator_table_checks(
    inputs: GenerationInput, tables: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """只核对固定中文字段和原始数值，不生成思想解读质量分数。"""
    fixed_labels = {
        "indicator_id": "指标标识",
        "status": "指标状态",
        "score": "指标分数",
        "unit": "单位",
        "observed_at": "观测日期",
        "url": "来源链接",
    }
    checks = []
    for metric_id, fields in required_fields(inputs).items():
        labels = metric_field_labels(metric_id)
        for path, expected in fields.items():
            label = labels[path.split(".", 1)[1]] if "." in path else fixed_labels[path]
            actual = tables.get(metric_id, {}).get(label)
            checks.append(
                {
                    "indicator_id": metric_id,
                    "field": path,
                    "row_label": label,
                    "expected": expected,
                    "actual": actual,
                    "correct": actual is not None
                    and value_matches(
                        actual, expected, allow_percent=path.endswith("_pct")
                    ),
                }
            )
    return checks


async def evaluate(
    client: Langfuse, inputs: GenerationInput, judge_model: str
) -> dict[str, Any]:
    generation_counter, judge_counter = CallCounter(), CallCounter()
    active_stage: str | None = None
    started = 0.0
    output: dict[str, Any] = {
        "execution_status": "error",
        "response": None,
        "judgment": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "field_checks": [],
        "indicator_contract_valid": False,
        "section_citations_valid": False,
        "skipped": False,
        "generation_duration_ms": 0.0,
        "judge_duration_ms": 0.0,
        "error_type": None,
    }
    try:
        async with AsyncExitStack() as stack:
            generation_http = await stack.enter_async_context(
                httpx.AsyncClient(timeout=120)
            )
            judge_http = await stack.enter_async_context(httpx.AsyncClient(timeout=120))
            generator = _make_model(
                settings.smart_model, http_async_client=generation_http
            )
            judge = _make_model(judge_model, http_async_client=judge_http)
            loader = SkillLoader.from_snapshot({inputs.skill_name: inputs.skill_source})
            skill = loader.load_definition(inputs.skill_name)
            result = inputs.result.model_dump()
            result["thought_context"] = inputs.thought_context.model_dump()
            result["scope"] = {"resolved_question": inputs.user_input}
            with client.start_as_current_observation(
                as_type="span",
                name="skill_generation.invoke",
                input=inputs.model_dump(mode="json"),
            ) as observation:
                started = time.perf_counter()
                active_stage = "generation"
                if inputs.mode == "generation":
                    response = await asyncio.wait_for(
                        generate_skill(
                            generator.with_config(callbacks=[generation_counter]),
                            loader,
                            skill,
                            result,
                            inputs.analysis_subject,
                        ),
                        timeout=300,
                    )
                else:
                    response = inputs.fixed_response
                output["generation_duration_ms"] = (
                    time.perf_counter() - started
                ) * 1000
                output["response"] = response
                active_stage = None
                if response == "":
                    output.update(
                        execution_status="success",
                        skipped=True,
                        indicator_contract_valid=not required_fields(inputs),
                        section_citations_valid=not inputs.thought_context.documents,
                    )
                    if inputs.mode == "generation":
                        output.update(
                            assessment_scope="indicator_tables",
                            thought_output_absent=True,
                            section_citations_valid=None,
                        )
                else:
                    tables, thought = parse_tables(response)
                    required = required_fields(inputs)
                    statements = response_statements(response, thought)
                    if inputs.mode == "generation" or (not thought and not statements):
                        checks = indicator_table_checks(inputs, tables)
                        thought_absent = (
                            not thought
                            and not statements
                            and not any(
                                line.strip().startswith("#") and "思想" in line
                                for line in response.splitlines()
                            )
                        )
                        output.update(
                            execution_status="success",
                            assessment_scope="indicator_tables",
                            field_checks=checks,
                            indicator_contract_valid=(
                                set(tables) == set(required)
                                and all(check["correct"] for check in checks)
                                and all(
                                    set(table)
                                    == {
                                        check["row_label"]
                                        for check in checks
                                        if check["indicator_id"] == metric_id
                                    }
                                    for metric_id, table in tables.items()
                                )
                            ),
                            thought_output_absent=thought_absent,
                            section_citations_valid=None,
                        )
                        observation.update(output=output)
                        output["generation_model_attempts"] = (
                            generation_counter.attempts
                        )
                        output["judge_model_attempts"] = judge_counter.attempts
                        output["token_usage"] = {
                            "generation": generation_counter.usage,
                            "judge": judge_counter.usage,
                        }
                        return output
                    if not statements:
                        raise ValueError("非空正文未提供待评解读")
                    output["response_statements"] = statements
                    contexts = {
                        "E0": [
                            item
                            for item in result["indicators"]
                            if item["status"] == "ok"
                        ],
                        "E1": {
                            "content": json.dumps(
                                {
                                    "query": inputs.thought_context.query,
                                    "status": inputs.thought_context.status,
                                    "documents": [
                                        doc.id
                                        for doc in inputs.thought_context.documents
                                    ],
                                },
                                ensure_ascii=False,
                            )
                        },
                    }
                    contexts.update(
                        {
                            f"E{index}": {
                                "content": doc.content,
                                "metadata": doc.metadata,
                            }
                            for index, doc in enumerate(
                                inputs.thought_context.documents, start=2
                            )
                        }
                    )

                    class GroundedJudgment(Judgment):
                        """核对待评句子完整性及证据编号，事实支持由裁判判断。"""

                        @model_validator(mode="after")
                        def validate_judgment(self) -> "GroundedJudgment":
                            ids = [claim.response_id for claim in self.claims]
                            if len(ids) != len(set(ids)) or set(ids) != set(statements):
                                raise ValueError(
                                    "裁判必须逐一评估全部正文句子，不能增删重复"
                                )
                            if set(self.field_mappings) != set(required) or any(
                                set(self.field_mappings[metric_id]) != set(fields)
                                for metric_id, fields in required.items()
                            ):
                                raise ValueError("裁判字段映射必须覆盖全部要求字段")
                            if any(
                                label and label not in tables.get(metric_id, {})
                                for metric_id, mapping in self.field_mappings.items()
                                for label in mapping.values()
                            ):
                                raise ValueError("裁判映射行名必须逐字来自原表格")
                            for claim in self.claims:
                                ids = set(claim.evidence_ids)
                                if not ids <= contexts.keys() or (
                                    claim.supported and not ids
                                ):
                                    raise ValueError("裁判未引用有效证据")
                            return self

                    schema = GroundedJudgment.model_json_schema()
                    schema["properties"]["field_mappings"] = {
                        "type": "object",
                        "properties": {
                            metric_id: {
                                "type": "object",
                                "properties": {
                                    path: {
                                        "type": "string",
                                        "enum": ["", *tables.get(metric_id, {})],
                                    }
                                    for path in fields
                                },
                                "required": list(fields),
                                "additionalProperties": False,
                            }
                            for metric_id, fields in required.items()
                        },
                        "required": list(required),
                        "additionalProperties": False,
                    }
                    schema["$defs"]["Claim"]["properties"]["response_id"]["enum"] = (
                        list(statements)
                    )
                    schema["$defs"]["Claim"]["properties"]["evidence_ids"]["items"][
                        "enum"
                    ] = list(contexts)
                    schema["$defs"]["Claim"]["allOf"] = [
                        {
                            "if": {"properties": {"supported": {"const": True}}},
                            "then": {"properties": {"evidence_ids": {"minItems": 1}}},
                        }
                    ]
                    output["evidence_references"] = {
                        "E0": "成功指标",
                        "E1": "检索状态与思想材料ID",
                        **{
                            f"E{index}": doc.id
                            for index, doc in enumerate(
                                inputs.thought_context.documents, start=2
                            )
                        },
                    }
                    prompt = skill_generation_judge_prompt(
                        inputs.user_input,
                        response,
                        contexts,
                        required,
                        tables,
                        inputs.skill_source,
                        statements,
                        schema,
                    )
                    started = time.perf_counter()
                    active_stage = "judge"
                    with client.start_as_current_observation(
                        as_type="generation",
                        name="skill_generation.judge",
                        model=judge_model,
                        input=prompt,
                    ) as judge_observation:
                        judgment = await asyncio.wait_for(
                            judge.with_structured_output(
                                GroundedJudgment, method="json_mode"
                            )
                            .with_retry(stop_after_attempt=2)
                            .ainvoke(
                                prompt,
                                config={"callbacks": [judge_counter]},
                                extra_body={"thinking": {"type": "disabled"}},
                            ),
                            timeout=300,
                        )
                        output["judge_duration_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                        judge_observation.update(output=judgment.model_dump())
                        output["judgment"] = judgment.model_dump()
                        active_stage = None
                    mapping = judgment.field_mappings
                    mapping_valid = set(mapping) == set(required)
                    checks: list[dict[str, Any]] = []
                    for metric_id, fields in required.items():
                        mapped = mapping.get(metric_id, {})
                        mapping_valid &= set(mapped) == set(fields)
                        labels = [label for label in mapped.values() if label]
                        mapping_valid &= len(labels) == len(set(labels))
                        table = tables.get(metric_id, {})
                        mapping_valid &= set(labels) == set(table)
                        for path, expected in fields.items():
                            label = mapped.get(path, "")
                            actual = table.get(label)
                            checks.append(
                                {
                                    "indicator_id": metric_id,
                                    "field": path,
                                    "row_label": label,
                                    "expected": expected,
                                    "actual": actual,
                                    "correct": actual is not None
                                    and value_matches(
                                        actual,
                                        expected,
                                        allow_percent=path.endswith("_pct"),
                                    ),
                                }
                            )
                    contract = (
                        mapping_valid
                        and set(tables) == set(required)
                        and all(check["correct"] for check in checks)
                    )
                    sections = thought.get("来源章节", "")
                    if inputs.mode == "judge_calibration":
                        citation_valid = True
                    elif not inputs.thought_context.documents:
                        citation_valid = sections.strip("`* ") == "N/A" and bool(
                            thought.get("当前解读")
                        )
                    else:
                        allowed = {
                            doc.metadata["section"]
                            for doc in inputs.thought_context.documents
                        }
                        cited = [
                            s.strip("`* ")
                            for s in sections.replace("<br/>", "<br>")
                            .replace("<br />", "<br>")
                            .split("<br>")
                        ]
                        citation_valid = (
                            bool(cited)
                            and all(s in allowed for s in cited)
                            and bool(thought.get("当前解读"))
                        )
                    if inputs.mode != "judge_calibration":
                        citation_valid &= set(thought) == {"来源章节", "当前解读"}
                    for claim in judgment.claims:
                        if claim.supported and (
                            not claim.evidence_ids
                            or not set(claim.evidence_ids) <= contexts.keys()
                        ):
                            raise ValueError("裁判声称支持，但未引用有效证据")
                    total = len(checks) + len(judgment.claims)
                    supported = sum(c["correct"] for c in checks) + sum(
                        c.supported for c in judgment.claims
                    )
                    if total == 0:
                        raise ValueError("非空回答未取得可评测的事实")
                    output.update(
                        execution_status="success",
                        judgment=judgment.model_dump(),
                        field_checks=checks,
                        indicator_contract_valid=contract,
                        section_citations_valid=citation_valid,
                        faithfulness=supported / total,
                        answer_relevancy=sum(judgment.relevance.model_dump().values())
                        / 8,
                    )
                observation.update(output=output)
    except Exception as exc:  # noqa: BLE001 - 单项失败保留未知评分，继续剩余案例。
        output["error_type"] = type(exc).__name__
        output["error_stage"] = active_stage or "validation"
        cause = exc.__cause__
        if isinstance(cause, ValidationError):
            output["validation_errors"] = cause.errors(
                include_input=False, include_context=False
            )
        if active_stage == "judge":
            output["judge_raw_response"] = judge_counter.last_response
        message = str(exc).replace(settings.zhipuai_api_key, "[已隐藏]")
        output["error_message"] = _log_reason(message)[:400]
        print(
            json.dumps(
                {
                    key: output[key]
                    for key in ("error_type", "error_stage", "error_message")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if active_stage is not None:
            output[f"{active_stage}_duration_ms"] = (
                time.perf_counter() - started
            ) * 1000
    output["generation_model_attempts"] = generation_counter.attempts
    output["judge_model_attempts"] = judge_counter.attempts
    output["token_usage"] = {
        "generation": generation_counter.usage,
        "judge": judge_counter.usage,
    }
    return output


def case_pass(output: dict[str, Any], expected: Expected, mode: str) -> bool:
    if output["execution_status"] != "success" or output["skipped"] != expected.skipped:
        return False
    if output.get("assessment_scope") == "indicator_tables":
        return output["indicator_contract_valid"] and output["thought_output_absent"]
    if expected.skipped:
        return (
            output["response"] == ""
            and output["generation_model_attempts"] == 0
            and output["indicator_contract_valid"]
            and output["section_citations_valid"]
        )
    return (
        expected.faithfulness_min <= output["faithfulness"] <= expected.faithfulness_max
        and expected.answer_relevancy_min
        <= output["answer_relevancy"]
        <= expected.answer_relevancy_max
        and (
            mode == "judge_calibration"
            or (
                output["indicator_contract_valid"] and output["section_citations_valid"]
            )
        )
    )


def score_case(
    *, output: Any, expected_output: Any, input: Any, **kwargs: Any
) -> list[Evaluation]:
    expected = Expected.model_validate(expected_output)
    prefix = "judge_calibration_" if input["mode"] == "judge_calibration" else ""
    scores = [
        Evaluation(
            name=f"{prefix}case_pass",
            value=int(case_pass(output, expected, input["mode"])),
        )
    ]
    for name in ("indicator_contract_valid", "section_citations_valid"):
        if input["mode"] != "judge_calibration" and output[name] is not None:
            scores.append(Evaluation(name=name, value=int(output[name])))
    if output.get("assessment_scope") == "indicator_tables":
        scores.append(
            Evaluation(
                name="thought_output_absent", value=int(output["thought_output_absent"])
            )
        )
    for name in ("faithfulness", "answer_relevancy"):
        if output[name] is not None:
            scores.append(
                Evaluation(
                    name=f"{prefix}{name}",
                    value=output[name],
                    comment=json.dumps(output["judgment"], ensure_ascii=False),
                )
            )
    if input["mode"] != "judge_calibration":
        duration = (
            input["source_generation_duration_ms"]
            if input["mode"] == "response_replay"
            else output["generation_duration_ms"]
        )
        scores.append(Evaluation(name="duration_ms", value=duration))
    if output["judge_model_attempts"] > 0:
        scores.append(
            Evaluation(
                name=f"{prefix}judge_duration_ms", value=output["judge_duration_ms"]
            )
        )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="固定数据的技能正文与裁判校准评测")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("skill_generation_cases.jsonl"),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--judge-model",
        choices=(settings.fast_model, settings.smart_model),
        default=settings.smart_model,
    )
    parser.add_argument("--system-certificates", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("案例数量必须大于零")
        cases = cases[: args.limit]
    if args.validate_only:
        print(f"已校验 {len(cases)} 个案例")
        return
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    path = PROJECT_ROOT / "outputs/skill_generation_eval" / f"{args.run_name}.json"
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
        "generator_model": settings.smart_model,
        "judge_model": args.judge_model,
        "scoring_method": "project_llm_judge_v2",
        "faithfulness_definition": "正确表格字段与有证据支持的正文句子占全部待评字段及句子的比例；句内全部事实须有依据",
        "answer_relevancy_definition": "四个0至2的任务相关性量表之和除以8",
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "generation_prompt_sha256": hashlib.sha256(
            (PROJECT_ROOT / "prompts/crypto_analysis.py").read_bytes()
        ).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(
            (PROJECT_ROOT / "prompts/skill_generation_eval.py").read_bytes()
        ).hexdigest(),
        "model_timeout_seconds": 120,
        "max_attempts": 2,
        "data_mode": "fixed_indicators_and_thought_context",
        "source_execution_reports": {
            case.metadata.source_execution_report: case.metadata.source_execution_sha256
            for case in cases
            if case.metadata.source_execution_report is not None
        },
        "generation_origin": (
            "saved_production_response"
            if any(case.input.mode == "response_replay" for case in cases)
            else "direct_production_function"
        ),
    }
    rows: list[dict[str, Any]] = []

    async def task(*, item: Any) -> dict[str, Any]:
        inputs = GenerationInput.model_validate(item.input)
        output = await evaluate(client, inputs, args.judge_model)
        rows.append(
            {
                "input": item.input,
                "expected_output": item.expected_output,
                "metadata": item.metadata,
                "output": output,
            }
        )
        print(
            f"{item.metadata['case_id']}：{output['execution_status']}，faithfulness={output['faithfulness']}，answer_relevancy={output['answer_relevancy']}",
            flush=True,
        )
        return output

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败")
        dataset = "crypto-agent-skill-generation"
        client.create_dataset(
            name=dataset,
            description="固定指标与真实知识块的正文生成，含LLM裁判校准",
            input_schema=GenerationInput.model_json_schema(),
            expected_output_schema=Expected.model_json_schema(),
        )
        items = [
            client.create_dataset_item(
                dataset_name=dataset,
                id=str(uuid5(NAMESPACE_URL, f"{dataset}:{case.metadata.case_id}")),
                input=case.input.model_dump(mode="json"),
                expected_output=case.expected_output.model_dump(),
                metadata=case.metadata.model_dump(),
            )
            for case in cases
        ]
        experiment = client.run_experiment(
            name=dataset,
            run_name=args.run_name,
            data=items,
            task=task,
            evaluators=[score_case],
            max_concurrency=1,
            metadata=metadata,
        )
        for row, evaluated in zip(rows, experiment.item_results, strict=True):
            row["scores"] = [s.__dict__ for s in evaluated.evaluations]
            row["langfuse_trace_id"] = evaluated.trace_id
            row["case_pass"] = case_pass(
                row["output"],
                Expected.model_validate(row["expected_output"]),
                row["input"]["mode"],
            )
        if len(rows) != len(cases):
            raise RuntimeError("评测记录不完整")
        generated = [r for r in rows if r["input"]["mode"] != "judge_calibration"]
        calibrated = [r for r in rows if r["input"]["mode"] == "judge_calibration"]
        scored = [r for r in generated if r["output"]["faithfulness"] is not None]
        summary = {
            "table_only_cases": sum(
                r["output"].get("assessment_scope") == "indicator_tables"
                for r in generated
            ),
            "generation_cases": len(generated),
            "generation_passes": sum(r["case_pass"] for r in generated),
            "judge_calibration_cases": len(calibrated),
            "judge_calibration_passes": sum(r["case_pass"] for r in calibrated),
            "judge_calibration_passed": bool(calibrated)
            and all(r["case_pass"] for r in calibrated),
            "execution_errors": sum(
                r["output"]["execution_status"] == "error" for r in rows
            ),
            "scored_generation_cases": len(scored),
            "mean_faithfulness": sum(r["output"]["faithfulness"] for r in scored)
            / len(scored)
            if scored
            else None,
            "mean_answer_relevancy": sum(
                r["output"]["answer_relevancy"] for r in scored
            )
            / len(scored)
            if scored
            else None,
            "generation_duration": duration_summary(
                [
                    r["output"]["generation_duration_ms"]
                    for r in generated
                    if r["output"]["generation_model_attempts"] > 0
                ]
            ),
            "judge_duration": duration_summary(
                [
                    r["output"]["judge_duration_ms"]
                    for r in rows
                    if r["output"]["judge_model_attempts"] > 0
                ]
            ),
            "generation_model_attempts": sum(
                r["output"]["generation_model_attempts"] for r in rows
            ),
            "judge_model_attempts": sum(
                r["output"]["judge_model_attempts"] for r in rows
            ),
            "source_generation_duration": duration_summary(
                [
                    r["input"]["source_generation_duration_ms"]
                    for r in generated
                    if r["input"]["source_generation_duration_ms"] is not None
                    and not r["output"]["skipped"]
                ]
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "metadata": metadata,
                    "summary": summary,
                    "cases": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(experiment.format())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"本地结果：{path}")
    finally:
        client.flush()


if __name__ == "__main__":
    main()
