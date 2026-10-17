"""读取 writer 候选中的真实输入输出，使用 RAGAS 原生两项指标评测。"""

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
from typing import Any, Literal, TypeVar
from uuid import NAMESPACE_URL, uuid5

import httpx
import truststore
from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict
from langfuse import Evaluation, Langfuse
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config.config import PROJECT_ROOT, _env_files, settings
from app.llm.model import (
    ainvoke_structured,
    get_embeddings,
    make_bailian_fast_llm,
    make_bailian_smart_llm,
    model_session,
)

_Schema = TypeVar("_Schema", bound=BaseModel)


class EvalSettings(BaseSettings):
    """评测密钥沿用项目环境变量。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(), env_file_encoding="utf-8", extra="ignore"
    )
    langfuse_public_key: str = Field(min_length=1)
    langfuse_secret_key: str = Field(min_length=1)
    langfuse_base_url: str = "https://cloud.langfuse.com"


class MessageData(BaseModel):
    """完整保留候选中的 LangChain 消息字段。"""

    model_config = ConfigDict(extra="allow")
    content: str = Field(min_length=1)


class CapturedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["system", "human"]
    data: MessageData


class WriterInput(RootModel[list[CapturedMessage]]):
    @model_validator(mode="after")
    def validate_roles(self) -> WriterInput:
        if [message.type for message in self.root] != ["system", "human"]:
            raise ValueError("writer 案例必须保留一条 system 和一条 human 消息")
        return self

    def to_messages(self) -> list[BaseMessage]:
        return messages_from_dict(self.model_dump(mode="json"))


class MessageEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_index: Literal[0, 1]
    before: str = Field(min_length=1)
    after: str


class WriterMetadata(BaseModel):
    """只保留案例分类和来源，不保存旧评分规则或校准预期。"""

    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    label_reason: str = Field(min_length=1)
    source_candidate_file: str = Field(min_length=1)
    source_candidate_line: int = Field(ge=1)
    source_candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    edits: list[MessageEdit]


class WriterCase(BaseModel):
    """候选行原样保留，行号用于定位真实采集记录。"""

    model_config = ConfigDict(extra="forbid")
    node: Literal["writer"]
    input: WriterInput
    actual_output: str = Field(min_length=1)
    source_line: int = Field(ge=1)


class SkillContent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content: str = Field(min_length=1)
    cache_hit: bool


class ThoughtDocument(BaseModel):
    model_config = ConfigDict(extra="allow")
    content: str = Field(min_length=1)
    metadata: dict[str, Any]


class ThoughtContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    status: Literal["ok", "no_results", "error"]
    documents: list[ThoughtDocument]


class WriterEvidence(BaseModel):
    """只向指标提供真实问题和本轮证据。"""

    target_asset: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    retrieved_contexts: list[str]


def content_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def decode_section(text: str) -> Any:
    """只解析提示词中的完整 JSON，不用正则恢复内容。"""
    value, end = json.JSONDecoder().raw_decode(text.lstrip())
    if text.lstrip()[end:].strip():
        raise ValueError("候选消息的 JSON 段落包含额外内容")
    return value


def build_evidence(case: WriterCase) -> WriterEvidence:
    human = case.input.root[1].data.content
    scope, payload = human.split("## 任务上下文", 1)[1].split("## 各技能指标表格", 1)
    target, question = scope.split("- **目标资产**：", 1)[1].split(
        "- **用户问题**：", 1
    )
    # 旧采集消息可能含历史摘要，保留原始消息，但不把摘要送入裁判证据。
    question = question.split("- **此前结论摘要**：", 1)[0]
    tables, thoughts = payload.split("## 各技能思想原文（按 skill_name 对应）", 1)
    contents = [SkillContent.model_validate(item) for item in decode_section(tables)]
    materials = {
        name: ThoughtContext.model_validate(value)
        for name, value in decode_section(thoughts).items()
    }
    contexts = [
        json.dumps(
            {
                "target_asset": target.strip(),
                "skill_name": item.skill_name,
                "name": item.name,
                "content": item.content,
            },
            ensure_ascii=False,
        )
        for item in contents
    ]
    # 空检索状态也是输入事实，可用于核验“没有思想原文”的陈述。
    if contents or any(value.documents for value in materials.values()):
        contexts.extend(
            json.dumps(
                {
                    "skill_name": name,
                    "status": value.status,
                    "documents": [
                        {
                            "content": doc.content,
                            "source": doc.metadata.get("source"),
                            "section": doc.metadata.get("section"),
                        }
                        for doc in value.documents
                    ],
                },
                ensure_ascii=False,
            )
            for name, value in materials.items()
        )
    if any("USDT与USDC合计市值" in item.content for item in contents):
        contexts.append("本项目中，稳定币总市值是USDT与USDC合计市值的简称。")
    return WriterEvidence(
        target_asset=target.strip(),
        user_input=question.strip(),
        retrieved_contexts=contexts,
    )


def load_cases(path: Path) -> list[WriterCase]:
    """直接读取真实候选，不改写输入、不生成或使用人工参考报告。"""
    cases = [
        WriterCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("writer 候选文件为空")
    for case in cases:
        build_evidence(case)
    return cases


def create_metrics(judge_model: str, *, http: httpx.AsyncClient) -> dict[str, Any]:
    """仅实际评测时加载 RAGAS，离线案例校验不依赖模型框架。"""
    from langfuse.langchain import CallbackHandler
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    if judge_model == settings.bailian_smart_model:
        model = make_bailian_smart_llm(async_client=http)
    elif judge_model == settings.bailian_fast_model:
        model = make_bailian_fast_llm(async_client=http)
    else:
        raise ValueError(f"未配置的百炼裁判模型：{judge_model}")
    model = model.bind(enable_thinking=False).with_config(callbacks=[CallbackHandler()])

    class BailianJudge(InstructorBaseRagasLLM):
        """将项目百炼结构化调用接入 RAGAS 原生指标。"""

        def generate(self, prompt: str, response_model: type[_Schema]) -> _Schema:
            raise NotImplementedError("评测只使用异步 agenerate")

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def agenerate(
            self, prompt: str, response_model: type[_Schema]
        ) -> _Schema:
            result = await ainvoke_structured(
                model,
                [HumanMessage(content=prompt)],
                schema=response_model,
                timeout=120,
            )
            return response_model.model_validate(result)

    embeddings = get_embeddings()

    class BailianRagasEmbeddings(BaseRagasEmbedding):
        """将项目向量模型的异步接口接入 RAGAS。"""

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embeddings.embed_query(text)

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.wait_for(embeddings.aembed_query(text), timeout=120)

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_random_exponential(multiplier=1, max=5),
            reraise=True,
        )
        async def aembed_texts(
            self, texts: list[str], **kwargs: Any
        ) -> list[list[float]]:
            return await asyncio.wait_for(
                embeddings.aembed_documents(texts), timeout=120
            )

    judge = BailianJudge()
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=BailianRagasEmbeddings(),
            strictness=1,
        ),
    }


def error_message(exc: Exception) -> str:
    message = str(exc)
    for key in (settings.dashscope_api_key, settings.zhipuai_api_key):
        if key:
            message = message.replace(key, "[已隐藏]")
    return f"{type(exc).__name__}: {message[:600]}"


async def evaluate(
    client: Langfuse, case: WriterCase, judge_model: str
) -> dict[str, Any]:
    """直接评分采集的报告，不重新调用 writer。"""
    evidence = build_evidence(case)
    output: dict[str, Any] = {
        "actual_output": case.actual_output,
        "ragas_input": evidence.model_dump(),
        "faithfulness": None,
        "answer_relevancy": None,
        "errors": {},
    }
    with client.start_as_current_observation(
        as_type="span", name="writer.evaluate", input=case.model_dump(mode="json")
    ) as observation:
        async with model_session() as http:
            for name, metric in create_metrics(judge_model, http=http).items():
                if name == "faithfulness" and not evidence.retrieved_contexts:
                    continue
                arguments: dict[str, Any] = {
                    "user_input": evidence.user_input,
                    "response": case.actual_output,
                }
                if name == "faithfulness":
                    arguments["retrieved_contexts"] = evidence.retrieved_contexts
                with client.start_as_current_observation(
                    as_type="span", name=f"writer.ragas.{name}", input=arguments
                ) as metric_observation:
                    try:
                        result = await asyncio.wait_for(
                            metric.ascore(**arguments), timeout=600
                        )
                        if math.isfinite(result.value):
                            output[name] = result.value
                        metric_observation.update(output={"value": output[name]})
                    except Exception as exc:  # noqa: BLE001 - 指标失败不抹掉另一项得分。
                        output["errors"][name] = error_message(exc)
                        metric_observation.update(
                            output={"error": output["errors"][name]}, level="ERROR"
                        )
        observation.update(
            output=output, level="ERROR" if output["errors"] else "DEFAULT"
        )
    return output


def score_case(*, output: Any, **kwargs: Any) -> list[Evaluation]:
    """只上传两项质量分数，运行状态与耗时不再转换成分数。"""
    return [
        Evaluation(name=name, value=output[name])
        for name in ("faithfulness", "answer_relevancy")
        if output[name] is not None
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"cases": len(rows)}
    for name in ("faithfulness", "answer_relevancy"):
        values = [
            row["output"][name] for row in rows if row["output"][name] is not None
        ]
        result[name] = {
            "mean": sum(values) / len(values) if values else None,
            "scored_cases": len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="真实 writer 候选输入输出的 RAGAS 评测"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "eval_agent/candidates/writer_candidates.jsonl",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-certificates", action="store_true")
    parser.add_argument(
        "--judge-model",
        choices=(settings.bailian_fast_model, settings.bailian_smart_model),
        default=settings.bailian_smart_model,
    )
    args = parser.parse_args()
    cases = load_cases(args.candidates)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("案例数量必须大于零")
        cases = cases[: args.limit]
    if args.validate_only:
        print(f"已校验 {len(cases)} 条真实 writer 候选输入输出及 RAGAS 输入")
        return
    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "writer-ragas-%Y%m%dT%H%M%SZ"
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("实验名称不能包含路径")
    path = PROJECT_ROOT / "outputs/writer_eval" / f"{run_name}.json"
    if path.exists():
        parser.error("结果文件已存在，请使用新的实验名称")
    if not settings.dashscope_api_key.strip():
        parser.error("未配置 DASHSCOPE_API_KEY")
    if args.system_certificates:
        truststore.inject_into_ssl()
    # 禁用 RAGAS 自身的匿名使用上报，记录只写入配置的 Langfuse。
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    # 上传数据之前验证 RAGAS 依赖。
    import_module("ragas.metrics.collections")
    config = EvalSettings()
    client = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        base_url=config.langfuse_base_url,
    )
    metadata = {
        "data_mode": "captured_writer_input_output",
        "judge_model": args.judge_model,
        "embedding_model": settings.embedding_model,
        "ragas_version": version("ragas"),
        "langchain_community_version": version("langchain-community"),
        "scoring_method": "ragas_native",
        "metrics": ["faithfulness", "answer_relevancy"],
        "answer_relevancy_strictness": 1,
        "model_timeout_seconds": 120,
        "metric_timeout_seconds": 600,
        "max_attempts": 2,
        "candidate_file": str(args.candidates.resolve()),
        "candidate_file_sha256": hashlib.sha256(
            args.candidates.read_bytes()
        ).hexdigest(),
        "evaluator_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "boundary": "评估真实采集的writer输入和actual_output，不重新生成报告",
    }
    rows: list[dict[str, Any]] = []
    run_status = "failed"

    async def task(*, item: Any) -> dict[str, Any]:
        case = WriterCase.model_validate(item.input)
        output = await evaluate(client, case, args.judge_model)
        rows.append(
            {
                **case.model_dump(mode="json"),
                "output": output,
            }
        )
        print(
            f"候选第{case.source_line}行：faithfulness={output['faithfulness']}，answer_relevancy={output['answer_relevancy']}",
            flush=True,
        )
        return output

    try:
        if not client.auth_check():
            raise RuntimeError("Langfuse 认证失败")
        dataset = "crypto-agent-writer"
        client.create_dataset(
            name=dataset,
            description="真实采集的writer输入与实际报告，使用RAGAS原生两项指标评分",
            input_schema=WriterCase.model_json_schema(),
            expected_output_schema={},
        )
        items = [
            client.create_dataset_item(
                dataset_name=dataset,
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{dataset}:{content_sha256(case.model_dump(mode='json'))}",
                    )
                ),
                input=case.model_dump(mode="json"),
            )
            for case in cases
        ]
        experiment = client.run_experiment(
            name=dataset,
            run_name=run_name,
            data=items,
            task=task,
            evaluators=[score_case],
            max_concurrency=3,
            metadata=metadata,
        )
        rows_by_line = {row["source_line"]: row for row in rows}
        for result in experiment.item_results:
            row = rows_by_line[result.item.input["source_line"]]
            row["scores"] = [score.__dict__ for score in result.evaluations]
            row["langfuse_trace_id"] = result.trace_id
        if len(rows) != len(cases) or len(experiment.item_results) != len(rows):
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
                    "planned_cases": len(cases),
                    "metadata": metadata,
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
