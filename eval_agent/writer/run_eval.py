"""读取人工维护的 writer 案例，使用 RAGAS 原生两项指标评测。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict
from pydantic import BaseModel, ConfigDict, Field, RootModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.config import PROJECT_ROOT, settings
from app.llm.model import (
    ainvoke_structured,
    get_embeddings,
    make_qwen3_max_llm,
)

_Schema = TypeVar("_Schema", bound=BaseModel)


class MessageData(BaseModel):
    """完整保留候选中的 LangChain 消息字段。"""

    model_config = ConfigDict(extra="allow")
    content: str = Field(min_length=1)


class CapturedMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["system", "human"]
    data: MessageData


class WriterInput(RootModel[list[CapturedMessage]]):
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
    """保留案例输入输出和采集时间，自动记录案例文件行号。"""

    model_config = ConfigDict(extra="forbid")
    node: Literal["writer"]
    ts: datetime | None = None
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
    if [message.type for message in case.input.root] != ["system", "human"]:
        raise ValueError("writer 案例必须保留一条 system 和一条 human 消息")
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
    """读取人工维护的案例文件，不改写输入或重新生成报告。"""
    cases = [
        WriterCase.model_validate({**json.loads(line), "source_line": number})
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip()
    ]
    if not cases:
        raise ValueError("writer 案例文件为空，请先添加案例")
    for case in cases:
        build_evidence(case)
    return cases


def create_metrics(
    judge_model: list[str],
    *,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> dict[str, Any]:
    """仅实际评测时加载 RAGAS，离线案例校验不依赖模型框架。"""
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    if judge_model != settings.qwen3_max_model:
        raise ValueError(f"未配置的千问 Max 裁判模型：{judge_model}")
    model = make_qwen3_max_llm()
    if callbacks is not None:
        model = model.with_config(callbacks=callbacks)

    class QwenJudge(InstructorBaseRagasLLM):
        """将千问 Max 结构化调用接入 RAGAS 原生指标。"""

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
            )
            return response_model.model_validate(result)

    embeddings = get_embeddings()

    class RagasEmbeddings(BaseRagasEmbedding):
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

    judge = QwenJudge()
    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=RagasEmbeddings(),
            strictness=1,
        ),
    }


def error_message(exc: Exception) -> str:
    message = str(exc)
    key = settings.dashscope_api_key
    if key:
        message = message.replace(key, "[已隐藏]")
    return f"{type(exc).__name__}: {message[:600]}"


async def evaluate(case: WriterCase, judge_model: list[str]) -> dict[str, Any]:
    """直接评分案例中的报告，不重新调用 writer，结果仅保存在本地。"""
    evidence = build_evidence(case)
    output: dict[str, Any] = {
        "actual_output": case.actual_output,
        "ragas_input": evidence.model_dump(),
        "faithfulness": None,
        "answer_relevancy": None,
        "errors": {},
    }
    for name, metric in create_metrics(judge_model).items():
        if name == "faithfulness" and not evidence.retrieved_contexts:
            continue
        arguments: dict[str, Any] = {
            "user_input": evidence.user_input,
            "response": case.actual_output,
        }
        if name == "faithfulness":
            arguments["retrieved_contexts"] = evidence.retrieved_contexts
        try:
            result = await asyncio.wait_for(metric.ascore(**arguments), timeout=600)
            if math.isfinite(result.value):
                output[name] = result.value
        except Exception as exc:  # noqa: BLE001 - 指标失败不抹掉另一项得分。
            output["errors"][name] = error_message(exc)
    return output


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


async def score_samples(cases: list[WriterCase]) -> list[dict[str, Any]]:
    """保持三案例并发，评分报告并返回本地记录。"""
    semaphore = asyncio.Semaphore(3)

    async def one(case: WriterCase) -> dict[str, Any]:
        async with semaphore:
            output = await evaluate(case, settings.qwen3_max_model)
        print(
            f"案例第{case.source_line}行：faithfulness={output['faithfulness']}，answer_relevancy={output['answer_relevancy']}",
            flush=True,
        )
        return {
            **case.model_dump(mode="json"),
            "output": output,
            "scores": [
                {"name": name, "value": output[name]}
                for name in ("faithfulness", "answer_relevancy")
                if output[name] is not None
            ],
        }

    return list(await asyncio.gather(*(one(case) for case in cases)))


def run_eval(cases: list[WriterCase]) -> dict[str, Any]:
    """执行两项 RAGAS 指标并汇总结果，不重新生成报告。"""
    print(f"开始 {len(cases)} 条 writer 案例评测...", flush=True)
    rows = asyncio.run(score_samples(cases))
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "cases": rows}


def main() -> None:
    cases_path = Path(__file__).with_name("writer_cases.jsonl")
    output_path = PROJECT_ROOT / "eval_agent/outputs/writer_eval/results.json"
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
