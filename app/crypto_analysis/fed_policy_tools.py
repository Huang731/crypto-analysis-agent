"""维度02工具：搜索候选材料、浏览官方原文、按规则形成指标记录。"""

import asyncio
import json
import re
from dataclasses import asdict
from datetime import date
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import ValidationError

from app.browser_tool import get_browser_tool
from app.crypto_analysis.calculators.fed_policy_calculator import (
    FedPolicyAssessment,
    calculate_fed_policy,
    communication_sentences,
    comparison_date,
    evidence_paragraphs,
    is_fed_url,
    publication_date,
    qt_monthly_caps,
)
from app.crypto_analysis.crypto_analysis_agent import (
    MetricContext,
    _normalize_result,
    _tool_error,
    _tool_message,
    missing_metric,
)
from app.crypto_analysis.sources.fred_source import (
    get_fed_dff_background as get_fed_dff_background_data,
)
from app.crypto_analysis.sources.web_search_source import search_bocha
from app.tracing import estimate_tokens, trace_generation, trace_span
from prompts.fed_policy import evaluation_prompt


def _check_dimension(runtime: ToolRuntime[MetricContext]) -> ToolMessage | None:
    if runtime.context["dimension"].dimension_id != "02":
        return _tool_error(runtime, "此工具只用于维度02")
    return None


def _assessment_material(
    evidence_id: str, record: dict[str, Any], deadline: date
) -> dict[str, Any]:
    """各指标只展示相关原文，避免模型把实际调息当成沟通或缩表变化。"""
    material = {"evidence_id": evidence_id, "status": record["status"]}
    if record["status"] != "ok":
        return {**material, "reason": record["reason"]}
    raw = record["raw"]
    paragraphs = [
        {"paragraph_index": i + 1, "text": text[:6000], "truncated": len(text) > 6000}
        for i, text in enumerate(evidence_paragraphs(raw))
    ]
    published = publication_date(raw)
    balance_terms = re.compile(
        r"holdings|monthly cap|principal payments|purchases|reinvest|balance sheet",
        re.IGNORECASE,
    )
    return {
        **material,
        "url": raw["url"],
        "title": raw["page_title"],
        "published_date_from_source": published.isoformat() if published else None,
        "date_paragraphs": [
            p
            for p in paragraphs
            if re.search(r"\b20\d\d\b|\beffective\b", p["text"], re.IGNORECASE)
        ],
        "sep_paragraphs": [
            p
            for p in paragraphs
            if "Federal funds rate" in p["text"]
            or "target range for the federal funds rate" in p["text"]
        ],
        "communication_sentences": communication_sentences(raw),
        "balance_sheet_paragraphs": [
            p for p in paragraphs if balance_terms.search(p["text"])
        ],
        "qt_monthly_caps_at_cutoff": qt_monthly_caps(raw, deadline),
        "tables": [
            table
            for table in raw.get("tables", [])
            if any(row and "Federal funds rate" in row[0] for row in table["rows"])
        ],
    }


def build_fed_policy_tools(llm: Any) -> list[BaseTool]:
    """模型通过闭包注入，工具参数不暴露请求范围、规则或原始数值。"""
    evaluator = llm.with_structured_output(
        FedPolicyAssessment, method="function_calling"
    ).with_retry(retry_if_exception_type=(ValidationError,), stop_after_attempt=2)
    evaluation_parameters: dict[str, Any] = {"temperature": 0}
    if getattr(llm, "model_name", "").startswith("glm-4.5"):
        # 仅政策比较调用启用深度思考，不改变其他维度与全局模型配置。
        evaluation_parameters["extra_body"] = {"thinking": {"type": "enabled"}}

    @tool
    async def search_web(
        query: str, runtime: ToolRuntime[MetricContext], max_results: int = 6
    ) -> Command | ToolMessage:
        """用 Bocha 查找 Federal Reserve 官方材料。返回候选 URL；摘要不可评分，须继续 browse_fed_evidence。"""
        error = _check_dimension(runtime)
        if error is not None:
            return error
        # 搜索范围固定官方站点；发布日期仍由原始材料核验，不能靠搜索排名确定。
        query = f"site:federalreserve.gov {query}"
        evidence_id = runtime.tool_call_id
        with trace_span("analysis.fed.search") as span:
            try:
                raw = await search_bocha(query, max_results)
            except Exception as exc:  # noqa: BLE001 - 保存搜索服务实际失败原因，不切换来源。
                span.metadata["status"] = "error"
                record = {
                    "data_kind": "fed_search",
                    "source_action": "bocha.search",
                    "status": "data_missing",
                    "reason": str(exc),
                }
                summary = {"evidence_id": evidence_id, **record}
                return Command(
                    update={
                        "evidence": {evidence_id: record},
                        "messages": [_tool_message(runtime, summary)],
                    }
                )
            # 搜索服务偶尔忽略 site 操作符，程序只接受实际的官方域名。
            candidates = [row for row in raw["results"] if is_fed_url(row["url"])]
            record = {
                "data_kind": "fed_search",
                "source_action": "bocha.search",
                "status": "ok" if candidates else "data_missing",
                "raw": raw,
            }
            if not candidates:
                record["reason"] = "Bocha 未返回 Federal Reserve 官方候选链接"
            research = [
                {
                    "evidence_id": evidence_id,
                    "kind": "search_candidate",
                    "content": row["snippet"],
                    "source": row["url"],
                    "title": row["title"],
                    "observed_at": None,
                    "accessed_at": raw["accessed_at"],
                    "status": "candidate",
                }
                for row in candidates
            ]
            summary = {
                "evidence_id": evidence_id,
                "status": record["status"],
                "candidates": candidates,
            }
            if "reason" in record:
                summary["reason"] = record["reason"]
            span.output = {"candidate_count": len(candidates)}
            return Command(
                update={
                    "evidence": {evidence_id: record},
                    "research_evidence": research,
                    "messages": [_tool_message(runtime, summary)],
                }
            )

    @tool
    async def browse_fed_evidence(
        task: str,
        urls: list[str],
        runtime: ToolRuntime[MetricContext],
        max_pages: int = 6,
    ) -> Command | ToolMessage:
        """自主浏览官方候选页面，点击详情/翻页并提取原文和 SEP 表格。\
返回页面 evidence_ids，供 evaluate_fed_policy 使用。"""
        error = _check_dimension(runtime)
        if error is not None:
            return error
        if not urls or not all(is_fed_url(url) for url in urls):
            return _tool_error(
                runtime, "urls 必须是 Federal Reserve 官方 http/https 页面"
            )
        if not 1 <= max_pages <= 8:
            return _tool_error(runtime, "max_pages 必须为 1 至 8")
        scope = runtime.context["scope"]
        question = (
            f"{task}\n截止时间固定为 {scope['as_of']}，比较日期为 {comparison_date(scope['as_of'])}。"
            "只收集截止时已发布材料，优先最新版 SEP、当前生效目标区间、最新与上一可比沟通，"
            "以及当前和比较日已生效的资产负债表政策与操作速度。"
            "保存完整的发布日期、生效日期、利率表格表头与原文；需要多份证据时分别 extract。"
        )
        call_id = runtime.tool_call_id
        with trace_span("analysis.fed.browse") as span:
            try:
                pages = await get_browser_tool().browse_for_evidence(
                    question, urls, max_pages=max_pages, llm=llm
                )
            except Exception as exc:  # noqa: BLE001 - 保存浏览器或模型实际失败原因，不补造证据。
                span.metadata["status"] = "error"
                record = {
                    "data_kind": "fed_page",
                    "source_action": "browser.browse_for_evidence",
                    "status": "data_missing",
                    "reason": str(exc),
                }
                return Command(
                    update={
                        "evidence": {call_id: record},
                        "messages": [
                            _tool_message(runtime, {"evidence_id": call_id, **record})
                        ],
                    }
                )
            records, research, summaries = {}, [], []
            for index, page in enumerate(pages):
                if not is_fed_url(page.url):
                    continue
                evidence_id = f"{call_id}:{index + 1}"
                records[evidence_id] = {
                    "data_kind": "fed_page",
                    "source_action": "browser.browse_for_evidence",
                    "status": "ok",
                    "raw": asdict(page),
                }
                research.append(
                    {
                        "evidence_id": evidence_id,
                        "kind": "official_page",
                        "content": page.content,
                        "source": page.url,
                        "title": page.page_title,
                        "observed_at": None,
                        "accessed_at": page.accessed_at,
                        "status": "retrieved",
                        "tables": page.tables,
                    }
                )
                summaries.append(
                    {
                        "evidence_id": evidence_id,
                        "url": page.url,
                        "title": page.page_title,
                        "snapshot": page.snapshot,
                        "table_count": len(page.tables),
                    }
                )
            if not records:
                records[call_id] = {
                    "data_kind": "fed_page",
                    "source_action": "browser.browse_for_evidence",
                    "status": "data_missing",
                    "reason": "浏览器未保存官方原文证据",
                }
            summary = {
                "status": "ok" if summaries else "data_missing",
                "pages": summaries,
                "evidence_ids": list(records),
                "reason": None if summaries else "浏览器未保存官方原文证据",
            }
            span.output = {"evidence_count": len(summaries)}
            return Command(
                update={
                    "evidence": records,
                    "research_evidence": research,
                    "messages": [_tool_message(runtime, summary)],
                }
            )

    @tool
    async def evaluate_fed_policy(
        evidence_ids: list[str], runtime: ToolRuntime[MetricContext]
    ) -> Command | ToolMessage:
        """读取已保存官方页面，评估 SEP_Policy_Stance、Fed_Communication、QE_QT。\
仅传真实页面 evidence_ids，不传原文、数值或分数。"""
        error = _check_dimension(runtime)
        if error is not None:
            return error
        if not evidence_ids:
            return _tool_error(runtime, "请先取得官方原文 evidence_ids")
        records = {}
        for evidence_id in dict.fromkeys(evidence_ids):
            record = runtime.state["evidence"].get(evidence_id)
            if record is None or record.get("data_kind") != "fed_page":
                return _tool_error(
                    runtime,
                    f"{evidence_id} 不是已保存的官方页面 ID，搜索摘要不能用于评分",
                )
            records[evidence_id] = record
        scope = runtime.context["scope"]
        # 完整原文留在状态中；评估上下文控制长度，并完整保留原始 SEP 网格。
        materials = [
            _assessment_material(key, record, comparison_date(scope["as_of"]))
            for key, record in records.items()
        ]
        prompt = evaluation_prompt(
            scope["as_of"],
            comparison_date(scope["as_of"]),
            json.dumps(FedPolicyAssessment.model_json_schema(), ensure_ascii=False),
            json.dumps(materials, ensure_ascii=False),
        )
        with trace_span("analysis.fed.evaluate") as span:
            try:
                with trace_generation(
                    "fed_policy_assessment",
                    model=getattr(llm, "model_name", "injected"),
                ) as generation:
                    assessment = await asyncio.wait_for(
                        evaluator.ainvoke(
                            prompt, config=runtime.config, **evaluation_parameters
                        ),
                        timeout=60,
                    )
                    if not isinstance(assessment, FedPolicyAssessment):
                        raise TypeError("政策评估模型未返回 FedPolicyAssessment")
                    generation.usage = {
                        "input": estimate_tokens(prompt),
                        "output": estimate_tokens(assessment.model_dump_json()),
                    }
            except Exception as exc:  # noqa: BLE001 - 重试耗尽或超时明确记为评估失败，不换模型、不补分数。
                span.metadata["status"] = "error"
                reason = f"政策评估失败：{type(exc).__name__}: {str(exc)[:2000]}"
                indicators = {
                    metric.id: missing_metric(
                        "02", metric.id, "calculation_error", reason
                    )
                    for metric in runtime.context["dimension"].metrics
                }
                return Command(
                    update={
                        "indicator_results": indicators,
                        "messages": [
                            _tool_message(
                                runtime,
                                {"status": "calculation_error", "reason": reason},
                            )
                        ],
                    }
                )
            calculated = calculate_fed_policy(assessment, records, scope)
            indicators = {
                metric_id: _normalize_result(
                    "02", metric_id, result, "evaluate_fed_policy"
                )
                for metric_id, result in calculated.items()
            }
            verified_quotes = [
                {
                    "evidence_id": citation["evidence_id"],
                    "kind": "verified_quote",
                    "indicator_id": metric_id,
                    "content": citation["quote"],
                    "source": citation["url"],
                    "observed_at": citation["published_at"],
                    "effective_at": citation["effective_at"],
                    "accessed_at": citation["accessed_at"],
                    "status": "ok",
                }
                for metric_id, result in calculated.items()
                if result["status"] == "ok"
                for citation in result["citations"]
            ]
            span.output = {
                key: {"status": val["status"], "score": val["score"]}
                for key, val in indicators.items()
            }
            record = {
                "data_kind": "fed_policy_assessment",
                "source_action": "evaluate_fed_policy",
                "status": "ok",
                "raw": assessment.model_dump(mode="json"),
                "evidence_ids": list(records),
            }
            return Command(
                update={
                    "evidence": {runtime.tool_call_id: record},
                    "indicator_results": indicators,
                    "research_evidence": verified_quotes,
                    "messages": [
                        _tool_message(
                            runtime, {"indicators": list(indicators.values())}
                        )
                    ],
                }
            )

    @tool
    async def get_fed_dff_background(
        runtime: ToolRuntime[MetricContext],
    ) -> Command | ToolMessage:
        """从 FRED 读取当前与三个月前 DFF，仅作背景，不计分、不计入覆盖率。失败也不阻止三个政策指标评估。"""
        error = _check_dimension(runtime)
        if error is not None:
            return error
        call_id = runtime.tool_call_id
        try:
            raw = await get_fed_dff_background_data(runtime.context["scope"]["as_of"])
        except Exception as exc:  # noqa: BLE001 - DFF失败只记背景缺失，不影响政策指标。
            record = {
                "data_kind": "dff_background",
                "source_action": "fred.DFF",
                "status": "data_missing",
                "reason": str(exc),
            }
        else:
            record = {
                "data_kind": "dff_background",
                "source_action": "fred.DFF",
                "status": "ok",
                "raw": raw,
            }
        return Command(
            update={
                "evidence": {call_id: record},
                "research_evidence": [
                    {
                        "evidence_id": call_id,
                        "kind": "background",
                        "scored": False,
                        **record,
                    }
                ],
                "messages": [
                    _tool_message(
                        runtime, {"evidence_id": call_id, "scored": False, **record}
                    )
                ],
            }
        )

    return [
        search_web,
        browse_fed_evidence,
        evaluate_fed_policy,
        get_fed_dff_background,
    ]
