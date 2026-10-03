"""维度02证据校验与固定分档：数值由原始表格和声明计算，文本仅提交分类。"""

import calendar
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    """引文必须来自浏览工具已保存的官方页面。"""

    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    paragraph_index: int = Field(ge=1)
    published_at: date
    effective_at: date | None = None
    material_type: Literal["sep", "statement", "press_conference", "minutes", "balance_sheet_policy", "implementation_note"]


class EffectiveEvidenceRef(EvidenceRef):
    """生效政策必须有生效日期，不能用默认 null 绕过必要证据。"""

    effective_at: date


class SEPAssessment(BaseModel):
    """不接受模型填写利率值，由程序从 SEP 网格和目标区间引文读取。"""

    model_config = ConfigDict(extra="forbid")
    sep: EvidenceRef
    target_range: EffectiveEvidenceRef


class CommunicationRef(EvidenceRef):
    """沟通只引用可比较的政策表述句子，排除已执行调息等重复计分内容。"""

    sentence_index: int = Field(ge=1)


class CommunicationAssessment(BaseModel):
    """可比版本的文本分类，不接受模型分数。"""

    model_config = ConfigDict(extra="forbid")
    current: CommunicationRef
    previous: CommunicationRef
    change: Literal["new_tightening_guidance", "marginally_hawkish", "unchanged", "conflict", "marginally_dovish", "new_easing_guidance"]
    interpretation: str = Field(min_length=1)
    confidence: Literal[1, 0.75, 0.5]


class BalanceSheetAssessment(BaseModel):
    """仅分类已生效的政策变化，未知方向不能给分。"""

    model_config = ConfigDict(extra="forbid")
    current: EffectiveEvidenceRef
    baseline: EffectiveEvidenceRef
    change: Literal["start_qt", "accelerate_qt", "reduce_or_end_qe", "unchanged", "slow_or_stop_qt", "increase_existing_qe", "start_policy_qe", "unknown"]
    interpretation: str = Field(min_length=1)
    confidence: Literal[1, 0.75]


class FedPolicyAssessment(BaseModel):
    """模型只能从已保存的证据抽取引文、日期和文本类别；缺证据时填 null。"""

    model_config = ConfigDict(extra="forbid")
    sep: SEPAssessment | None
    communication: CommunicationAssessment | None
    balance_sheet: BalanceSheetAssessment | None
    missing_reasons: dict[str, str]


def is_fed_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme in {"http", "https"} and (host == "federalreserve.gov" or host.endswith(".federalreserve.gov"))


def comparison_date(as_of: str) -> date:
    """按日历向前移动三个月，月末取目标月实际最后一天。"""
    cutoff = datetime.fromisoformat(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    day = cutoff.astimezone(timezone.utc).date()
    index = day.year * 12 + day.month - 1 - 3
    year, month = index // 12, index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _compact(text: str) -> str:
    return " ".join(text.split())


def evidence_paragraphs(raw: dict[str, Any]) -> list[str]:
    """给保存的连续原文段落编号，模型只能选择编号，不能拼接或改写引文。"""
    return [part.strip() for part in re.split(r"\n\s*\n", raw["content"]) if part.strip()]


def communication_sentences(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """句号后按英文句首切分；保留原文及编号，过滤不属于沟通指标的句子。"""
    candidates = []
    excluded = re.compile(
        r"\b(?:decided|voted)\s+(?:unanimously\s+)?to\s+(?:lower|raise|maintain|keep)|"
        r"\b(?:holdings|reinvest|purchases|monthly cap|principal payments|balance sheet|"
        r"summary of economic projections|dot plot)\b", re.IGNORECASE)
    for paragraph_index, paragraph in enumerate(evidence_paragraphs(raw), 1):
        for sentence_index, sentence in enumerate(re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph), 1):
            if not excluded.search(sentence):
                candidates.append({"paragraph_index": paragraph_index, "sentence_index": sentence_index, "text": sentence})
    return candidates


def publication_date(raw: dict[str, Any]) -> date | None:
    """SEP和新闻稿URL中的日期是发布日期，会议纪要URL中的日期是会议日期。"""
    path = urlparse(raw["url"]).path
    if "fomcprojtabl" not in path and "/pressreleases/monetary" not in path:
        return None
    matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", path)
    if not matches:
        return None
    encoded = matches[0]
    return date(int(encoded[:4]), int(encoded[4:6]), int(encoded[6:]))


def _quote(ref: EvidenceRef, raw: dict[str, Any]) -> str:
    if isinstance(ref, CommunicationRef):
        for sentence in communication_sentences(raw):
            if sentence["paragraph_index"] == ref.paragraph_index and sentence["sentence_index"] == ref.sentence_index:
                return sentence["text"]
        raise ValueError("沟通句子无效，已执行调息、点阵图和资产负债表内容不能用于沟通评分")
    paragraphs = evidence_paragraphs(raw)
    if ref.paragraph_index > len(paragraphs):
        raise ValueError("引用段落编号不在保存的原文中")
    return paragraphs[ref.paragraph_index - 1]


def _verify_ref(ref: EvidenceRef, records: dict[str, dict[str, Any]], deadline: date,
                effective: bool = False) -> dict[str, Any]:
    record = records.get(ref.evidence_id)
    if not record or record.get("data_kind") != "fed_page" or record.get("status") != "ok":
        raise ValueError("引文未绑定成功取得的官方页面证据")
    raw = record["raw"]
    if not is_fed_url(raw["url"]):
        raise ValueError("评分只接受 Federal Reserve 官方页面")
    _quote(ref, raw)
    if ref.published_at > deadline:
        raise ValueError("引用材料在截止日期后发布")
    # 日期优先核对官方材料 URL 中的日期；无日期 URL 时核对正文日期。
    actual_date = publication_date(raw)
    if actual_date is not None:
        if ref.published_at != actual_date:
            raise ValueError("材料发布日期与官方 URL 不一致")
    else:
        body = _compact(raw["content"])
        english_date = f"{ref.published_at:%B} {ref.published_at.day}, {ref.published_at.year}"
        if english_date not in body and ref.published_at.isoformat() not in body:
            raise ValueError("无法从官方正文核验材料发布日期")
    if effective and (ref.effective_at is None or ref.effective_at > deadline or ref.effective_at < ref.published_at):
        raise ValueError("缺少可核验的已生效日期，或政策尚未生效")
    return raw


def _citation(ref: EvidenceRef, raw: dict[str, Any]) -> dict[str, Any]:
    return {**ref.model_dump(mode="json"), "quote": _quote(ref, raw), "url": raw["url"], "accessed_at": raw["accessed_at"]}


def _sep_median(raw: dict[str, Any], year: int) -> Decimal:
    """按展开的 Median 表头和当年列读取，不把下一年或前次修订当成当前值。"""
    values = set()
    for table in raw.get("tables", []):
        rows = table["rows"]
        for group in rows:
            median_cols = [i for i, cell in enumerate(group)
                           if re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹*\s]+$", "", _compact(cell)).lower() == "median"]
            if not median_cols:
                continue
            for header in rows:
                cols = [i for i in median_cols if i < len(header) and header[i].strip() == str(year)]
                for row in rows:
                    if not row or re.sub(r"[\d¹²³⁴⁵⁶⁷⁸⁹*\s]+$", "", _compact(row[0])).lower() != "federal funds rate":
                        continue
                    for col in cols:
                        if col < len(row) and re.fullmatch(r"\d+(?:\.\d+)?", row[col].strip()):
                            values.add(Decimal(row[col].strip()))
    if len(values) != 1:
        raise ValueError(f"未取得唯一的 {year} 年末联邦基金利率 Median 表格值")
    return values.pop()


_RATE = r"\d+(?:\.\d+)?(?:\s*[-–]\s*\d+/\d+)?"


def _rate_value(text: str) -> Decimal:
    if "/" not in text:
        return Decimal(text)
    whole, numerator, denominator = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)/(\d+)", text).groups()
    return Decimal(whole) + Decimal(numerator) / Decimal(denominator)


def _target_range(quote: str) -> tuple[Decimal, Decimal]:
    # 调息声明会先写 by 1/2 percentage point，再写实际目标区间，不能把调息幅度当利率。
    match = re.search(r"target range for the federal funds rate\s+"
                      r"(?:by\s+\d+(?:/\d+|\.\d+)?\s+percentage\s+points?\s+)?(?:at|to|of)\s+(" + _RATE +
                      r")\s+(?:to|and)\s+(" + _RATE + r")\s+percent", _compact(quote), re.IGNORECASE)
    if not match:
        raise ValueError("目标区间引文缺少可解析的联邦基金利率上下限")
    lower, upper = _rate_value(match[1]), _rate_value(match[2])
    if lower > upper:
        raise ValueError("目标区间上下限顺序错误")
    return lower, upper


def _sep_score(delta_bp: Decimal) -> float:
    if delta_bp >= 50:
        return 0
    if delta_bp > 10:
        return 2.5
    if delta_bp >= -10:
        return 5
    if delta_bp > -50:
        return 7.5
    return 10


def qt_monthly_caps(raw: dict[str, Any], deadline: date) -> dict[str, float] | None:
    """读取国债和机构证券的月度缩表上限；宣布的未来安排到生效日后才采用。"""
    caps = {}
    published = publication_date(raw)
    if published is not None and published > deadline:
        return None
    for sentence in re.split(r"\n|(?<=[.!?])\s+(?=[A-Z])", raw["content"]):
        match = re.search(r"cap of \$(\d+(?:\.\d+)?) billion per month", sentence, re.IGNORECASE)
        if not match or "holdings" not in sentence.lower():
            continue
        beginning = re.search(r"Beginning (?:on )?([A-Z][a-z]+) (\d{1,2})(?:, (20\d{2}))?", sentence)
        if beginning:
            if published is None:
                raise ValueError("月度上限的生效年份缺少官方发布日期")
            effective = datetime.strptime(f"{beginning[1]} {beginning[2]} {beginning[3] or published.year}", "%B %d %Y").replace(tzinfo=timezone.utc).date()
            if effective > deadline:
                continue
        if "agency" in sentence.lower():
            caps["agency_usd_bn"] = float(match[1])
        elif "treasury" in sentence.lower():
            caps["treasury_usd_bn"] = float(match[1])
    return caps if len(caps) == 2 else None


def calculate_fed_policy(assessment: FedPolicyAssessment, records: dict[str, dict[str, Any]],
                         scope: dict[str, str]) -> dict[str, dict[str, Any]]:
    """每项独立校验，必要证据缺失不记零分，也不影响其他合法指标。"""
    cutoff = datetime.fromisoformat(scope["as_of"])
    if cutoff.tzinfo is None:
        raise ValueError("as_of 必须包含时区")
    today = cutoff.astimezone(timezone.utc).date()
    baseline_day = comparison_date(scope["as_of"])
    results = {}
    for metric_id, item in (("SEP_Policy_Stance", assessment.sep),
                            ("Fed_Communication", assessment.communication), ("QE_QT", assessment.balance_sheet)):
        try:
            if item is None:
                raise ValueError(assessment.missing_reasons.get(metric_id) or "未取得必要的官方证据")
            if metric_id == "SEP_Policy_Stance":
                if item.sep.material_type != "sep" or item.target_range.material_type not in {"statement", "implementation_note"}:
                    raise ValueError("SEP 或目标区间的材料类型不正确")
                sep_raw = _verify_ref(item.sep, records, today)
                target_raw = _verify_ref(item.target_range, records, today, effective=True)
                median = _sep_median(sep_raw, today.year)
                lower, upper = _target_range(_quote(item.target_range, target_raw))
                midpoint = (lower + upper) / 2
                delta = 100 * (median - midpoint)
                result = {
                    "year": today.year, "year_end_median_pct": float(median),
                    "target_lower_pct": float(lower), "target_upper_pct": float(upper),
                    "target_midpoint_pct": float(midpoint), "delta_bp": float(delta),
                    "citations": [_citation(item.sep, sep_raw), _citation(item.target_range, target_raw)],
                    "score": _sep_score(delta), "confidence": 1, "unit": "bp",
                    "observed_at": item.sep.published_at.isoformat(), "source": sep_raw["url"], "stale": False,
                }
            elif metric_id == "Fed_Communication":
                allowed = {"statement", "press_conference", "minutes"}
                if item.current.material_type not in allowed or item.current.material_type != item.previous.material_type:
                    raise ValueError("政策沟通必须比较相同类型的官方材料")
                current = _verify_ref(item.current, records, today)
                previous = _verify_ref(item.previous, records, today)
                if item.current.evidence_id == item.previous.evidence_id or item.current.published_at < item.previous.published_at:
                    raise ValueError("最新与上一可比沟通材料的引用或日期无效")
                if item.change != "conflict" and item.confidence == 0.5:
                    raise ValueError("0.5置信度只用于同级沟通证据冲突")
                if _compact(_quote(item.current, current)) == _compact(_quote(item.previous, previous)) and item.change != "unchanged":
                    raise ValueError("相同沟通表述不能支持新增指引、边际变化或同级冲突")
                if item.change in {"new_easing_guidance", "new_tightening_guidance"}:
                    guidance = _quote(item.current, current)
                    future = re.search(r"\b(?:will|expect\w*|anticipat\w*|intend\w*|plan\w*|likely|appropriate)\b", guidance, re.IGNORECASE)
                    direction = r"\b(?:reduc\w*|lower\w*|cuts?|eas\w*)\b" if item.change == "new_easing_guidance" else r"\b(?:increas\w*|rais\w*|tighten\w*)\b"
                    if not future or not re.search(direction, guidance, re.IGNORECASE):
                        raise ValueError("新增明确指引缺少原文中的未来政策方向，不能用经济判断或已执行调息替代")
                scores = {"new_tightening_guidance": 0, "marginally_hawkish": 2.5, "unchanged": 5,
                          "conflict": 5, "marginally_dovish": 7.5, "new_easing_guidance": 10}
                result = {
                    "current_material": _citation(item.current, current), "previous_material": _citation(item.previous, previous),
                    "citations": [_citation(item.current, current), _citation(item.previous, previous)],
                    "policy_change": item.change, "interpretation": item.interpretation,
                    "stale": item.current.published_at < baseline_day, "score": scores[item.change],
                    "confidence": 0.5 if item.change == "conflict" else item.confidence,
                    "unit": "policy_category", "observed_at": item.current.published_at.isoformat(), "source": current["url"],
                }
            else:
                if item.current.material_type not in {"statement", "balance_sheet_policy", "implementation_note"} or item.baseline.material_type not in {"statement", "balance_sheet_policy", "implementation_note"}:
                    raise ValueError("QE/QT 必须引用官方资产负债表政策材料")
                current = _verify_ref(item.current, records, today, effective=True)
                baseline = _verify_ref(item.baseline, records, baseline_day, effective=True)
                quotes = [_compact(_quote(ref, raw)).lower() for ref, raw in ((item.current, current), (item.baseline, baseline))]
                if not all(any(term in quote for term in ("holdings", "monthly cap", "principal payments", "purchases", "reinvest")) for quote in quotes):
                    raise ValueError("QE/QT引文缺少资产负债表政策或操作速度，不能用调息替代")
                if quotes[0] == quotes[1] and item.change not in {"unchanged", "unknown"}:
                    raise ValueError("相同的QE/QT原文不能支持政策变化分类")
                if item.change == "unknown":
                    raise ValueError("资产负债表政策净方向不明")
                current_caps = qt_monthly_caps(current, today)
                baseline_caps = qt_monthly_caps(baseline, baseline_day)
                if current_caps is not None and baseline_caps is not None:
                    changes = [current_caps[key] - baseline_caps[key] for key in current_caps]
                    if any(value > 0 for value in changes) and any(value < 0 for value in changes):
                        raise ValueError("国债与机构证券缩表上限方向冲突，净方向不明")
                    expected = "accelerate_qt" if any(value > 0 for value in changes) else "slow_or_stop_qt" if any(value < 0 for value in changes) else "unchanged"
                    if not any(baseline_caps.values()) and any(current_caps.values()):
                        expected = "start_qt"
                    if item.change != expected:
                        raise ValueError("QE/QT分类与官方已生效月度缩表上限变化不一致")
                scores = {"start_qt": 0, "accelerate_qt": 2.5, "reduce_or_end_qe": 2.5, "unchanged": 5,
                          "slow_or_stop_qt": 7.5, "increase_existing_qe": 7.5, "start_policy_qe": 10}
                result = {
                    "current_policy": _citation(item.current, current), "baseline_policy": _citation(item.baseline, baseline),
                    "citations": [_citation(item.current, current), _citation(item.baseline, baseline)],
                    "policy_change": item.change, "interpretation": item.interpretation,
                    "comparison_date": baseline_day.isoformat(), "stale": False, "score": scores[item.change],
                    "current_monthly_caps": current_caps, "baseline_monthly_caps": baseline_caps,
                    "confidence": item.confidence, "unit": "policy_category",
                    "observed_at": item.current.effective_at.isoformat(), "source": current["url"],
                }
            results[metric_id] = {"status": "ok", **result}
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            results[metric_id] = {"status": "data_missing", "reason": str(exc)}
    return results
