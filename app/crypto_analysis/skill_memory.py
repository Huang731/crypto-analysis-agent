"""校验并合并同一会话内可复用的指标与理论材料。"""

import json
from datetime import date
from math import isfinite
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from app.crypto_analysis.skill_loader import LoadedSkill

# 共享范围属于程序配置，不由模型推断，也不跨会话共享。
_SKILL_SCOPES: dict[str, Literal["global", "asset"]] = {
    "analyzing-crypto-liquidity": "global",
    "analyzing-crypto-fed-policy": "global",
    "analyzing-crypto-sentiment": "global",
    "analyzing-crypto-etf-flows": "asset",
    "analyzing-crypto-derivatives": "asset",
    "analyzing-crypto-trend": "asset",
}


class SkillMemoryKey(BaseModel):
    """会话 State 内部的共享范围与技能身份。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    asset_scope: str = Field(pattern=r"^[A-Z0-9]+$")
    skill_name: str = Field(min_length=1)

    @property
    def cache_key(self) -> str:
        """会话隔离由 Checkpointer 的 thread_id 提供。"""
        return f"{self.asset_scope}:{self.skill_name}"


class IndicatorMemory(BaseModel):
    """仅保存已标准化的成功指标，零分也是有效结果。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    skill_name: str = Field(min_length=1)
    indicator_id: str = Field(min_length=1)
    status: Literal["ok"]
    reason: None
    raw_values: dict[str, Any]
    calculated_values: dict[str, Any]
    unit: str = Field(min_length=1)
    observed_at: str
    source: str = Field(min_length=1)
    url: str = Field(min_length=1)
    score: int | float

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value: Any) -> int | float:
        """拒绝布尔值、非有限值以及超出评分范围的数值。"""
        if (
            type(value) not in (int, float)
            or not isfinite(value)
            or not 0 <= value <= 10
        ):
            raise ValueError("缓存指标分数必须是有限的 0 至 10 数值")
        return value

    @field_validator("observed_at")
    @classmethod
    def validate_date(cls, value: str) -> str:
        """保留源数据观测日期，不用本轮分析时间替换。"""
        date.fromisoformat(value)
        return value


class ThoughtDocument(BaseModel):
    """完整保留检索正文与章节元数据。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | None
    content: str = Field(min_length=1)
    metadata: dict[str, Any]


class ThoughtMemory(BaseModel):
    """正常完成的检索可复用，空结果与检索异常分别处理。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1)
    status: Literal["ok", "no_results"]
    documents: list[ThoughtDocument]

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        """状态必须与实际文档数量一致。"""
        if (self.status == "ok") != bool(self.documents):
            raise ValueError("缓存检索状态与文档数量不一致")
        return self


class SkillMemoryEntry(SkillMemoryKey):
    """指标与 RAG 一起保存，Skill 聚合分数每轮重新计算。"""

    indicators: list[IndicatorMemory] = Field(min_length=1)
    thought_context: ThoughtMemory

    @model_validator(mode="after")
    def validate_indicators(self) -> Self:
        """缓存内指标必须属于当前技能且不能重复。"""
        ids = [item.indicator_id for item in self.indicators]
        if len(ids) != len(set(ids)) or any(
            item.skill_name != self.skill_name for item in self.indicators
        ):
            raise ValueError("缓存指标的技能归属或唯一性无效")
        return self


def resolve_memory_key(target_asset: str, skill_name: str) -> SkillMemoryKey:
    """按固定 Skill 映射选择全局或币种缓存，未知技能明确报错。"""
    if skill_name not in _SKILL_SCOPES:
        raise ValueError(f"Skill 未登记缓存共享范围：{skill_name}")
    asset_scope = "GLOBAL" if _SKILL_SCOPES[skill_name] == "global" else target_asset
    return SkillMemoryKey(asset_scope=asset_scope, skill_name=skill_name)


def merge_skill_memory(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """按键合并并行技能更新，不原地修改已有会话状态。"""
    return {**left, **right}


def read_skill_memory(
    memory: dict[str, dict[str, Any]], key: SkillMemoryKey
) -> dict[str, Any] | None:
    """只将键不存在视为未命中，已存在的损坏条目明确报错。"""
    if key.cache_key not in memory:
        return None
    value = memory[key.cache_key]
    if not isinstance(value, dict):
        raise TypeError(f"Skill 缓存条目不是有效对象：{key.cache_key}")
    return value


def validate_skill_memory(
    value: dict[str, Any], key: SkillMemoryKey, skill: LoadedSkill
) -> SkillMemoryEntry:
    """读写统一校验；损坏记录不能当作未命中重新取数。"""
    # 延迟读取现有字段规范，避免与工具 State 模块形成循环导入。
    from app.crypto_analysis.metric_service import _METRIC_FIELDS, _SOURCE_WEBSITES

    entry = SkillMemoryEntry.model_validate(value)
    if entry.asset_scope != key.asset_scope or entry.skill_name != key.skill_name:
        raise ValueError("缓存条目身份与本轮缓存键不一致")
    if {item.indicator_id for item in entry.indicators} != {
        metric.id for metric in skill.scoring.metrics
    }:
        raise ValueError("缓存指标集合与当前 Skill 不一致")
    for item in entry.indicators:
        raw_fields, calculated_fields = _METRIC_FIELDS[item.indicator_id]
        if any(item.raw_values.get(name) is None for name in raw_fields) or any(
            item.calculated_values.get(name) is None for name in calculated_fields
        ):
            raise ValueError(f"缓存指标字段不完整：{item.indicator_id}")
        source_host = urlsplit(item.source).hostname
        if (
            source_host not in _SOURCE_WEBSITES
            or item.url != _SOURCE_WEBSITES[source_host]
        ):
            raise ValueError(f"缓存指标来源无效：{item.indicator_id}")
    # 校验嵌套原始值、计算值与 metadata，确保 checkpoint 不含非有限数值。
    json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
    return entry
