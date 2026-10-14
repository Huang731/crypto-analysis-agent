"""校验并合并同会话内可复用的技能表格与思想原文。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class SkillMemoryContent(BaseModel):
    """在原字符串缓存中保存指标表格和思想原文。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    content: str = Field(min_length=1)
    thought_context: dict[str, Any]


def resolve_memory_key(target_asset: str, skill_name: str) -> SkillMemoryKey:
    """按固定 Skill 映射选择全局或币种缓存，未知技能明确报错。"""
    if skill_name not in _SKILL_SCOPES:
        raise ValueError(f"Skill 未登记缓存共享范围：{skill_name}")
    asset_scope = "GLOBAL" if _SKILL_SCOPES[skill_name] == "global" else target_asset
    return SkillMemoryKey(asset_scope=asset_scope, skill_name=skill_name)


def validate_skill_memory(value: str, key: SkillMemoryKey) -> str:
    """校验缓存字符串非空，表格与思想载荷由 SkillMemoryContent 校验。"""
    if not isinstance(value, str):
        raise TypeError(f"Skill 缓存条目不是字符串：{key.cache_key}")
    if not value.strip():
        raise ValueError(f"Skill 缓存内容为空：{key.cache_key}")
    return value


def merge_skill_memory(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """按键合并并行技能更新，不原地修改已有会话状态。"""
    return {**left, **right}


def read_skill_memory(memory: dict[str, str], key: SkillMemoryKey) -> str | None:
    """只有键不存在才是未命中，已存在的错误内容直接报错。"""
    if key.cache_key not in memory:
        return None
    return validate_skill_memory(memory[key.cache_key], key)
