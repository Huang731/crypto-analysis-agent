"""基于用户提供的 SkillLoader：目录索引、按名加载与评分规则解析。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config.config import settings
from app.crypto_analysis.tools.registry import get_tool_registry
from app.observability import get_logger

log = get_logger("skills")


class _UniqueSafeLoader(yaml.SafeLoader):
    """拒绝重复字段，避免同一权重出现多个解释。"""

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML 字段必须是唯一字符串")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _parse_document(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 缺少 frontmatter")
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        raise ValueError("SKILL.md frontmatter 未闭合")
    header = "".join(lines[1:closing])
    if any(isinstance(event, yaml.events.AliasEvent) for event in yaml.parse(header)):
        raise ValueError("Skill frontmatter 不支持 YAML 别名")
    data = yaml.load(header, Loader=_UniqueSafeLoader)
    if not isinstance(data, dict) or set(data) != {
        "name",
        "description",
        "tools",
        "scoring",
    }:
        raise ValueError("frontmatter 必须仅包含 name、description、tools 和 scoring")
    name, description = data["name"], data["description"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("Skill name 格式无效")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Skill description 不能为空")
    body = "".join(lines[closing + 1 :]).strip()
    if not body:
        raise ValueError("Skill 正文不能为空")
    return data, body


class MetricRule(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, strict=True)
    label: str = Field(min_length=1, strict=True)
    weight: float = Field(gt=0, le=1, allow_inf_nan=False)
    kind: Literal["numeric", "text"]

    @field_validator("weight", mode="before")
    @classmethod
    def numeric_weight(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("指标权重必须是数值")  # noqa: TRY004 - Pydantic 捕获 ValueError。
        return value


class SkillScoring(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, strict=True)
    metrics: list[MetricRule] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_metrics(self):
        ids = [metric.id for metric in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Skill 指标重复")
        if abs(sum(metric.weight for metric in self.metrics) - 1) > 1e-6:
            raise ValueError("Skill 指标权重总和必须为 1")
        return self


class LoadedSkill(BaseModel):
    name: str
    description: str
    tools: list[str] = Field(min_length=1)
    scoring: SkillScoring

    @field_validator("tools")
    @classmethod
    def valid_tools(cls, names: list[str]) -> list[str]:
        """加载时拒绝重复或未注册工具，不补充默认工具。"""
        if len(names) != len(set(names)):
            raise ValueError("Skill 工具名称重复")
        unknown = set(names) - get_tool_registry().keys()
        if unknown:
            raise ValueError(f"Skill 包含未注册工具：{'、'.join(sorted(unknown))}")
        return names


@dataclass
class SkillMeta:
    """Skill 的元信息（轻量，进 system prompt 用）。"""

    name: str
    description: str  # 一行描述
    path: Path | None  # 快照恢复时无需磁盘路径


class SkillLoader:
    """先提供描述索引，再按 LLM 选择的名称加载全文。"""

    def __init__(self, skills_dir: str | Path | None = None):
        self._skills_dir = Path(skills_dir) if skills_dir else self._default_dir()
        self._cache: dict[str, str] = {}  # name → 全文（加载后缓存）
        self._metas: list[SkillMeta] = []
        self._sources: dict[str, str] = {}
        self._definitions: dict[str, LoadedSkill] = {}
        self._scan()

    def _default_dir(self) -> Path:
        return Path(settings.skills_dir)

    def _scan(self):
        """扫描 skills 目录，读取每个 skill 的元信息。"""
        if not self._skills_dir.exists():
            raise ValueError(f"skills 目录不存在：{self._skills_dir}")

        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = self._parse_frontmatter(skill_md, skill_dir)
            if meta:
                self._metas.append(meta)

        if not self._metas:
            raise ValueError(f"skills 目录没有可用技能：{self._skills_dir}")

        log.info(f"skills 扫描完成：{len(self._metas)} 个 skill @ {self._skills_dir}")

    def _parse_frontmatter(self, skill_md: Path, skill_dir: Path) -> SkillMeta:
        text = skill_md.read_text(encoding="utf-8-sig")
        return self._register(text, skill_dir.name, skill_dir)

    def _register(self, text: str, expected_name: str, path: Path | None) -> SkillMeta:
        try:
            data, _ = _parse_document(text)
            name = data["name"]
            if name != expected_name or name in self._sources:
                raise ValueError("Skill 名称必须与目录一致且不重复")
            self._sources[name] = text
            return SkillMeta(
                name=name, description=data["description"].strip(), path=path
            )
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Skill {expected_name} 加载失败：{exc}") from exc

    def snapshot(self) -> dict[str, str]:
        return dict(self._sources)

    @classmethod
    def from_snapshot(cls, sources: dict[str, str]) -> SkillLoader:
        loader = cls.__new__(cls)
        loader._skills_dir = Path(settings.skills_dir)
        loader._cache = {}
        loader._sources = {}
        loader._definitions = {}
        loader._metas = [
            loader._register(text, name, None) for name, text in sources.items()
        ]
        return loader

    def load_definition(self, name: str) -> LoadedSkill:
        if name not in self._definitions:
            try:
                if name not in self._sources:
                    raise ValueError(f"skill '{name}' 不存在")
                data, _ = _parse_document(self._sources[name])
                self._definitions[name] = LoadedSkill(
                    name=data["name"],
                    description=data["description"],
                    tools=data["tools"],
                    scoring=SkillScoring.model_validate(data["scoring"]),
                )
            except (ValueError, yaml.YAMLError) as exc:
                raise ValueError(f"Skill {name} 配置无效：{exc}") from exc
        return self._definitions[name]

    def load_analysis_output(self, name: str) -> str:
        """从请求快照提取完整的分析输出章节，供技能生成表格使用。"""
        _, body = _parse_document(self.load_skill(name))
        lines = body.splitlines()
        try:
            start = lines.index("## 分析输出")
        except ValueError as exc:
            raise ValueError(f"Skill {name} 缺少分析输出章节") from exc
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
            len(lines),
        )
        if not any(line.strip() for line in lines[start + 1 : end]):
            raise ValueError(f"Skill {name} 分析输出章节为空")
        return "\n".join(lines[start:end])

    def validate_selected(self, names: Sequence[str]) -> None:
        metric_ids = set()
        for name in names:
            scoring = self.load_definition(name).scoring
            ids = {metric.id for metric in scoring.metrics}
            if metric_ids & ids:
                raise ValueError("已选 Skill 的指标归属冲突")
            metric_ids.update(ids)

    def list_skills(self) -> list[SkillMeta]:
        """返回所有 skill 的元信息（轻量，进 system prompt 用）。"""
        return list(self._metas)

    def format_skill_descriptions(self) -> str:
        """把所有 skill 的一行描述格式化成 system prompt 片段。

        这是渐进式披露的第一层：Agent 只看到描述，不看全文。
        """
        if not self._metas:
            return ""
        lines = ["【可用技能 Skills】"]
        for m in self._metas:
            lines.append(f"  - {m.name}: {m.description}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        """加载某个 skill 的全文（渐进式披露的第二层：用到才加载）。

        使用请求快照，首次加载后缓存，避免混用文件版本。
        """
        if name in self._cache:
            return self._cache[name]

        if name not in self._sources:
            raise ValueError(f"skill '{name}' 不存在")
        text = self._sources[name]
        self._cache[name] = text
        log.info(f"加载 skill 全文：{name}（{len(text)} 字符）")
        return text

    # 中文停用词片段：这些 2 字片段太常见，匹配它们会误命中大量 skill
    _STOP_SEGS: ClassVar[set[str]] = {
        "研究",
        "生成",
        "流程",
        "规范",
        "结构",
        "层级",
        "方式",
        "多个",
        "呈现",
        "差异",
        "规定",
        "标题",
        "标注",
    }

    def match_skills(self, query: str) -> list[str]:
        """根据 query 匹配需要加载的 skill（简单关键词匹配）。

        生产可换成 LLM 判断（"这个任务需要哪些 skill"）。
        本课用关键词匹配：query 含 skill name 或 description 关键词 → 命中。

        中文处理：2-4 字片段匹配，但排除停用词（如"研究""流程"等泛化词）。
        """
        matched = []
        query_lower = query.lower()
        for m in self._metas:
            # name 出现在 query 里 → 命中
            if m.name.lower() in query_lower:
                matched.append(m.name)
                continue
            # description 的片段出现在 query 里 → 命中
            desc = m.description.lower()
            segments = set()
            for length in (2, 3, 4):
                for i in range(len(desc) - length + 1):
                    seg = desc[i : i + length]
                    if seg.strip() and seg not in self._STOP_SEGS:
                        segments.add(seg)
            # 英文：按空格分词
            for w in desc.split():
                if len(w) > 2:
                    segments.add(w.lower())
            if any(seg in query_lower for seg in segments):
                matched.append(m.name)
        return matched

    def load_matched_skills(self, query: str) -> str:
        """一步到位：匹配 + 加载全文，返回拼好的 skill 内容。

        writer 节点用这个：传入研究主题/摘要 → 返回匹配 skill 的全文。
        """
        matched = self.match_skills(query)
        if not matched:
            return ""
        parts = []
        for name in matched:
            content = self.load_skill(name)
            if content:
                parts.append(f"── 技能 {name} ──\n{content}")
        return "\n\n".join(parts)


def get_skill_loader() -> SkillLoader:
    """每个请求独立加载，避免全局缓存跨请求过期。"""
    return SkillLoader(settings.skills_dir)
