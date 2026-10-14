"""运行环境配置；分析规则定义在 skills/ 下各 SKILL.md，用 pydantic-settings 从 .env 读取。

设计原则：
    - 所有"会变的值"（模型名、并发数、阈值、路径）都不写死在代码里
    - 通过环境变量覆盖，方便 Docker / 不同环境部署
    - 一处定义，全项目 import 复用（避免散落的 magic number）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]


def _env_files() -> tuple[str, ...]:
    """候选 .env 路径：当前工作目录、项目根目录及其上级。

    兼容三种运行方式：
        - 在项目目录跑（portfolio-projects/research-assistant/）
        - 在仓库根跑
        - 测试时从任意目录
    """
    # config.py 位于 app/config/；parents[1] 是当前项目根目录。
    here = Path(__file__).resolve().parent
    candidates = [
        Path(".env"),  # 当前工作目录
        # 按原顺序枚举项目根目录及其上级，容器内只取实际存在的父目录层级。
        *(parent / ".env" for parent in here.parents[1:4]),
    ]
    return tuple(str(p) for p in candidates)


class Settings(BaseSettings):
    """应用配置。优先从 .env 读取，缺省值保证开箱即用。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(),  # 多候选路径，pydantic 取第一个存在的
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM 模型（多模型路由：L06/L09 的成本优化）──────────────
    # 默认空串：允许 import 时（.env 还没加载）不崩；运行前由 cli/app 校验非空。
    zhipuai_api_key: str = Field(default="", description="智谱 API Key，必填")
    model_base_url: str = (
        "https://open.bigmodel.cn/api/paas/v4/"  # ChatOpenAI 调用智谱的兼容接口。
    )
    docs_dir: str = str(PROJECT_ROOT / "data" / "docs")
    # 会话状态数据库，可通过 CHECKPOINT_DB_PATH 环境变量覆盖。
    checkpoint_db_path: str = str(PROJECT_ROOT / "data" / "checkpoints.sqlite")
    # 路由候选采集单独开关，记录实际输出，不作为评测正确答案。
    routing_capture_enabled: bool = False
    routing_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "routing" / "routing_candidates.jsonl"
    )
    # 规划候选采集独立于路由，默认记录正常请求中的实际计划。
    planning_capture_enabled: bool = True
    planning_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "planning" / "planning_candidates.jsonl"
    )
    # 两类 Skill 候选独立采集，实际结果需人工确认后才能作为评测预期。
    skill_execution_capture_enabled: bool = True
    skill_execution_capture_path: str = str(
        PROJECT_ROOT
        / "eval_agent"
        / "skill_execution"
        / "skill_execution_candidates.jsonl"
    )
    skill_generation_capture_enabled: bool = True
    skill_generation_capture_path: str = str(
        PROJECT_ROOT
        / "eval_agent"
        / "skill_generation"
        / "skill_generation_candidates.jsonl"
    )

    # 决策/写作模型：贵但聪明，用在质量关键节点（summarize/writer/reviewer）
    smart_model: str = "glm-4.5-air"
    # 执行模型：免费快，用在并行量大、对质量要求不极致的节点（split/researcher）
    fast_model: str = "glm-4-flash"

    # 百炼向量模型通过 DashScope SDK 调用，使用独立的 API Key。
    dashscope_api_key: str = Field(
        default="",
        description="百炼向量、重排与视觉模型 API Key",
    )
    embedding_model: str = "text-embedding-v4"
    # 仅配置 FAISS 索引维度，不传给 SDK；必须与模型默认输出维度一致。
    embedding_dimensions: int = Field(default=1024, gt=0)
    # 更换模型后使用新目录重新入库，避免混用旧模型的向量。
    faiss_path: str = str(PROJECT_ROOT / "data" / "faiss" / "text-embedding-v4-1024")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    embed_batch_size: int = Field(default=10, ge=1, le=10)

    # PDF 表格表示与图片理解，可通过环境变量覆盖。
    table_format: Literal["markdown", "html"] = "markdown"
    vision_model: str = "qwen3.5-ocr"
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vision_timeout: float = Field(default=120.0, gt=0)
    vision_max_retries: int = Field(default=2, ge=1)

    # 分析规范、指标和权重定义在各自 SKILL.md 中。
    skills_dir: str = str(PROJECT_ROOT / "skills")

    model_price_table: dict[str, dict[str, float]] = {
        "glm-4.5-air": {"input": 50.0, "output": 50.0},
        "glm-4-flash": {"input": 0.0, "output": 0.0},  # flash 当前免费档
        "text-embedding-v4": {"input": 0.5, "output": 0.0},
        "qwen3.7-text-rerank": {"input": 0.5, "output": 0.0},
    }

    # ── 检索层（阶段 2）───────────────────────────────────────
    retrieve_k: int = 8  # 混合召回条数（rerank 前）
    final_k: int = 2  # 最终送入 prompt 的材料条数
    rerank_model: str = "qwen3.7-text-rerank"  # 百炼文本重排模型，与向量模型分别配置。
    bm25_weight: float = 0.4  # EnsembleRetriever 权重（与向量检索互补）
    vector_weight: float = 0.6
    enable_multi_query: bool = False  # 多视角问题改写会增加模型调用，默认关闭。
    retrieval_max_concurrency: int = Field(default=3, ge=1)  # 单进程检索并发上限。

    # 每个专题研究子问题的模型决策轮数上限。
    browser_max_steps: int = 12
    # 单页操作超时（秒）
    browser_page_timeout: int = 15
    # 无头模式（测试/CI 用 True，调试用 False）
    browser_headless: bool = True
    # 固定域名白名单，逗号分隔；空值允许默认域名及实际搜索结果中的域名。
    browser_domain_allowlist: str = ""


@lru_cache
def get_settings() -> Settings:
    """单例配置。lru_cache 保证全进程只读一次 .env。"""
    return Settings()  # type: ignore[call-arg]


# 便捷别名，全项目用 `from .config import settings`
settings = get_settings()
