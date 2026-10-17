"""运行环境配置；按 Agent 配置模型服务，分析规则保留在 skills/ 中。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_files() -> tuple[str, ...]:
    """依次加载工作目录、项目根目录及上级的 .env，后加载的值覆盖先加载的值。"""
    here = Path(__file__).resolve().parent
    candidates = [
        Path(".env"),
        *(parent / ".env" for parent in here.parents[1:4]),
    ]
    return tuple(str(path) for path in candidates)


class Settings(BaseSettings):
    """操作系统环境变量优先于 .env；未声明的环境配置不参与应用设置。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # 百炼连接：聊天、视觉、向量与重排共用；密钥只从环境变量或 .env 读取。
    dashscope_api_key: str = Field(default="", description="百炼 API Key")
    bailian_smart_model: str = "glm-5.1"
    bailian_fast_model: str = "qwen3.7-flash-2026-07-15"

    # GLM 模型使用智谱兼容接口，与百炼模型独立配置。
    zhipuai_api_key: str = Field(default="", description="智谱 API Key")
    glm_smart_model: str = "glm-4.5-air"
    glm_4_flash_model: str = "glm-4-flash"

    # 模型用量的成本估算：input/output 的单位均为元/百万 Token。
    # 保留历史评测模型的估算配置；未配置的模型成本按未知处理。
    model_price_table: dict[str, dict[str, float]] = {
        "glm-5.1": {"input": 50.0, "output": 50.0},
        "qwen3.7-flash-2026-07-15": {"input": 0.0, "output": 0.0},
        "text-embedding-v4": {"input": 0.5, "output": 0.0},
        "gte-rerank-v2": {"input": 0.5, "output": 0.0},
        "glm-4.5-air": {"input": 50.0, "output": 50.0},
        "glm-4-flash": {"input": 0.0, "output": 0.0},
    }

    # 文档入库：源文档目录、切片长度与重叠字符数。
    docs_dir: str = str(PROJECT_ROOT / "data" / "docs")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    table_format: Literal["markdown", "html"] = "markdown"
    # 图片描述与图像问答使用多模态原生接口；统一沿用模型超时与重试配置。
    vision_model: str = "qwen3.5-ocr"

    # 向量索引：维度必须与模型默认输出一致；更换向量模型后需重新入库。
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = Field(default=1024, gt=0)
    embed_batch_size: int = Field(default=10, ge=1, le=10)
    faiss_path: str = str(PROJECT_ROOT / "data" / "faiss" / "text-embedding-v4-1024")

    # 混合检索：先融合 BM25 与向量召回结果，再由重排模型选出最终材料。
    retrieve_k: int = 8
    final_k: int = 2
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    rerank_model: str = "gte-rerank-v2"
    # 多查询会增加改写调用；并发上限约束单个共享检索实例的正在执行任务。
    enable_multi_query: bool = False
    retrieval_max_concurrency: int = Field(default=3, ge=1)

    # 技能定义与会话持久化：分析规则来自 Skill，会话状态保存到 SQLite。
    skills_dir: str = str(PROJECT_ROOT / "skills")
    checkpoint_db_path: str = str(PROJECT_ROOT / "data" / "checkpoints.sqlite")

    # LLM 候选采集：保存节点名称及模型输入输出，按配置路径写入。
    routing_capture_enabled: bool = True
    routing_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "routing_candidates.jsonl"
    )
    planning_capture_enabled: bool = True
    planning_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "planning_candidates.jsonl"
    )
    rewrite_capture_enabled: bool = True
    rewrite_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "rewrite_candidates.jsonl"
    )
    calculate_metrics_capture_enabled: bool = True
    calculate_metrics_capture_path: str = str(
        PROJECT_ROOT
        / "eval_agent"
        / "candidates"
        / "calculate_metrics_candidates.jsonl"
    )
    generate_content_capture_enabled: bool = True
    generate_content_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "generate_content_candidates.jsonl"
    )
    writer_capture_enabled: bool = True
    writer_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "writer_candidates.jsonl"
    )
    reviewer_capture_enabled: bool = True
    reviewer_capture_path: str = str(
        PROJECT_ROOT / "eval_agent" / "candidates" / "reviewer_candidates.jsonl"
    )
    summarize_history_capture_enabled: bool = True
    summarize_history_capture_path: str = str(
        PROJECT_ROOT
        / "eval_agent"
        / "candidates"
        / "summarize_history_candidates.jsonl"
    )

    # 浏览器取证：单页超时单位为秒。
    browser_page_timeout: int = 15
    browser_headless: bool = True


@lru_cache
def get_settings() -> Settings:
    """每个进程只加载一次配置。"""
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
