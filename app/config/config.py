"""配置中心：所有可配置项集中在这里，用 pydantic-settings 从 .env 读取。

设计原则：
    - 所有"会变的值"（模型名、并发数、阈值、路径）都不写死在代码里
    - 通过环境变量覆盖，方便 Docker / 不同环境部署
    - 一处定义，全项目 import 复用（避免散落的 magic number）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
        here.parents[1] / ".env",  # 当前项目根目录
        here.parents[2] / ".env",  # 项目上级目录
        here.parents[3] / ".env",  # 更上一级目录
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

    # 决策/写作模型：贵但聪明，用在质量关键节点（summarize/writer/reviewer）
    smart_model: str = "glm-4.5-air"
    # 执行模型：免费快，用在并行量大、对质量要求不极致的节点（split/researcher）
    fast_model: str = "glm-4-flash"

    # 百炼向量模型通过 DashScope SDK 调用，使用独立的 API Key。
    dashscope_api_key: str = Field(
        default="sk-ws-H.PDMPHML.YRf1.MEQCIB1uZlBLpuHaPGQICan6bo3YokH41UETNoNUAKA2kGdzAiBOuh2mmfyqUkljcbWSlldwJeNRdNLMmcTV_NE3aZqNxg",
        description="百炼向量模型 API Key",
    )
    embedding_model: str = "text-embedding-v4"
    # 仅配置 FAISS 索引维度，不传给 SDK；必须与模型默认输出维度一致。
    embedding_dimensions: int = Field(default=1024, gt=0)
    # 更换模型后使用新目录重新入库，避免混用旧模型的向量。
    faiss_path: str = str(PROJECT_ROOT / "data" / "faiss" / "text-embedding-v4-1024")
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    embed_batch_size: int = Field(default=10, ge=1, le=10)

    dimensions: dict[str, list[str]] = {
        "美元流动性与加密内部流动性": [
            "DXY_Trend",
            "Stable_MCap",
            "M2_YoY",
            "Fed_Balance",
        ]
    }

    model_price_table: dict[str, dict[str, float]] = {
        "glm-4.5-air": {"input": 50.0, "output": 50.0},
        "glm-4-flash": {"input": 0.0, "output": 0.0},  # flash 当前免费档
        "text-embedding-v4": {"input": 0.5, "output": 0.0},
        "gte-rerank-v2": {"input": 0.5, "output": 0.0},
    }

    # ── 检索层（阶段 2）───────────────────────────────────────
    retrieve_k: int = 8  # 混合召回条数（rerank 前）
    final_k: int = 4  # 最终送入 prompt 的材料条数
    rerank_model: str = "gte-rerank-v2"  # 百炼文本重排模型，与向量模型分别配置。
    bm25_weight: float = 0.4  # EnsembleRetriever 权重（与向量检索互补）
    vector_weight: float = 0.6
    enable_multi_query: bool = False  # 多视角问题改写会增加模型调用，默认关闭。

    # 浏览器 agent 循环步数上限（成本控制，每步一次 LLM+一次浏览器操作）
    browser_max_steps: int = 12
    # 单页操作超时（秒）
    browser_page_timeout: int = 15
    # 无头模式（测试/CI 用 True，调试用 False）
    browser_headless: bool = True
    # 域名 allowlist（L07 安全：逗号分隔，空=用默认 DEFAULT_ALLOWED_DOMAINS）
    browser_domain_allowlist: str = ""


@lru_cache
def get_settings() -> Settings:
    """单例配置。lru_cache 保证全进程只读一次 .env。"""
    return Settings()  # type: ignore[call-arg]


# 便捷别名，全项目用 `from .config import settings`
settings = get_settings()
