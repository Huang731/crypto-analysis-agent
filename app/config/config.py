"""配置中心：所有可配置项集中在这里，用 pydantic-settings 从 .env 读取。

设计原则：
    - 所有"会变的值"（模型名、并发数、阈值、路径）都不写死在代码里
    - 通过环境变量覆盖，方便 Docker / 不同环境部署
    - 一处定义，全项目 import 复用（避免散落的 magic number）
"""
from __future__ import annotations

from pathlib import Path

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
        Path(".env"),               # 当前工作目录
        here.parents[1] / ".env",   # 当前项目根目录
        here.parents[2] / ".env",   # 项目上级目录
        here.parents[3] / ".env",   # 更上一级目录
    ]
    return tuple(str(p) for p in candidates)


class Settings(BaseSettings):
    """应用配置。优先从 .env 读取，缺省值保证开箱即用。"""

    model_config = SettingsConfigDict(
        env_file=_env_files(),       # 多候选路径，pydantic 取第一个存在的
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM 模型（多模型路由：L06/L09 的成本优化）──────────────
    # 默认空串：允许 import 时（.env 还没加载）不崩；运行前由 cli/app 校验非空。
    zhipuai_api_key: str = Field(default="", description="智谱 API Key，必填")
    model_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"  # ChatOpenAI 调用智谱的兼容接口。

    # 决策/写作模型：贵但聪明，用在质量关键节点（summarize/writer/reviewer）
    smart_model: str = "glm-4.5-air"
    # 执行模型：免费快，用在并行量大、对质量要求不极致的节点（split/researcher）
    fast_model: str = "glm-4-flash"

    dimensions: dict[str, list[str]] = {"美元流动性与加密内部流动性": ["DXY_Trend", "Stable_MCap", "M2_YoY", "Fed_Balance"]}

@lru_cache
def get_settings() -> Settings:
    """单例配置。lru_cache 保证全进程只读一次 .env。"""
    return Settings()  # type: ignore[call-arg]


# 便捷别名，全项目用 `from .config import settings`
settings = get_settings()
