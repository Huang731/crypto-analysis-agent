"""图表与图片理解：百炼 Qwen-VL-OCR 的两段式消费。

两段式设计（成本-精度主线在此）：
    ① 入库时 describe_image：VLM 提结构化描述（轴标签/数值/趋势），做索引 + 缓存
    ② 问答时 answer_with_image：检索命中图表后，现场看图作答（忠实但花钱）

为什么两段式：
    - 只用描述（入库时生成）：检索友好但可能丢细节（描述写不下的数字）
    - 只现场看图（问答时）：忠实但每问都花钱（重复问同一张图反复调 VLM）
    - 两段式：描述负责「被搜到」，原图负责「答得准」，钱花在刀刃上

缓存：describe_image 的结果按图片内容哈希落盘，重复入库不重复调 VLM。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from app.config.config import settings
from prompts.rag_vision import answer_with_image_prompt, describe_image_prompt

# 描述缓存目录（落盘，按图片哈希去重）
_CACHE_DIR = Path(settings.docs_dir).parent / "vision_cache"


def _make_vision_model() -> ChatOpenAI:
    """通过百炼兼容接口创建视觉模型，设置超时与 SDK 重试。"""
    api_key = settings.dashscope_api_key.strip()
    if not api_key:
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    base_url = settings.vision_base_url.strip()
    if not base_url:
        raise ValueError("未配置 VISION_BASE_URL")
    return ChatOpenAI(
        model=settings.vision_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        timeout=settings.vision_timeout,
        max_retries=settings.vision_max_retries,
        max_tokens=4096,
        use_responses_api=False,
    )


def _image_hash(image_bytes: bytes) -> str:
    """图片内容的 MD5，作缓存 key（去重依据）。"""
    return hashlib.md5(image_bytes).hexdigest()


def _cache_path(img_hash: str) -> Path:
    """缓存文件路径：vision_cache/{hash}.json。"""
    return _CACHE_DIR / f"{img_hash}.json"


def describe_image(
    image_bytes: bytes,
    *,
    force_refresh: bool = False,
    use_mock: bool = False,
) -> str:
    """用百炼视觉模型给图片生成结构化描述（带哈希缓存）。

    缓存逻辑：按图片内容 MD5 落盘到 vision_cache/，重复入库命中缓存不调 VLM。
    force_refresh=True 时忽略缓存强制重新描述（描述质量抽查用）。
    use_mock=True 时不调 VLM，返回预录描述（教学/测试用，省 token）。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    img_hash = _image_hash(image_bytes)
    cache_file = _cache_path(img_hash)

    # ① 命中缓存直接返回（重复入库不重复调 VLM）
    if not force_refresh and cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data["description"]

    # ② 生成描述（真调 VLM）
    desc = _call_vlm_describe(image_bytes)

    # ③ 落盘缓存
    cache_file.write_text(
        json.dumps(
            {"hash": img_hash, "description": desc}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return desc


def _call_vlm_describe(image_bytes: bytes) -> str:
    """调用百炼视觉模型生成描述（需 API key）。

    使用百炼 OpenAI 兼容接口，多模态消息格式：
    [{"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]}]
    """
    import base64

    # 图片转 base64 data URL，避免上传临时文件。
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    vlm = _make_vision_model()
    resp = vlm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": describe_image_prompt()},
                ],
            }
        ]
    )
    if not isinstance(resp.content, str):
        raise TypeError("百炼视觉模型未返回文本内容")
    return resp.content.strip()


def answer_with_image(
    image_bytes: bytes,
    question: str,
    *,
    use_mock: bool = False,
) -> str:
    """现场看图作答：把图片与问题交给百炼视觉模型，返回答案。

    这是两段式的第二段——检索命中图表元素后，用原图（而非描述）作答。
    优势：忠实（直接看图，不受描述质量限制）；劣势：每次调用都花钱。
    use_mock=True 时返回占位答案（教学/测试用）。
    """
    if use_mock:
        return (
            "[mock 答案] 需配置 DASHSCOPE_API_KEY 调百炼视觉模型现场看图。"
            f"问题：{question}"
        )

    import base64

    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    vlm = _make_vision_model()
    resp = vlm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {
                        "type": "text",
                        "text": answer_with_image_prompt(question),
                    },
                ],
            }
        ]
    )
    if not isinstance(resp.content, str):
        raise TypeError("百炼视觉模型未返回文本内容")
    return resp.content.strip()


def is_cached(image_bytes: bytes) -> bool:
    """检查图片是否已有缓存（诊断用：看缓存命中率）。"""
    return _cache_path(_image_hash(image_bytes)).exists()


def clear_cache() -> int:
    """清空描述缓存（测试/重新描述用）。返回清除的文件数。"""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for f in _CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
