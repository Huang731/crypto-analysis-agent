"""读取维度机器配置，并校验配置自身的一致性。"""

import json  # 读取标准 JSON 配置。
from pathlib import Path  # 定位项目内的配置文件。
from typing import Literal  # 限定指标类型和接入状态。

from pydantic import BaseModel, ConfigDict, Field  # 校验配置字段。


_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "data" / "config"  # 配置目录。


class MetricConfig(BaseModel):  # 一项指标的机器配置。
    model_config = ConfigDict(extra="forbid")  # 拒绝拼错或未定义的字段。
    id: str  # 指标标识。
    weight: float = Field(gt=0, le=1)  # 指标在本维度内的原权重。
    kind: Literal["numeric", "text"]  # 数值计算或文本判断。
    status: Literal["ready", "pending"]  # 工具是否已接入。
    allowed_tools: list[str]  # LLM 可选择的数据工具。
    allowed_calculators: list[str]  # LLM 可选择的计算器。
    raw_fields: list[str] = Field(default_factory=list)  # 报告需要保留的原始字段。
    calculated_fields: list[str] = Field(default_factory=list)  # 报告需要保留的计算字段。
    source_action: str | None = None  # 工具成功时记录的调用名。


class DimensionConfig(BaseModel):  # 一个维度的执行配置。
    model_config = ConfigDict(extra="forbid")  # 配置只接受已定义字段。
    dimension_id: str  # 与总配置一致的两位维度编号。
    name: str  # 维度显示名称。
    min_coverage: float = Field(ge=0, le=1)  # 正式得分所需覆盖率。
    required_metrics: list[str]  # 必须有效的专属指标。
    metrics: list[MetricConfig] = Field(min_length=1)  # 本维度全部必算指标。


class DimensionCatalog(BaseModel):  # 一轮分析使用的完整配置快照。
    enabled_dimensions: list[str]  # 按总配置顺序执行的维度。
    dimension_weights: dict[str, float]  # 维度综合权重。
    dimensions: dict[str, DimensionConfig]  # 每维的指标配置。


def load_dimension_catalog() -> DimensionCatalog:  # 每轮读取一次总配置和维度配置。
    global_path = _CONFIG_ROOT / "analysis.json"  # 总配置沿用项目现有文件。
    global_data = json.loads(global_path.read_text(encoding="utf-8"))  # UTF-8 读取总配置。
    enabled = global_data["enabled_dimensions"]  # 取得本轮启用的维度编号。
    weights = global_data["dimension_weights"]  # 取得本轮固定的维度权重。
    if not isinstance(enabled, list) or not enabled or len(enabled) != len(set(enabled)):  # 检查编号列表。
        raise ValueError("config_error：enabled_dimensions 必须非空且无重复")  # 配置无效时停止。
    if set(enabled) != set(weights):  # 检查总权重与启用维度是否一致。
        raise ValueError("config_error：dimension_weights 与启用维度不一致")  # 不猜测缺失权重。
    if abs(sum(weights.values()) - 1.0) > 0.000001:  # 维度权重须合计为一。
        raise ValueError("config_error：维度权重总和不为 1")  # 不临时归一化配置错误。
    dimensions: dict[str, DimensionConfig] = {}  # 收集本轮维度配置。
    for dimension_id in enabled:  # 按总配置顺序读取每维文件。
        path = _CONFIG_ROOT / "dimensions" / f"{dimension_id}.json"  # 每维独立配置路径。
        dimension = DimensionConfig.model_validate_json(path.read_text(encoding="utf-8"))  # 校验字段类型。
        if dimension.dimension_id != dimension_id:  # 文件编号必须与索引相同。
            raise ValueError(f"config_error：维度 {dimension_id} 的配置编号不一致")  # 防止串维。
        ids = [metric.id for metric in dimension.metrics]  # 提取本维指标标识。
        if len(ids) != len(set(ids)):  # 指标不能重复。
            raise ValueError(f"config_error：维度 {dimension_id} 的指标重复")  # 防止重复计权。
        if abs(sum(metric.weight for metric in dimension.metrics) - 1.0) > 0.000001:  # 指标权重合计为一。
            raise ValueError(f"config_error：维度 {dimension_id} 的指标权重总和不为 1")  # 停止错误评分。
        if not set(dimension.required_metrics) <= set(ids):  # 必需指标必须在指标清单内。
            raise ValueError(f"config_error：维度 {dimension_id} 的必需指标不存在")  # 防止门槛失效。
        for metric in dimension.metrics:  # 分别校验已接入和未接入指标。
            if metric.status == "ready" and (not metric.allowed_tools or not metric.allowed_calculators):  # 已接入须可执行。
                raise ValueError(f"config_error：{dimension_id}.{metric.id} 缺少工具或计算器")  # 不虚构接入状态。
            if metric.status == "pending" and (metric.allowed_tools or metric.allowed_calculators):  # 未接入不得给出伪选择。
                raise ValueError(f"config_error：{dimension_id}.{metric.id} 的待接入配置不应列工具")  # 保留真实缺失。
        dimensions[dimension_id] = dimension  # 加入本轮配置快照。
    return DimensionCatalog(enabled_dimensions=enabled, dimension_weights=weights, dimensions=dimensions)  # 返回完整配置。
