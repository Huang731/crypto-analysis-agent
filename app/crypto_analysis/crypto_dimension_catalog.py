"""从 Settings 创建维度配置快照，并校验配置自身的一致性。"""

from typing import Literal  # 限定指标类型和接入状态。

from pydantic import BaseModel, ConfigDict, Field  # 校验配置字段。

from app.config.config import settings


class MetricConfig(BaseModel):  # 一项指标的机器配置。
    model_config = ConfigDict(extra="forbid")  # 拒绝拼错或未定义的字段。
    id: str  # 指标标识。
    label: str = Field(min_length=1)  # 用于思想体系检索的指标显示名称。
    weight: float = Field(gt=0, le=1)  # 指标在本维度内的原权重。
    kind: Literal["numeric", "text"]  # 数值计算或文本判断。


class DimensionConfig(BaseModel):  # 一个维度的执行配置。
    model_config = ConfigDict(extra="forbid")  # 配置只接受已定义字段。
    dimension_id: str  # 与总配置一致的两位维度编号。
    name: str  # 维度显示名称。
    min_coverage: float = Field(ge=0, le=1)  # 正式得分所需覆盖率。
    metrics: list[MetricConfig] = Field(min_length=1)  # 本维度全部必算指标。


class DimensionCatalog(BaseModel):  # 一轮分析使用的完整配置快照。
    enabled_dimensions: list[str]  # 按总配置顺序执行的维度。
    dimension_weights: dict[str, float]  # 维度综合权重。
    dimensions: dict[str, DimensionConfig]  # 每维的指标配置。


def load_dimension_catalog() -> DimensionCatalog:
    """每轮读取启用维度，并创建独立配置快照。"""
    enabled = list(settings.enabled_dimensions)
    weights = dict(settings.dimension_weights)
    if (
        not isinstance(enabled, list)
        or not enabled
        or len(enabled) != len(set(enabled))
    ):  # 检查编号列表。
        raise ValueError(
            "config_error：enabled_dimensions 必须非空且无重复"
        )  # 配置无效时停止。
    if set(enabled) != set(weights):  # 检查总权重与启用维度是否一致。
        raise ValueError(
            "config_error：dimension_weights 与启用维度不一致"
        )  # 不猜测缺失权重。
    if abs(sum(weights.values()) - 1.0) > 0.000001:  # 维度权重须合计为一。
        raise ValueError("config_error：维度权重总和不为 1")  # 不临时归一化配置错误。
    dimensions: dict[str, DimensionConfig] = {}  # 收集本轮维度配置。
    for dimension_id in enabled:
        dimension = DimensionConfig.model_validate(settings.dimensions[dimension_id])
        if dimension.dimension_id != dimension_id:  # 文件编号必须与索引相同。
            raise ValueError(
                f"config_error：维度 {dimension_id} 的配置编号不一致"
            )  # 防止串维。
        ids = [metric.id for metric in dimension.metrics]  # 提取本维指标标识。
        if len(ids) != len(set(ids)):  # 指标不能重复。
            raise ValueError(
                f"config_error：维度 {dimension_id} 的指标重复"
            )  # 防止重复计权。
        if (
            abs(sum(metric.weight for metric in dimension.metrics) - 1.0) > 0.000001
        ):  # 指标权重合计为一。
            raise ValueError(
                f"config_error：维度 {dimension_id} 的指标权重总和不为 1"
            )  # 停止错误评分。
        dimensions[dimension_id] = dimension  # 加入本轮配置快照。
    return DimensionCatalog(
        enabled_dimensions=enabled, dimension_weights=weights, dimensions=dimensions
    )  # 返回完整配置。
