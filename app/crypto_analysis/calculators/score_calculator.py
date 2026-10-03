"""维度指标共用的线性评分公式。"""

from decimal import Decimal  # 保留原有十进制计算精度。


def linear_score(
    value: float | Decimal, low: float, high: float
) -> float:  # 定义该指标的程序计算函数。
    """把未舍入的指标百分数映射为 0 到 10 分。"""
    number = Decimal(str(value))  # 计算或保存待评分的十进制数。
    lower = Decimal(str(low))  # 计算或保存评分下界。
    upper = Decimal(str(high))  # 计算或保存评分上界。
    score = (number - lower) / (upper - lower) * 10  # 计算或保存指标分数。
    return float(min(Decimal(10), max(Decimal(0), score)))  # 返回本步骤的结果。
