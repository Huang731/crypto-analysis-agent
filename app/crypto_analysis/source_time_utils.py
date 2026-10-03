"""解析外部数据接口返回的 UTC 时间。"""

from datetime import date, datetime, timezone  # 统一处理日期和 UTC 时间。
from typing import Any  # 接收外部接口的原始时间字段。


def utc_time(value: Any) -> datetime:  # 将 CMC 原始时间规范化为 UTC。
    """解析 CMC 返回的 ISO 时间或 Unix 秒时间。"""
    if isinstance(value, str) and value.isdigit():  # 纯数字字符串表示 Unix 秒。
        result = datetime.fromtimestamp(int(value), timezone.utc)  # 转成带时区的时间。
    elif isinstance(value, str):  # 其他字符串按 ISO 时间处理。
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )  # 接受 Z 形式的 UTC 后缀。
    else:  # 非字符串不是约定的时间格式。
        raise ValueError("CMC 观测时间格式无效")  # 明确拒绝错误响应。
    if result.tzinfo is None:  # 接口时间必须带时区。
        raise ValueError("CMC 观测时间缺少时区")  # 防止本地时区误判。
    return result.astimezone(timezone.utc)  # 返回统一 UTC 时间。


def shift_month(month: date, offset: int) -> date:  # 按自然月移动 FRED 请求或计算窗口。
    """按自然月移动，返回目标月份第一天。"""
    index = month.year * 12 + month.month - 1 + offset  # 将年月转换为连续月份编号。
    return date(index // 12, index % 12 + 1, 1)  # 将编号还原为年月第一天。
