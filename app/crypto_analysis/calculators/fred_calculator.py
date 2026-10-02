"""FRED 原始观测的校验、指标计算与评分。"""

from calendar import monthrange  # 核对月末及观测有效期。
from datetime import date, timedelta  # 计算月度与周度窗口。
from decimal import Decimal, InvalidOperation  # 保留原有精度和格式校验。
from typing import Any  # 接收 FRED 原始响应。

from app.crypto_analysis.calculators.score_calculator import (
    linear_score,  # 使用原有线性评分口径。
)
from app.crypto_analysis.source_time_utils import (
    shift_month,  # 与 FRED 请求共享自然月换算规则。
)

_FRED_SOURCE = "https://api.stlouisfed.org/fred/series/observations"  # 保留原有结果来源标识。

def calculate_m2_yoy_change(payload: dict[str, Any], as_of: date) -> dict[str, Any]:  # 定义该指标的程序计算函数。
    """用 M2SL 原始月值计算最新月同比与上月同比之差。"""
    rows = payload.get("observations")  # 计算或保存原始观测列表。
    if not isinstance(rows, list):  # 校验当前条件并决定后续计算。
        raise TypeError("FRED 响应缺少 observations 数组")  # 拒绝无效或不足的数据。
    values: dict[date, Decimal] = {}  # 计算或保存按日期整理的有效观测。
    for row in rows:  # 依次处理原始数据中的记录。
        if not isinstance(row, dict):  # 校验当前条件并决定后续计算。
            raise TypeError("FRED 月度观测格式无效")  # 拒绝无效或不足的数据。
        try:  # 开始解析外部数据字段。
            observed_month = date.fromisoformat(row["date"])  # 计算或保存当前观测月份。
            raw_value = row["value"]  # 计算或保存接口返回的原始数值。
        except (KeyError, TypeError, ValueError) as exc:  # 捕获格式错误并保留明确原因。
            raise ValueError("FRED 月度观测缺少有效日期或数值") from exc  # 拒绝无效或不足的数据。
        if observed_month.day != 1 or observed_month > as_of:  # 校验当前条件并决定后续计算。
            raise ValueError("M2SL 观测日期不是已到达月份的第一天")  # 拒绝无效或不足的数据。
        if observed_month in values:  # 校验当前条件并决定后续计算。
            raise ValueError("M2SL 月度观测日期重复")  # 拒绝无效或不足的数据。
        if raw_value == ".":  # 校验当前条件并决定后续计算。
            continue  # 跳过该无效观测。
        if not isinstance(raw_value, str):  # 校验当前条件并决定后续计算。
            raise TypeError("M2SL 观测值不是字符串")  # 拒绝无效或不足的数据。
        try:  # 开始解析外部数据字段。
            value = Decimal(raw_value)  # 计算或保存已解析的有效数值。
        except InvalidOperation as exc:  # 捕获格式错误并保留明确原因。
            raise ValueError("M2SL 观测值格式无效") from exc  # 拒绝无效或不足的数据。
        if not value.is_finite() or value <= 0:  # 校验当前条件并决定后续计算。
            raise ValueError("M2SL 观测值必须是正的有限数值")  # 拒绝无效或不足的数据。
        values[observed_month] = value  # 计算或保存values[observed_month]。

    if not values:  # 校验当前条件并决定后续计算。
        return {"indicator": "M2_YoY", "status": "data_missing", "reason": "没有已发布的 M2SL 月值"}  # 返回本步骤的结果。
    latest_month = max(values)  # 计算或保存最新有效月份。
    previous_month = shift_month(latest_month, -1)  # 取得上月。
    latest_year_ago = shift_month(latest_month, -12)  # 取得去年同月。
    previous_year_ago = shift_month(latest_month, -13)  # 取得上月对应的去年月份。
    required = (latest_month, previous_month, latest_year_ago, previous_year_ago)  # 计算或保存计算所需日期。
    if any(month not in values for month in required):  # 校验当前条件并决定后续计算。
        return {  # 返回本步骤的结果。
            "indicator": "M2_YoY",  # 输出 indicator 字段供后续核对。
            "status": "data_missing",  # 输出 status 字段供后续核对。
            "reason": "缺少最新月、上月或其去年同月的 M2SL 月值",  # 输出 reason 字段供后续核对。
        }  # 结束当前数据结构。
    month_end = date(latest_month.year, latest_month.month, monthrange(latest_month.year, latest_month.month)[1])  # 计算或保存最新观测月的月末日期。
    if month_end > as_of or as_of - month_end > timedelta(days=75):  # 校验当前条件并决定后续计算。
        return {"indicator": "M2_YoY", "status": "data_missing", "reason": "最新 M2SL 观测月末超过 75 天有效期"}  # 返回本步骤的结果。

    latest_yoy = Decimal(100) * (values[latest_month] / values[latest_year_ago] - 1)  # 计算或保存最新月份同比增速。
    previous_yoy = Decimal(100) * (values[previous_month] / values[previous_year_ago] - 1)  # 计算或保存上月同比增速。
    yoy_change = latest_yoy - previous_yoy  # 计算或保存两期同比增速之差。
    return {  # 返回本步骤的结果。
        "indicator": "M2_YoY",  # 输出 indicator 字段供后续核对。
        "status": "ok",  # 输出 status 字段供后续核对。
        "source": _FRED_SOURCE,  # 输出 source 字段供后续核对。
        "series_id": "M2SL",  # 输出 series_id 字段供后续核对。
        "observed_at": latest_month.isoformat(),  # 输出 observed_at 字段供后续核对。
        "unit": "十亿美元，季节调整",  # 输出 unit 字段供后续核对。
        "latest_value": float(values[latest_month]),  # 输出 latest_value 字段供后续核对。
        "previous_value": float(values[previous_month]),  # 输出 previous_value 字段供后续核对。
        "latest_year_ago_value": float(values[latest_year_ago]),  # 输出 latest_year_ago_value 字段供后续核对。
        "previous_year_ago_value": float(values[previous_year_ago]),  # 输出 previous_year_ago_value 字段供后续核对。
        "latest_yoy_pct": float(latest_yoy),  # 输出 latest_yoy_pct 字段供后续核对。
        "previous_yoy_pct": float(previous_yoy),  # 输出 previous_yoy_pct 字段供后续核对。
        "yoy_change_pp": float(yoy_change),  # 输出 yoy_change_pp 字段供后续核对。
        "score": linear_score(yoy_change, -0.5, 0.5),  # 输出 score 字段供后续核对。
        "comparison_months": [month.isoformat() for month in required],  # 输出 comparison_months 字段供后续核对。
    }  # 结束当前数据结构。


def calculate_walcl_ma4(payload: dict[str, Any], as_of: date) -> dict[str, Any]:  # 定义该指标的程序计算函数。
    """用最近四个连续周三的 WALCL 原始值计算 MA4 与偏离百分比。"""
    rows = payload.get("observations")  # 计算或保存原始观测列表。
    if not isinstance(rows, list):  # 校验当前条件并决定后续计算。
        raise TypeError("FRED 响应缺少 observations 数组")  # 拒绝无效或不足的数据。
    values: dict[date, Decimal] = {}  # 计算或保存按日期整理的有效观测。
    for row in rows:  # 依次处理原始数据中的记录。
        if not isinstance(row, dict):  # 校验当前条件并决定后续计算。
            raise TypeError("WALCL 周度观测格式无效")  # 拒绝无效或不足的数据。
        try:  # 开始解析外部数据字段。
            observed_at = date.fromisoformat(row["date"])  # 计算或保存实际观测时间。
            raw_value = row["value"]  # 计算或保存接口返回的原始数值。
        except (KeyError, TypeError, ValueError) as exc:  # 捕获格式错误并保留明确原因。
            raise ValueError("WALCL 周度观测缺少有效日期或数值") from exc  # 拒绝无效或不足的数据。
        if observed_at.weekday() != 2 or observed_at > as_of:  # 校验当前条件并决定后续计算。
            raise ValueError("WALCL 观测日期不是已到达的周三")  # 拒绝无效或不足的数据。
        if observed_at in values:  # 校验当前条件并决定后续计算。
            raise ValueError("WALCL 周度观测日期重复")  # 拒绝无效或不足的数据。
        if raw_value == ".":  # 校验当前条件并决定后续计算。
            continue  # 跳过该无效观测。
        if not isinstance(raw_value, str):  # 校验当前条件并决定后续计算。
            raise TypeError("WALCL 观测值不是字符串")  # 拒绝无效或不足的数据。
        try:  # 开始解析外部数据字段。
            value = Decimal(raw_value)  # 计算或保存已解析的有效数值。
        except InvalidOperation as exc:  # 捕获格式错误并保留明确原因。
            raise ValueError("WALCL 观测值格式无效") from exc  # 拒绝无效或不足的数据。
        if not value.is_finite() or value <= 0:  # 校验当前条件并决定后续计算。
            raise ValueError("WALCL 观测值必须是正的有限数值")  # 拒绝无效或不足的数据。
        values[observed_at] = value  # 计算或保存values[observed_at]。

    if not values:  # 校验当前条件并决定后续计算。
        return {"indicator": "Fed_Balance（WALCL）", "status": "data_missing", "reason": "没有已发布的 WALCL 周值"}  # 返回本步骤的结果。
    latest_day = max(values)  # 计算或保存最新有效观测日期。
    required = [latest_day - timedelta(days=7 * index) for index in range(4)]  # 计算或保存计算所需日期。
    if any(day not in values for day in required):  # 校验当前条件并决定后续计算。
        return {"indicator": "Fed_Balance（WALCL）", "status": "data_missing", "reason": "缺少最近四个连续周三的 WALCL 观测"}  # 返回本步骤的结果。
    if as_of - latest_day > timedelta(days=14):  # 校验当前条件并决定后续计算。
        return {"indicator": "Fed_Balance（WALCL）", "status": "data_missing", "reason": "最新 WALCL 观测超过 14 天有效期"}  # 返回本步骤的结果。

    ma4 = sum((values[day] for day in required), Decimal(0)) / 4  # 计算或保存连续四周观测的平均值。
    latest = values[latest_day]  # 计算或保存最新指标原始值。
    ma4_change_pct = Decimal(100) * (latest / ma4 - 1)  # 计算或保存最新值相对四周均值的偏离百分比。
    return {  # 返回本步骤的结果。
        "indicator": "Fed_Balance（WALCL）",  # 输出 indicator 字段供后续核对。
        "status": "ok",  # 输出 status 字段供后续核对。
        "source": _FRED_SOURCE,  # 输出 source 字段供后续核对。
        "series_id": "WALCL",  # 输出 series_id 字段供后续核对。
        "observed_at": latest_day.isoformat(),  # 输出 observed_at 字段供后续核对。
        "unit": "百万美元，未季节调整",  # 输出 unit 字段供后续核对。
        "latest_value": float(latest),  # 输出 latest_value 字段供后续核对。
        "ma4_value": float(ma4),  # 输出 ma4_value 字段供后续核对。
        "ma4_change_pct": float(ma4_change_pct),  # 输出 ma4_change_pct 字段供后续核对。
        "score": linear_score(ma4_change_pct, -0.5, 0.5),  # 输出 score 字段供后续核对。
        "comparison_dates": [day.isoformat() for day in required],  # 输出 comparison_dates 字段供后续核对。
        "weekly_values": [float(values[day]) for day in required],  # 输出 weekly_values 字段供后续核对。
    }  # 结束当前数据结构。
