"""指标分析工具的数据源异常分类。

所有数据源异常统一继承 ``DataSourceError``，调用方按异常类型区分数据缺失、
请求不可用和配置缺失，无需针对 FRED、Binance 等数据源分别编写异常处理分支。

    DataSourceError
    ├── DataSourceValidationError     参数或响应字段取值无效
    ├── DataSourceResponseError       响应结构或 JSON 格式无效
    ├── DataMissingError              数据为空、过期等导致没有可用数据
    ├── DataSourceUnavailableError    数据源限流或请求失败
    └── DataSourceNotConfiguredError  缺少 API 密钥或必要配置

异常类型按处理方式划分。空数据和过期数据均使用 ``DataMissingError``，
具体原因通过 ``detail`` 字段说明。
"""

from __future__ import annotations


class DataSourceError(Exception):
    """指标数据源无法提供可用数据时的异常基类。"""


class DataSourceValidationError(DataSourceError, ValueError):
    """数据源参数或响应字段取值无效，同时兼容已有的 ValueError 处理。"""


class DataSourceResponseError(DataSourceError, TypeError):
    """数据源响应结构或 JSON 格式无效，同时兼容已有的 TypeError 处理。"""


class DataMissingError(DataSourceError):
    """数据源未返回指定标的的可用数据，包括空结果和过期数据。

    保留用户请求的标的代码、实际查询使用的标准代码以及具体缺失原因，
    供调用方生成明确的数据缺失记录。
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class DataSourceUnavailableError(DataSourceError):
    """数据源因限流或请求失败而不可用。

    此异常仅表示请求未成功，不能据此判断目标标的是否存在可用数据。
    """


class DataSourceNotConfiguredError(DataSourceError, ValueError):
    """数据源缺少 API 密钥或必要配置。

    同时继承 ValueError，保留调用方通过 ValueError 捕获配置异常的行为。
    """
