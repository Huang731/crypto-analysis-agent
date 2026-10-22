"""指标数据源共用的 HTTP 请求与连通性检查工具。"""

from typing import Any

import requests


def get_with_redacted_errors(
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    secret: str,
    allowed_status_codes: tuple[int, ...] = (),
) -> requests.Response:
    """发送 GET 请求并检查响应状态，对请求异常中的敏感值进行脱敏。

    通过查询参数认证的数据源会将密钥放入 URL。requests 的 HTTP、连接和超时异常
    可能包含完整 URL，直接记录日志或异常堆栈会泄露密钥（#1324）。提供非空 secret 时，
    使用相同的异常类型重新构造异常，并替换错误消息中的敏感值；新异常不附带包含 URL 的
    请求、响应对象或原始异常链，因此在 except 代码块外抛出。未提供敏感值时保留原异常。
    allowed_status_codes 中的状态码不触发状态检查，由调用方处理对应响应。
    """
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code not in allowed_status_codes:
            response.raise_for_status()
        return response
    except requests.RequestException as exc:
        error = type(exc)(str(exc).replace(secret, "***")) if secret else exc
    raise error


def is_data_source_reachable(url: str, timeout: float = 5.0) -> bool:
    """通过 HEAD 请求检查数据源是否可达，用于区分空结果与连接故障。

    某些客户端在请求失败时返回空结果而不抛出异常，因此仅在结果为空时调用此检查。
    收到 HTTP 响应即视为可达，不依据响应状态码判断业务接口是否可用。
    """
    try:
        requests.head(url, timeout=timeout, allow_redirects=True)
        return True
    except requests.RequestException:
        return False
