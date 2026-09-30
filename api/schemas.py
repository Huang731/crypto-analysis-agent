"""只接收用户问题的流式问答请求。"""

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """前端只需提交问题；路由、目标币种和截止时间由后端确定。"""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
