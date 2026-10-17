"""固定候选消息调用生产 writer，按人工参考报告和检查项评分。"""

from typing import Any

from langfuse import Langfuse
from pydantic import Field, RootModel, model_validator

from eval_agent.report_common import (
    ReportCase,
    evaluate_case,
    run_evaluation,
)


class WriterExpected(RootModel[str]):
    root: str = Field(min_length=1)


class WriterCase(ReportCase):
    expected_output: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_calibration(self) -> "WriterCase":
        if self.metadata.mode == "judge_calibration" and not isinstance(
            self.metadata.calibration_output, str
        ):
            raise ValueError("writer 裁判校准输出必须是报告字符串")
        return self


async def evaluate(
    client: Langfuse, case: ReportCase, judge_model: str
) -> dict[str, Any]:
    return await evaluate_case(client, case, judge_model, "writer")


def main() -> None:
    run_evaluation("writer", WriterCase, evaluate, WriterExpected.model_json_schema())


if __name__ == "__main__":
    main()
