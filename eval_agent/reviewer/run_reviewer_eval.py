"""固定候选消息调用生产 reviewer，分别评分判断、核查依据和修改意见。"""

from typing import Any

from langfuse import Langfuse
from pydantic import model_validator

from app.crypto_analysis.crypto_analysis_parent_agent import ReportReview
from eval_agent.report_common import ReportCase, evaluate_case, run_evaluation


class ReviewerCase(ReportCase):
    expected_output: ReportReview

    @model_validator(mode="after")
    def validate_calibration(self) -> "ReviewerCase":
        if self.metadata.mode == "judge_calibration" and not isinstance(
            self.metadata.calibration_output, ReportReview
        ):
            raise ValueError("reviewer 裁判校准输出必须保留生产 ReportReview 结构")
        return self


async def evaluate(
    client: Langfuse, case: ReportCase, judge_model: str
) -> dict[str, Any]:
    return await evaluate_case(client, case, judge_model, "reviewer")


def main() -> None:
    run_evaluation("reviewer", ReviewerCase, evaluate, ReportReview.model_json_schema())


if __name__ == "__main__":
    main()
