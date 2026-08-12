from __future__ import annotations

from pydantic import BaseModel, Field


class MetricEvaluation(BaseModel):
    """Score and feedback for one response-quality metric."""

    feedback: str = Field(
        min_length=1,
        description="Detailed feedback for the evaluated metric",
    )

    score: int = Field(
        ge=1,
        le=5,
        description=(
            "The score of the evaluated metric, between 1 (very poor) "
            "and 5 (excellent)"
        ),
    )


class ResponseEvaluation(BaseModel):
    """Structured evaluation of an Energy Advisor response."""

    accuracy: MetricEvaluation
    relevance: MetricEvaluation
    completeness: MetricEvaluation
    usefulness: MetricEvaluation
    strengths: list[str]
    weaknesses: list[str]
    overall_feedback: str = Field(
        min_length=1,
        description="Overall assessment and the most important improvement",
    )
