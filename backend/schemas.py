"""Pydantic request and response models for the RAG evaluation API."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class RAGTriple(BaseModel):
    question: str = Field(min_length=1)
    retrieved_chunks: list[str] = Field(min_length=1)
    answer: str = Field(min_length=1)

    @field_validator("question", "answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("retrieved_chunks")
    @classmethod
    def validate_chunks(cls, chunks: list[str]) -> list[str]:
        if any(not chunk.strip() for chunk in chunks):
            raise ValueError("chunks must not be blank")
        return [chunk.strip() for chunk in chunks]


class Scores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)


class ScoreDiagnosisResponse(BaseModel):
    scores: Scores
    diagnosis: str | None


class EvaluatedTriple(BaseModel):
    triple: RAGTriple
    scores: Scores
    diagnosis: str | None


class FailureTypes(BaseModel):
    retrieval_context_precision: int = Field(ge=0)
    faithfulness: int = Field(ge=0)
    answer_relevance: int = Field(ge=0)


class ReportSummary(BaseModel):
    total_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    failure_types: FailureTypes
    average_scores: Scores


class ReportResponse(BaseModel):
    results: list[EvaluatedTriple]
    report: ReportSummary


class TuningConfiguration(BaseModel):
    chunk_size: int = Field(gt=0)
    top_k: int = Field(gt=0)
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)


class TuningRecommendation(BaseModel):
    configuration: TuningConfiguration
    combined_score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)


PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class TuneRequest(BaseModel):
    questions: list[str] = Field(min_length=1)
    chunk_sizes: list[PositiveInt] = Field(
        default_factory=lambda: [100, 150, 200, 300],
        min_length=1,
    )
    top_ks: list[PositiveInt] = Field(
        default_factory=lambda: [2, 3, 5],
        min_length=1,
    )

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
        if any(not question.strip() for question in questions):
            raise ValueError("questions must be non-empty strings")
        return [question.strip() for question in questions]


class TuneResponse(BaseModel):
    sweep_results: list[TuningConfiguration]
    recommendation: TuningRecommendation


class SaveRunRequest(BaseModel):
    report: ReportSummary
    label: str = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def validate_label(cls, label: str) -> str:
        if not label.strip():
            raise ValueError("label must not be blank")
        return label.strip()


class RunHistoryRecord(BaseModel):
    id: int = Field(gt=0)
    timestamp: str = Field(min_length=1)
    label: str | None
    report: ReportSummary


class RunComparison(BaseModel):
    label_a: str
    label_b: str
    average_scores_a: Scores
    average_scores_b: Scores
    difference_b_minus_a: dict[str, float]
