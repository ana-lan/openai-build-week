"""Pydantic request and response models for the RAG evaluation API."""

from __future__ import annotations

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
