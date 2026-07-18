"""FastAPI entry point for scoring and diagnosing RAG outputs."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI
from pydantic import Field

from backend.diagnoser import diagnose_triple, generate_report
from backend.schemas import (
    EvaluatedTriple,
    RAGTriple,
    ReportResponse,
    ScoreDiagnosisResponse,
)
from backend.scorer import score_triple


app = FastAPI(
    title="RAG Eval Sidekick API",
    description="Score RAG outputs and explain likely pipeline failures.",
    version="0.1.0",
)


def _evaluate_triple(triple: RAGTriple) -> EvaluatedTriple:
    triple_data = triple.model_dump()
    scores = score_triple(triple_data)
    diagnosis = diagnose_triple(triple_data, scores)
    return EvaluatedTriple(triple=triple, scores=scores, diagnosis=diagnosis)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score-and-diagnose", response_model=ScoreDiagnosisResponse)
def score_and_diagnose(triple: RAGTriple) -> ScoreDiagnosisResponse:
    result = _evaluate_triple(triple)
    return ScoreDiagnosisResponse(scores=result.scores, diagnosis=result.diagnosis)


@app.post("/report", response_model=ReportResponse)
def report(
    triples: Annotated[list[RAGTriple], Field(min_length=1)],
) -> ReportResponse:
    results = [_evaluate_triple(triple) for triple in triples]
    summary = generate_report([result.model_dump() for result in results])
    return ReportResponse(results=results, report=summary)
