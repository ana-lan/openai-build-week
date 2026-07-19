"""FastAPI entry point for scoring and diagnosing RAG outputs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import Field

from backend.diagnoser import diagnose_triple, generate_report
from backend.history import compare_runs, get_run_history, save_run
from backend.mini_rag import generate_rag_triple
from backend.schemas import (
    EvaluatedTriple,
    RAGTriple,
    ReportResponse,
    RunComparison,
    RunHistoryRecord,
    SaveRunRequest,
    ScoreDiagnosisResponse,
    TuneRequest,
    TuneResponse,
)
from backend.scorer import score_triple
from backend.tuner import recommend_best_configuration, sweep_configurations


MAX_SOURCE_BYTES = 5 * 1024 * 1024

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


@app.post("/generate-triple", response_model=RAGTriple)
def generate_triple(
    source_file: Annotated[UploadFile, File(description="A UTF-8 .txt document")],
    question: Annotated[str, Form(min_length=1)],
) -> RAGTriple:
    """Generate a fresh RAG triple from an uploaded source document."""
    filename = source_file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Source file must have a .txt extension")

    content = source_file.file.read(MAX_SOURCE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Source file is empty")
    if len(content) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Source file must be 5 MB or smaller")

    try:
        with TemporaryDirectory(prefix="rag-eval-source-") as directory:
            source_path = Path(directory) / "source.txt"
            source_path.write_bytes(content)
            triple = generate_rag_triple(question, source_path=source_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RAGTriple.model_validate(triple)


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


@app.post("/tune", response_model=TuneResponse)
def tune(request: TuneRequest) -> TuneResponse:
    sweep_results = sweep_configurations(
        request.questions,
        chunk_sizes=request.chunk_sizes,
        top_ks=request.top_ks,
    )
    recommendation = recommend_best_configuration(sweep_results)
    return TuneResponse(
        sweep_results=sweep_results,
        recommendation=recommendation,
    )


@app.post("/save-run", response_model=RunHistoryRecord)
def save_evaluation_run(request: SaveRunRequest) -> RunHistoryRecord:
    try:
        record = save_run(request.report.model_dump(), request.label)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RunHistoryRecord.model_validate(record)


@app.get("/runs", response_model=list[RunHistoryRecord])
def list_runs() -> list[RunHistoryRecord]:
    return [RunHistoryRecord.model_validate(run) for run in get_run_history()]


@app.get("/compare-runs", response_model=RunComparison)
def compare_evaluation_runs(
    label_a: Annotated[str, Query(min_length=1)],
    label_b: Annotated[str, Query(min_length=1)],
) -> RunComparison:
    try:
        comparison = compare_runs(label_a, label_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RunComparison.model_validate(comparison)
