"""Persist evaluation report history in a local SQLite database."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


DATABASE_PATH = Path(__file__).resolve().parents[1] / "runs.db"
SCORE_NAMES = ("faithfulness", "answer_relevance", "context_precision")


class RunRecord(TypedDict):
    id: int
    timestamp: str
    label: str | None
    report: dict[str, object]


class RunComparison(TypedDict):
    label_a: str
    label_b: str
    average_scores_a: dict[str, float]
    average_scores_b: dict[str, float]
    difference_b_minus_a: dict[str, float]


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            label TEXT UNIQUE,
            report_json TEXT NOT NULL,
            faithfulness REAL NOT NULL,
            answer_relevance REAL NOT NULL,
            context_precision REAL NOT NULL
        )
        """
    )
    return connection


def _validate_report(report: Mapping[str, object]) -> tuple[dict[str, object], dict[str, float]]:
    """Validate the report shape needed for storage and comparison."""
    raw_averages = report.get("average_scores")
    if not isinstance(raw_averages, Mapping):
        raise ValueError("report.average_scores must be an object")

    averages: dict[str, float] = {}
    for metric in SCORE_NAMES:
        value = raw_averages.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"report.average_scores.{metric} must be from 0 to 1")
        numeric_value = float(value)
        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f"report.average_scores.{metric} must be from 0 to 1")
        averages[metric] = numeric_value

    required_counts = ("total_count", "passed_count", "failed_count")
    for name in required_counts:
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"report.{name} must be a non-negative integer")

    failure_types = report.get("failure_types")
    if not isinstance(failure_types, Mapping):
        raise ValueError("report.failure_types must be an object")

    try:
        normalized_report = json.loads(json.dumps(report))
    except (TypeError, ValueError) as exc:
        raise ValueError("report must be JSON serializable") from exc
    return normalized_report, averages


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not isinstance(label, str):
        raise ValueError("label must be a string or None")
    normalized = label.strip()
    if not normalized:
        raise ValueError("label must not be blank")
    return normalized


def save_run(report: Mapping[str, object], label: str | None = None) -> RunRecord:
    """Save one aggregate report and return its history record."""
    normalized_report, averages = _validate_report(report)
    normalized_label = _normalize_label(label)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO runs (
                    timestamp, label, report_json,
                    faithfulness, answer_relevance, context_precision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    normalized_label,
                    json.dumps(normalized_report, separators=(",", ":")),
                    averages["faithfulness"],
                    averages["answer_relevance"],
                    averages["context_precision"],
                ),
            )
            run_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A run with label {normalized_label!r} already exists") from exc

    return {
        "id": run_id,
        "timestamp": timestamp,
        "label": normalized_label,
        "report": normalized_report,
    }


def get_run_history() -> list[RunRecord]:
    """Return all saved runs, newest first."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, timestamp, label, report_json FROM runs ORDER BY id DESC"
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "timestamp": str(row["timestamp"]),
            "label": row["label"],
            "report": json.loads(row["report_json"]),
        }
        for row in rows
    ]


def compare_runs(label_a: str, label_b: str) -> RunComparison:
    """Compare labeled runs, returning average-score deltas as B minus A."""
    normalized_a = _normalize_label(label_a)
    normalized_b = _normalize_label(label_b)
    assert normalized_a is not None and normalized_b is not None

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT label, faithfulness, answer_relevance, context_precision
            FROM runs WHERE label IN (?, ?)
            """,
            (normalized_a, normalized_b),
        ).fetchall()

    by_label = {str(row["label"]): row for row in rows}
    missing = [label for label in (normalized_a, normalized_b) if label not in by_label]
    if missing:
        raise ValueError(f"No saved run found for label(s): {', '.join(missing)}")

    scores_a = {metric: float(by_label[normalized_a][metric]) for metric in SCORE_NAMES}
    scores_b = {metric: float(by_label[normalized_b][metric]) for metric in SCORE_NAMES}
    return {
        "label_a": normalized_a,
        "label_b": normalized_b,
        "average_scores_a": scores_a,
        "average_scores_b": scores_b,
        "difference_b_minus_a": {
            metric: scores_b[metric] - scores_a[metric] for metric in SCORE_NAMES
        },
    }
