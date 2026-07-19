"""Sweep retrieval settings and compare aggregate RAG evaluation scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
from openai import OpenAI

from backend.mini_rag import (
    EMBEDDING_MODEL,
    _chunk_text,
    _cosine_similarities,
    _generate_answer,
    _read_source_text,
)
from backend.scorer import score_triple


class ConfigurationResult(TypedDict):
    """Average scores for one retrieval configuration."""

    chunk_size: int
    top_k: int
    faithfulness: float
    answer_relevance: float
    context_precision: float


class ConfigurationRecommendation(TypedDict):
    """Best sweep result plus its equal-weight score and explanation."""

    configuration: ConfigurationResult
    combined_score: float
    explanation: str


def _validate_positive_values(name: str, values: Sequence[int]) -> list[int]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive integers")
    return list(dict.fromkeys(values))


def _embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding API returned an unexpected number of vectors")
    return embeddings


def _sweep_configurations(
    questions: Sequence[str],
    chunk_sizes: Sequence[int],
    top_ks: Sequence[int],
    client: OpenAI,
) -> list[ConfigurationResult]:
    """Internal sweep implementation with an injectable OpenAI client."""
    normalized_questions = []
    for question in questions:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("questions must contain only non-empty strings")
        normalized_questions.append(question.strip())
    if not normalized_questions:
        raise ValueError("questions must not be empty")

    normalized_chunk_sizes = _validate_positive_values("chunk_sizes", chunk_sizes)
    normalized_top_ks = _validate_positive_values("top_ks", top_ks)
    source_text = _read_source_text()

    # Question vectors do not depend on chunking, so one batch serves the full sweep.
    question_embeddings = _embed(client, normalized_questions)
    results: list[ConfigurationResult] = []

    for chunk_size in normalized_chunk_sizes:
        chunks = _chunk_text(source_text, chunk_size=chunk_size)
        chunk_embeddings = _embed(client, chunks)
        similarities_by_question = [
            _cosine_similarities(question_embedding, chunk_embeddings)
            for question_embedding in question_embeddings
        ]

        for top_k in normalized_top_ks:
            totals = {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
            }
            result_count = min(top_k, len(chunks))

            for question, similarities in zip(
                normalized_questions, similarities_by_question, strict=True
            ):
                ranked_indices = np.argsort(-similarities, kind="stable")[:result_count]
                retrieved_chunks = [chunks[int(index)] for index in ranked_indices]
                answer = _generate_answer(question, retrieved_chunks, client)
                triple = {
                    "question": question,
                    "retrieved_chunks": retrieved_chunks,
                    "answer": answer,
                }
                cached_context_precision = float(
                    np.clip(similarities[ranked_indices], 0.0, 1.0).mean()
                )
                scores = score_triple(
                    triple,
                    client=client,
                    context_precision_override=cached_context_precision,
                )
                for metric in totals:
                    totals[metric] += scores[metric]

            question_count = len(normalized_questions)
            results.append(
                {
                    "chunk_size": chunk_size,
                    "top_k": top_k,
                    "faithfulness": totals["faithfulness"] / question_count,
                    "answer_relevance": totals["answer_relevance"] / question_count,
                    "context_precision": totals["context_precision"] / question_count,
                }
            )

    return results


def sweep_configurations(
    questions: Sequence[str],
    chunk_sizes: Sequence[int] = (100, 150, 200, 300),
    top_ks: Sequence[int] = (2, 3, 5),
) -> list[ConfigurationResult]:
    """Evaluate every chunk-size/top-k combination across the given questions.

    Chunk embeddings are computed once per chunk size, and all question embeddings
    are computed once for the entire sweep. The function still performs generation
    and both LLM-judge evaluations for every question/configuration pair.
    """
    return _sweep_configurations(questions, chunk_sizes, top_ks, OpenAI())


def recommend_best_configuration(
    sweep_results: Sequence[Mapping[str, object]],
) -> ConfigurationRecommendation:
    """Recommend the configuration with the highest equal-weight metric mean."""
    if not sweep_results:
        raise ValueError("sweep_results must not be empty")

    validated: list[tuple[ConfigurationResult, float]] = []
    metric_names = ("faithfulness", "answer_relevance", "context_precision")

    for index, raw_result in enumerate(sweep_results):
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"sweep result {index} must be an object")

        chunk_size = raw_result.get("chunk_size")
        top_k = raw_result.get("top_k")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError(f"sweep result {index}.chunk_size must be positive")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"sweep result {index}.top_k must be positive")

        scores: dict[str, float] = {}
        for metric in metric_names:
            value = raw_result.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"sweep result {index}.{metric} must be from 0 to 1")
            numeric_value = float(value)
            if not 0.0 <= numeric_value <= 1.0:
                raise ValueError(f"sweep result {index}.{metric} must be from 0 to 1")
            scores[metric] = numeric_value

        configuration: ConfigurationResult = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevance"],
            "context_precision": scores["context_precision"],
        }
        combined_score = sum(scores.values()) / len(metric_names)
        validated.append((configuration, combined_score))

    best_configuration, best_score = max(validated, key=lambda item: item[1])
    worst_configuration, worst_score = min(validated, key=lambda item: item[1])
    improvement = best_score - worst_score

    if improvement == 0.0:
        comparison = "All tested configurations have the same combined score."
    else:
        comparison = (
            f"It is {improvement:.3f} points ({improvement * 100:.1f} percentage "
            f"points) better than the worst configuration, which used "
            f"chunk_size={worst_configuration['chunk_size']} and "
            f"top_k={worst_configuration['top_k']}."
        )

    explanation = (
        f"chunk_size={best_configuration['chunk_size']} and "
        f"top_k={best_configuration['top_k']} is the best configuration because "
        f"it has the highest equal-weight combined score ({best_score:.3f}). "
        f"Its averages are faithfulness {best_configuration['faithfulness']:.3f}, "
        f"answer relevance {best_configuration['answer_relevance']:.3f}, and "
        f"context precision {best_configuration['context_precision']:.3f}. "
        f"{comparison}"
    )
    return {
        "configuration": best_configuration,
        "combined_score": best_score,
        "explanation": explanation,
    }
