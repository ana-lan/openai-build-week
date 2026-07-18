"""Generate good and deliberately flawed RAG triples for the demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from backend.mini_rag import GENERATION_MODEL, generate_rag_triple


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "example_triples.json"
SOURCE_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "source_text.txt"

GOOD_QUESTIONS = (
    "What role does self-attention play in the Transformer architecture?",
    "How does multi-head attention differ from using a single attention head?",
    "What pre-training tasks were used to train the original BERT model?",
)
IRRELEVANT_RETRIEVAL_QUESTION = (
    "Which techniques help stabilize training in the original Transformer model?"
)
FABRICATED_ANSWER_QUESTION = (
    "Why can BERT use context from both directions when representing a word?"
)
FABRICATED_CLAIM = (
    "The original BERT training curriculum doubled the input sequence length "
    "every 10,000 optimization steps."
)


def _bert_chunk() -> str:
    """Return a BERT-specific chunk for the retrieval-corruption example."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    marker = "===== BERT (language model) ====="
    try:
        bert_text = source.split(marker, maxsplit=1)[1]
    except IndexError as exc:
        raise RuntimeError(f"Could not find the BERT article header in {SOURCE_PATH}") from exc

    words = bert_text.split()
    if not words:
        raise RuntimeError("The BERT article section is empty")
    return " ".join(words[:200])


def _make_irrelevant_retrieval_example() -> dict[str, object]:
    """Replace one retrieved Transformer chunk with an unrelated BERT chunk."""
    triple = generate_rag_triple(IRRELEVANT_RETRIEVAL_QUESTION)
    triple["retrieved_chunks"][-1] = _bert_chunk()
    return triple


def _make_fabricated_answer_example(client: OpenAI) -> dict[str, object]:
    """Ask the model to introduce one known-unsupported claim into an answer."""
    triple = generate_rag_triple(FABRICATED_ANSWER_QUESTION)
    context = "\n\n".join(
        f"[Chunk {index}]\n{chunk}"
        for index, chunk in enumerate(triple["retrieved_chunks"], start=1)
    )
    response = client.responses.create(
        model=GENERATION_MODEL,
        instructions=(
            "Create a deliberately flawed answer for testing a RAG evaluator. "
            "Answer naturally using the supplied context, but include the exact "
            "fabricated sentence provided by the user. Do not label it as fabricated "
            "or mention that this is test data."
        ),
        input=(
            f"Question:\n{FABRICATED_ANSWER_QUESTION}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Fabricated sentence to include exactly:\n{FABRICATED_CLAIM}"
        ),
    )
    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("Generation API returned an empty fabricated answer")
    if FABRICATED_CLAIM not in answer:
        raise RuntimeError("Generated answer did not include the requested fabricated claim")

    triple["answer"] = answer
    return triple


def generate_samples() -> list[dict[str, object]]:
    """Generate three good triples followed by two intentionally bad triples."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set before generating samples")
    if not SOURCE_PATH.is_file() or SOURCE_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Source corpus not found at {SOURCE_PATH}. "
            "Run `python backend/fetch_source_data.py` first."
        )

    client = OpenAI()
    samples: list[dict[str, object]] = [
        generate_rag_triple(question) for question in GOOD_QUESTIONS
    ]
    samples.append(_make_irrelevant_retrieval_example())
    samples.append(_make_fabricated_answer_example(client))
    return samples


def main() -> None:
    """Generate and write the sample triples as formatted JSON."""
    samples = generate_samples()
    OUTPUT_PATH.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} sample triples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
