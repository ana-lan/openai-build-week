# RAG Eval Sidekick

RAG Eval Sidekick evaluates outputs from retrieval-augmented generation (RAG)
pipelines without requiring a person to inspect every answer manually. Give it one
or more triples of:

```text
{question, retrieved_chunks, answer}
```

It scores each triple for **faithfulness**, **answer relevance**, and **context
precision**, then explains likely root causes such as irrelevant retrieval or an
unsupported claim in the generated answer. It also includes retrieval
configuration tuning and SQLite-backed regression history.

The product has a Streamlit interface at <http://localhost:8501> and a FastAPI API
at <http://localhost:8000>.

## Run it with Docker Compose (recommended)

Prerequisites: Docker Desktop, or Docker Engine with Docker Compose.

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your OpenAI API key:

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Build and start the backend and frontend:

```bash
docker compose up --build
```

Older installations may use `docker-compose up --build` instead. When both
services are ready, open:

- Streamlit application: <http://localhost:8501>
- FastAPI documentation: <http://localhost:8000/docs>
- FastAPI health check: <http://localhost:8000/health>

Stop the services with `Ctrl+C`, then run:

```bash
docker compose down
```

Run history is stored in the named Docker volume `rag_eval_history`, so normal
container recreation does not erase saved runs.

## Try it immediately

The repository includes five prepared examples in
[`sample_data/example_triples.json`](sample_data/example_triples.json):

- Three normal RAG examples about Transformers, attention, and BERT.
- One deliberately poor retrieval example containing an irrelevant chunk.
- One deliberately unfaithful answer containing a plausible but unsupported claim.

After starting the application:

1. Open the **🔎 Evaluate** tab.
2. Click **Load example triples**.
3. Click **Run evaluation**.
4. Review the aggregate pass/fail summary, per-metric scores, diagnoses, and
   retrieved chunks.

The examples use the included Wikipedia-derived corpus at
[`sample_data/source_text.txt`](sample_data/source_text.txt).

## Two ways to use Sidekick

The two input paths serve different purposes and should not be confused.

### Evaluate outputs from your own RAG pipeline — the primary use case

Use **Custom triple**, `POST /score-and-diagnose`, or `POST /report` when you
already have RAG outputs. Sidekick does not require your application to use a
particular vector database, embedding model, framework, chunking strategy, or
generation model. Export each real result as:

```json
{
  "question": "What did the policy change?",
  "retrieved_chunks": ["First retrieved passage", "Second retrieved passage"],
  "answer": "The answer produced by your RAG system"
}
```

This is the actual product workflow: Sidekick evaluates the behavior of an
existing RAG system independently of how that system was built.

### Generate synthetic triples from a source document — demo and onboarding

The **Source document** uploader accepts a UTF-8 `.txt` file and a question. It
runs Sidekick's small built-in RAG pipeline to chunk the document, embed and
retrieve passages, and generate an answer before evaluating the result. This path
is useful for demos, onboarding, and producing test data when no external RAG
pipeline is available. It is not required to evaluate your own system.

## Main features

### 1. Evaluation, scoring, and diagnosis

For every RAG triple, Sidekick returns three scores from `0.0` to `1.0`:

- **Faithfulness:** whether factual claims in the answer are supported by the
  retrieved chunks. GPT-5.6 applies an explicit rubric; a specific unsupported
  factual or numerical claim caps the score even if the rest of the answer is
  well-supported.
- **Answer relevance:** whether the answer directly and completely addresses the
  question, judged with a separate GPT-5.6 rubric.
- **Context precision:** the average embedding cosine similarity between the
  question and retrieved chunks, used as a practical estimate of retrieval
  relevance.

A score below `0.6` fails. Failed triples receive a short diagnosis distinguishing
retrieval problems, unsupported generation, and off-topic or incomplete answers.
Batch reports include pass/fail counts, metric averages, and non-exclusive failure
type counts.

Relevant endpoints:

- `POST /score-and-diagnose`
- `POST /report`
- `POST /generate-triple` for the optional built-in demo pipeline

### 2. Auto-tuning

The **⚙️ Auto-tune** tab evaluates combinations of chunk size and retrieval count
(`top_k`) across a set of questions. Defaults are:

```text
chunk_sizes = [100, 150, 200, 300]
top_ks      = [2, 3, 5]
```

Smaller custom lists can be supplied for faster tests. The tuner averages all
three evaluation metrics with equal weight, highlights the best configuration,
and explains how much better its combined score is than the worst configuration.
It batches all question embeddings once and reuses chunk embeddings across
`top_k` values to avoid redundant embedding calls.

Relevant endpoint: `POST /tune`.

### 3. Run history and regression comparison

The **🕘 Run History** tab saves aggregate reports with unique labels such as
`v1-baseline` or `v2-smaller-chunks`. Saved runs include a UTC timestamp and the
full report. Two labeled runs can be compared side by side; score differences are
reported as **run B minus run A**, so positive values indicate improvement in the
second run.

History is stored locally in SQLite (`runs.db` outside Docker, or the persistent
Compose volume inside Docker).

Relevant endpoints:

- `POST /save-run`
- `GET /runs`
- `GET /compare-runs?label_a=...&label_b=...`

## Manual setup (fallback)

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

In manual mode, the application does not automatically load `.env`. Export the
key into both service terminals or their shared parent shell:

```bash
export OPENAI_API_KEY="your-key"
```

Start the backend:

```bash
uvicorn backend.app:app --reload
```

In a second terminal with the virtual environment active, start Streamlit:

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

Open <http://localhost:8501>. The API documentation is at
<http://localhost:8000/docs>.

## How Codex was used

This project was built iteratively with Codex as an implementation and testing
partner, rather than generated in one pass.

Codex accelerated:

- Scaffolding the FastAPI, Streamlit, sample-data, Docker, and Compose structure.
- Implementing the Wikipedia API downloader and the self-contained mini RAG
  pipeline, including custom document support.
- Turning the evaluation criteria into explicit, inspectable LLM-judge prompts and
  constrained structured outputs.
- Wiring scoring, diagnosis, aggregate reporting, multipart upload, tuning, SQLite
  history, and the tabbed frontend through typed API contracts.
- Optimizing the tuner to batch question embeddings and reuse chunk embeddings.
- Running offline fake-client tests and endpoint integration tests throughout. This
  caught issues during development such as missing dependencies, multipart request
  validation, API request-shape mismatches, and a floating-point equality mistake
  in a recommendation test. Codex also repeatedly boot-tested the Streamlit app and
  validated the Docker Compose configuration.

The product and evaluation decisions remained human-directed:

- Selecting faithfulness, answer relevance, and context precision as the core
  metrics.
- Defining the `0.6` health threshold and deciding that diagnoses should separate
  retrieval failures from generation failures.
- Reviewing the judge's behavior and identifying that the first faithfulness rubric
  was too lenient toward a confident fabricated statistic. The rubric was then
  tightened so unsupported falsifiable claims cannot receive the same score as a
  harmless nuance.
- Choosing the chunk-size and `top_k` sweep ranges.
- Expanding the initial evaluator into an auto-tuner and labeled regression-history
  tool.
- Keeping the built-in mini RAG path explicitly separate from the primary use case
  of evaluating outputs from any external RAG pipeline.

## License and source attribution

The project code is released under the MIT License. See [`LICENSE`](LICENSE).

The included source corpus contains text from the following English Wikipedia
articles:

- [Transformer (deep learning architecture)](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))
- [Attention (machine learning)](https://en.wikipedia.org/wiki/Attention_(machine_learning))
- [BERT (language model)](https://en.wikipedia.org/wiki/BERT_(language_model))

That source material is used under the
[Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/).
The attribution is also included at the end of
[`sample_data/source_text.txt`](sample_data/source_text.txt). The Wikipedia-derived
content remains subject to CC BY-SA 4.0; the project's MIT license applies to the
project code and does not replace the source material's license.
