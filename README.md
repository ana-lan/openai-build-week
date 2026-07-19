# RAG Eval Sidekick

RAG Eval Sidekick evaluates retrieval-augmented generation outputs for
faithfulness, answer relevance, and context precision. It also diagnoses likely
retrieval or generation failures, compares retrieval configurations, and tracks
evaluation history.

## Recommended setup: Docker Compose

Prerequisites: Docker Desktop or Docker Engine with Docker Compose.

1. Create the local environment file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and replace the placeholder with your OpenAI API key.

3. Build and start both services:

   ```bash
   docker compose up --build
   ```

   If your Docker installation uses the standalone Compose command, run
   `docker-compose up --build` instead.

4. Open the Streamlit interface at <http://localhost:8501>.

The FastAPI backend is available at <http://localhost:8000>, with interactive API
documentation at <http://localhost:8000/docs>. Run history is stored in the named
Docker volume `rag_eval_history`, so it survives container recreation.

Stop the services with `Ctrl+C`, followed by:

```bash
docker compose down
```

## Manual setup fallback

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the API key in your shell. The Python services read `OPENAI_API_KEY` from the
environment; they do not load `.env` automatically in manual mode.

```bash
export OPENAI_API_KEY="your-key"
```

Start the backend:

```bash
uvicorn backend.app:app --reload
```

In a second terminal with the virtual environment active, start the frontend:

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

Then open <http://localhost:8501>.

## Sample data

The repository includes the default source corpus and example RAG triples under
`sample_data/`. You can also upload a UTF-8 `.txt` document through the Streamlit
interface and generate a fresh triple from it.
