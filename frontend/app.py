"""Simple Streamlit interface for RAG Eval Sidekick."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st


SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "example_triples.json"
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HEALTHY_THRESHOLD = 0.6
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_precision": "Context precision",
}


def _validate_triples(data: object) -> list[dict[str, object]]:
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list of triples.")
    required = {"question", "retrieved_chunks", "answer"}
    for index, triple in enumerate(data, start=1):
        if not isinstance(triple, dict) or not required.issubset(triple):
            raise ValueError(
                f"Triple {index} must contain question, retrieved_chunks, and answer."
            )
    return data


def _request_report(
    backend_url: str, triples: list[dict[str, object]]
) -> dict[str, object]:
    response = requests.post(
        f"{backend_url}/report",
        json=triples,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _request_run_history(backend_url: str) -> list[dict[str, object]]:
    response = requests.get(f"{backend_url}/runs", timeout=30)
    response.raise_for_status()
    return response.json()


def _parse_optional_int_list(value: str, label: str) -> list[int] | None:
    """Parse comma- or whitespace-separated positive integers, or return None."""
    if not value.strip():
        return None
    try:
        values = [int(item) for item in value.replace(",", " ").split()]
    except ValueError as exc:
        raise ValueError(f"{label} must contain only whole numbers.") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label} must contain positive whole numbers.")
    return values


def _score_badge(label: str, score: float) -> str:
    color = "#15803d" if score >= HEALTHY_THRESHOLD else "#b91c1c"
    background = "#dcfce7" if score >= HEALTHY_THRESHOLD else "#fee2e2"
    return (
        f'<span style="display:inline-block;padding:0.3rem 0.55rem;margin:0 0.35rem '
        f'0.35rem 0;border-radius:0.4rem;background:{background};color:{color};'
        f'font-weight:600">{label}: {score:.2f}</span>'
    )


def _show_report(payload: dict[str, object]) -> None:
    report = payload["report"]
    st.divider()
    st.header("Evaluation summary")

    count_columns = st.columns(3)
    count_columns[0].metric("Total", report["total_count"])
    count_columns[1].metric("Passed", report["passed_count"])
    count_columns[2].metric("Failed", report["failed_count"])

    st.subheader("Average scores")
    badges = "".join(
        _score_badge(METRIC_LABELS[name], float(report["average_scores"][name]))
        for name in METRIC_LABELS
    )
    st.markdown(badges, unsafe_allow_html=True)

    st.subheader("Failure type breakdown")
    failure_types = report["failure_types"]
    failure_columns = st.columns(3)
    failure_columns[0].metric(
        "Retrieval / context precision",
        failure_types["retrieval_context_precision"],
    )
    failure_columns[1].metric("Faithfulness", failure_types["faithfulness"])
    failure_columns[2].metric(
        "Answer relevance", failure_types["answer_relevance"]
    )

    st.header("Triple results")
    for index, result in enumerate(payload["results"], start=1):
        triple = result["triple"]
        scores = result["scores"]
        passed = all(float(value) >= HEALTHY_THRESHOLD for value in scores.values())
        status = "Passed" if passed else "Failed"
        with st.expander(f"{index}. {status} — {triple['question']}", expanded=not passed):
            score_badges = "".join(
                _score_badge(METRIC_LABELS[name], float(scores[name]))
                for name in METRIC_LABELS
            )
            st.markdown(score_badges, unsafe_allow_html=True)
            if result.get("diagnosis"):
                st.error(result["diagnosis"])

            st.markdown("**Answer**")
            st.write(triple["answer"])
            with st.expander("Retrieved chunks"):
                for chunk_index, chunk in enumerate(
                    triple["retrieved_chunks"], start=1
                ):
                    st.markdown(f"**Chunk {chunk_index}**")
                    st.write(chunk)


def _render_evaluate_tab(backend_url: str) -> None:
    st.subheader("Evaluate RAG outputs")
    st.caption("Load examples, paste a triple, or generate one from your own document.")

    left, middle, right = st.columns(3, gap="large")
    with left:
        with st.container(border=True, height="stretch"):
            st.subheader("📁 Sample data")
            st.write("Load the five prepared good and deliberately flawed examples.")
            if st.button("Load example triples", use_container_width=True):
                try:
                    st.session_state.triples = _validate_triples(
                        json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
                    )
                    st.session_state.evaluation = None
                    st.success(
                        f"Loaded {len(st.session_state.triples)} example triples."
                    )
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    st.error(f"Could not load sample data: {exc}")
            st.container(height=16, border=False)

    with middle:
        with st.container(border=True, height="stretch"):
            st.subheader("✏️ Custom triple")
            custom_json = st.text_area(
                "Paste one triple as JSON",
                value=json.dumps(
                    {"question": "", "retrieved_chunks": [""], "answer": ""},
                    indent=2,
                ),
                height=180,
            )
            if st.button("Use custom triple", use_container_width=True):
                try:
                    custom_triple = json.loads(custom_json)
                    st.session_state.triples = _validate_triples([custom_triple])
                    st.session_state.evaluation = None
                    st.success("Custom triple is ready.")
                except (json.JSONDecodeError, ValueError) as exc:
                    st.error(f"Invalid triple: {exc}")
            st.container(height=16, border=False)

    with right:
        with st.container(border=True, height="stretch"):
            st.subheader("📄 Source document")
            upload_column, suggest_column = st.columns([3, 2])
            source_file = upload_column.file_uploader(
                "Upload a UTF-8 .txt file", type=["txt"], key="source_file"
            )
            suggest_column.write("")
            suggest_clicked = suggest_column.button(
                "Suggest questions",
                disabled=source_file is None,
                use_container_width=True,
            )

            if suggest_clicked and source_file is not None:
                try:
                    source_text = source_file.getvalue().decode("utf-8")
                    if not source_text.strip():
                        raise ValueError("The uploaded source file is empty.")
                    with st.spinner("Reading document and suggesting questions..."):
                        suggestion_response = requests.post(
                            f"{backend_url}/suggest-questions",
                            json={"source_text": source_text},
                            timeout=120,
                        )
                        suggestion_response.raise_for_status()
                        suggestions = suggestion_response.json()
                    if not isinstance(suggestions, list) or len(suggestions) != 4:
                        raise ValueError("Backend returned an invalid suggestion list.")
                    st.session_state.suggested_questions = suggestions
                except UnicodeDecodeError:
                    st.error("The uploaded source file must be valid UTF-8 text.")
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"Could not suggest questions: {exc}")

            if st.session_state.suggested_questions:
                st.divider()
                st.subheader("Suggested questions")
                st.caption("Choose one to fill the question field below.")
                for index, suggested_question in enumerate(
                    st.session_state.suggested_questions
                ):
                    display_question = (
                        suggested_question
                        if len(suggested_question) <= 72
                        else f"{suggested_question[:69].rstrip()}..."
                    )
                    if st.button(
                        display_question,
                        key=f"suggested_question_{index}",
                        help=suggested_question,
                        use_container_width=True,
                    ):
                        st.session_state.source_question = suggested_question

            with st.form("source_document_form"):
                source_question = st.text_input(
                    "Question to ask", key="source_question"
                )
                generate_submitted = st.form_submit_button(
                    "Generate and evaluate", use_container_width=True
                )

            if generate_submitted:
                if source_file is None or not source_question.strip():
                    st.error("Choose a .txt file and enter a question.")
                else:
                    try:
                        with st.spinner("Generating, scoring, and diagnosing..."):
                            generated_response = requests.post(
                                f"{backend_url}/generate-triple",
                                files={
                                    "source_file": (
                                        source_file.name,
                                        source_file.getvalue(),
                                        "text/plain",
                                    )
                                },
                                data={"question": source_question.strip()},
                                timeout=300,
                            )
                            generated_response.raise_for_status()
                            generated_triple = generated_response.json()
                            st.session_state.triples = _validate_triples(
                                [generated_triple]
                            )
                            st.session_state.evaluation = _request_report(
                                backend_url, st.session_state.triples
                            )
                        st.success("Generated and evaluated a new triple.")
                    except (requests.RequestException, ValueError) as exc:
                        st.error(f"Could not process source document: {exc}")
            st.container(height=16, border=False)

    st.container(height=24, border=False)
    action_column, status_column = st.columns([3, 1], vertical_alignment="center")
    with action_column:
        run_evaluation = st.button(
            "Run evaluation",
            type="primary",
            disabled=not st.session_state.triples,
            use_container_width=True,
        )
    with status_column:
        st.metric("Ready to evaluate", len(st.session_state.triples))

    if run_evaluation:
        try:
            with st.spinner("Scoring and diagnosing triples..."):
                st.session_state.evaluation = _request_report(
                    backend_url, st.session_state.triples
                )
        except requests.RequestException as exc:
            st.error(f"Backend request failed: {exc}")

    if st.session_state.evaluation:
        _show_report(st.session_state.evaluation)


def _render_tuning_tab(backend_url: str) -> None:
    st.subheader("Auto-tune retrieval settings")
    st.caption(
        "Compare chunk sizes and retrieval counts using the same questions. Leave "
        "the optional settings blank for chunk sizes 100, 150, 200, 300 and Top K "
        "values 2, 3, 5."
    )

    tuning_questions_text = st.text_area(
        "Evaluation questions",
        placeholder=(
            "What role does self-attention play in a Transformer?\n"
            "How is BERT pre-trained?\n"
            "What is multi-head attention?"
        ),
        height=150,
    )
    tuning_questions = [
        question.strip()
        for question in tuning_questions_text.splitlines()
        if question.strip()
    ]

    tuning_settings = st.columns(2, gap="large")
    chunk_sizes_text = tuning_settings[0].text_input(
        "Chunk sizes (optional)", placeholder="Example: 100, 200"
    )
    top_ks_text = tuning_settings[1].text_input(
        "Top K values (optional)", placeholder="Example: 2, 3"
    )

    if st.button(
        "Run auto-tuning",
        disabled=not tuning_questions,
        use_container_width=True,
    ):
        try:
            custom_chunk_sizes = _parse_optional_int_list(
                chunk_sizes_text, "Chunk sizes"
            )
            custom_top_ks = _parse_optional_int_list(top_ks_text, "Top K values")
            tuning_payload: dict[str, object] = {"questions": tuning_questions}
            if custom_chunk_sizes is not None:
                tuning_payload["chunk_sizes"] = custom_chunk_sizes
            if custom_top_ks is not None:
                tuning_payload["top_ks"] = custom_top_ks

            with st.spinner("Testing retrieval configurations..."):
                tuning_response = requests.post(
                    f"{backend_url}/tune",
                    json=tuning_payload,
                    timeout=1800,
                )
                tuning_response.raise_for_status()
                st.session_state.tuning = tuning_response.json()
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Auto-tuning request failed: {exc}")

    if st.session_state.tuning:
        st.divider()
        tuning = st.session_state.tuning
        recommendation = tuning["recommendation"]
        recommended = recommendation["configuration"]

        st.success(recommendation["explanation"])
        table_rows = []
        for result in tuning["sweep_results"]:
            is_recommended = (
                result["chunk_size"] == recommended["chunk_size"]
                and result["top_k"] == recommended["top_k"]
            )
            table_rows.append(
                {
                    "Recommendation": "⭐ Recommended" if is_recommended else "",
                    "Chunk size": result["chunk_size"],
                    "Top K": result["top_k"],
                    "Faithfulness": round(result["faithfulness"], 3),
                    "Answer relevance": round(result["answer_relevance"], 3),
                    "Context precision": round(result["context_precision"], 3),
                    "Combined": round(
                        (
                            result["faithfulness"]
                            + result["answer_relevance"]
                            + result["context_precision"]
                        )
                        / 3,
                        3,
                    ),
                }
            )
        st.dataframe(table_rows, hide_index=True, use_container_width=True)


def _render_history_tab(backend_url: str) -> None:
    st.subheader("Run History")
    st.caption("Save the current evaluation report and compare labeled runs over time.")

    save_column, refresh_column = st.columns([3, 1], gap="large")
    run_label = save_column.text_input(
        "Label for current run", placeholder="v1-baseline"
    )
    save_clicked = save_column.button(
        "Save current run",
        disabled=st.session_state.evaluation is None or not run_label.strip(),
        use_container_width=True,
    )
    refresh_column.write("")
    refresh_clicked = refresh_column.button(
        "Refresh history", use_container_width=True
    )

    if save_clicked:
        try:
            response = requests.post(
                f"{backend_url}/save-run",
                json={
                    "report": st.session_state.evaluation["report"],
                    "label": run_label.strip(),
                },
                timeout=30,
            )
            response.raise_for_status()
            st.session_state.run_history = _request_run_history(backend_url)
            st.success(f"Saved run '{run_label.strip()}'.")
        except requests.RequestException as exc:
            st.error(f"Could not save run: {exc}")

    if refresh_clicked:
        try:
            st.session_state.run_history = _request_run_history(backend_url)
        except requests.RequestException as exc:
            st.error(f"Could not load run history: {exc}")

    if st.session_state.run_history:
        st.divider()
        history_rows = []
        for run in st.session_state.run_history:
            report = run["report"]
            averages = report["average_scores"]
            history_rows.append(
                {
                    "Label": run["label"] or "(unlabeled)",
                    "Timestamp": run["timestamp"],
                    "Passed": report["passed_count"],
                    "Failed": report["failed_count"],
                    "Faithfulness": round(averages["faithfulness"], 3),
                    "Answer relevance": round(averages["answer_relevance"], 3),
                    "Context precision": round(averages["context_precision"], 3),
                }
            )
        st.dataframe(history_rows, hide_index=True, use_container_width=True)

        labeled_runs = [run for run in st.session_state.run_history if run["label"]]
        labels = [run["label"] for run in labeled_runs]
        if len(labels) >= 2:
            st.subheader("Compare runs")
            compare_columns = st.columns(2, gap="large")
            label_a = compare_columns[0].selectbox("Run A", labels, index=0)
            label_b = compare_columns[1].selectbox("Run B", labels, index=1)

            if st.button(
                "Compare selected runs",
                disabled=label_a == label_b,
                use_container_width=True,
            ):
                try:
                    comparison_response = requests.get(
                        f"{backend_url}/compare-runs",
                        params={"label_a": label_a, "label_b": label_b},
                        timeout=30,
                    )
                    comparison_response.raise_for_status()
                    st.session_state.run_comparison = comparison_response.json()
                except requests.RequestException as exc:
                    st.error(f"Could not compare runs: {exc}")

    if st.session_state.run_comparison:
        comparison = st.session_state.run_comparison
        comparison_rows = []
        for metric, label in METRIC_LABELS.items():
            comparison_rows.append(
                {
                    "Metric": label,
                    comparison["label_a"]: round(
                        comparison["average_scores_a"][metric], 3
                    ),
                    comparison["label_b"]: round(
                        comparison["average_scores_b"][metric], 3
                    ),
                    "B − A": round(
                        comparison["difference_b_minus_a"][metric], 3
                    ),
                }
            )
        st.subheader(
            f"Comparison: {comparison['label_a']} vs {comparison['label_b']}"
        )
        st.dataframe(comparison_rows, hide_index=True, use_container_width=True)


st.set_page_config(page_title="RAG Eval Sidekick", page_icon="🔎", layout="wide")
st.title("RAG Eval Sidekick")
st.caption(
    "Evaluate RAG quality, diagnose failures, tune retrieval settings, and track "
    "improvements over time."
)

if "triples" not in st.session_state:
    st.session_state.triples = []
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "tuning" not in st.session_state:
    st.session_state.tuning = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "run_comparison" not in st.session_state:
    st.session_state.run_comparison = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

backend_url = st.sidebar.text_input("Backend URL", DEFAULT_BACKEND_URL).rstrip("/")
evaluate_tab, tuning_tab, history_tab = st.tabs(
    ["🔎 Evaluate", "⚙️ Auto-tune", "🕘 Run History"]
)

with evaluate_tab:
    _render_evaluate_tab(backend_url)

with tuning_tab:
    _render_tuning_tab(backend_url)

with history_tab:
    _render_history_tab(backend_url)
