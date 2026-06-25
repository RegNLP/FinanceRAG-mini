# Step 09 - Streamlit App
#
# Role:
#   Provide the Streamlit user interface for asking questions and inspecting evidence.
#
# Why this exists:
#   The UI lets us interact with the RAG system, inspect retrieved chunks, and
#   understand failure cases visually.
#
# Input:
#   User question and retrieval settings from the browser UI.
#
# Output:
#   Generated answer, citations, evidence display, diagnostics, and feedback form.

from __future__ import annotations

import csv
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv


def load_module(module_name: str, file_name: str) -> Any:
    """Load another numbered pipeline file."""
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_helpers = load_module("config_helpers", "00_config.py")
pipeline_module = load_module("pipeline_module", "07_rag_pipeline.py")

load_config = config_helpers.load_config
project_path = config_helpers.project_path
run_rag_pipeline = pipeline_module.run_rag_pipeline


def save_feedback(feedback_path: Path, row: dict[str, Any]) -> None:
    """Append one human feedback row to CSV."""
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = feedback_path.exists()

    with feedback_path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def render_evidence(evidence_chunks: list[dict[str, Any]]) -> None:
    """Render retrieved evidence chunks."""
    for chunk in evidence_chunks:
        title = (
            f"{chunk['rank']}. {chunk['chunk_id']} | "
            f"{chunk['document_title']} | pages {chunk['page_start']}-{chunk['page_end']}"
        )
        with st.expander(title):
            st.write(
                {
                    "method": chunk["method"],
                    "score": chunk["score"],
                    "document_id": chunk["document_id"],
                    "source_file": chunk["source_file"],
                    "token_count": chunk.get("token_count"),
                }
            )
            st.text_area(
                "Chunk text",
                chunk["chunk_text"],
                height=260,
                key=f"text_{chunk['chunk_id']}",
            )


def main() -> None:
    st.set_page_config(page_title="FinanceRAG Mini", layout="wide")
    load_dotenv(project_path(".env"), override=True)
    config = load_config()

    st.title("FinanceRAG Mini")
    st.caption("End-to-end RAG over FinanceBench PDFs with citations and diagnostics.")

    with st.sidebar:
        st.header("Retrieval")
        method = st.selectbox(
            "Method",
            options=["hybrid", "bm25", "dense"],
            index=["hybrid", "bm25", "dense"].index(config["retrieval"]["method"]),
        )
        final_top_k = st.slider(
            "Evidence chunks",
            min_value=1,
            max_value=10,
            value=int(config["generation"]["evidence_top_k"]),
        )
        use_reranker = st.checkbox(
            "Use reranker",
            value=bool(config["retrieval"]["use_reranker"]),
        )
        candidate_top_k = st.slider(
            "Reranker candidates",
            min_value=5,
            max_value=50,
            value=int(config["reranking"]["candidate_top_k"]),
            step=5,
            disabled=not use_reranker,
        )
        dry_run = st.checkbox("Dry run", value=False)

    examples = [
        "What was Netflix revenue in 2017?",
        "What is the FY2018 capital expenditure amount for 3M?",
        "Who won the FIFA World Cup in 2018?",
    ]

    with st.form("query_form"):
        query = st.text_area(
            "Question",
            value=examples[0],
            height=90,
        )
        submitted = st.form_submit_button("Run RAG")

    if submitted:
        if not query.strip():
            st.warning("Enter a question.")
            return

        with st.spinner("Running retrieval and generation..."):
            try:
                result = run_rag_pipeline(
                    query=query.strip(),
                    config=config,
                    method=method,
                    use_reranker=use_reranker,
                    candidate_top_k=candidate_top_k,
                    final_top_k=final_top_k,
                    dry_run=dry_run,
                )
            except Exception as exc:
                st.error(str(exc))
                return

        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.info("Ask a question to retrieve evidence and generate an answer.")
        return

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Answer")
        if result["answer"] is None:
            st.warning("Dry run mode: no LLM call was made.")
            st.text_area("Prompt", result["prompt"], height=420)
        else:
            st.markdown(result["answer"])

        st.subheader("Human Feedback")
        with st.form("feedback_form"):
            answer_quality = st.selectbox(
                "Answer correct?",
                ["yes", "partial", "no", "not sure"],
            )
            evidence_quality = st.selectbox(
                "Evidence sufficient?",
                ["yes", "partial", "no", "not sure"],
            )
            hallucination = st.selectbox(
                "Hallucination?",
                ["no", "yes", "not sure"],
            )
            notes = st.text_area("Notes", height=90)
            feedback_submitted = st.form_submit_button("Save feedback")

        if feedback_submitted:
            feedback_path = project_path(config["paths"]["human_feedback_path"])
            save_feedback(
                feedback_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "query": result["query"],
                    "answer": result["answer"] or "",
                    "answer_quality": answer_quality,
                    "evidence_quality": evidence_quality,
                    "hallucination": hallucination,
                    "notes": notes,
                    "method": result["diagnostics"]["method"],
                    "use_reranker": result["diagnostics"]["use_reranker"],
                    "evidence_chunk_ids": ",".join(
                        chunk["chunk_id"] for chunk in result["evidence_chunks"]
                    ),
                },
            )
            st.success(f"Feedback saved to {feedback_path}")

    with right:
        st.subheader("Diagnostics")
        st.json(result["diagnostics"])

        evidence_table = pd.DataFrame(
            [
                {
                    "rank": chunk["rank"],
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["document_title"],
                    "pages": f"{chunk['page_start']}-{chunk['page_end']}",
                    "score": chunk["score"],
                    "method": chunk["method"],
                }
                for chunk in result["evidence_chunks"]
            ]
        )
        st.dataframe(evidence_table, use_container_width=True)

    st.subheader("Evidence")
    render_evidence(result["evidence_chunks"])


if __name__ == "__main__":
    main()
