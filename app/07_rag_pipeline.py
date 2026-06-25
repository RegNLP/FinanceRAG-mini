# Step 07 - RAG Pipeline
#
# Role:
#   Orchestrate retrieval, optional reranking, and answer generation.
#
# Why this exists:
#   The app and evaluation code should call one simple pipeline function instead
#   of manually wiring retrieval, reranking, and generation every time.
#
# Input:
#   User question
#   Runtime settings from config.yaml
#
# Output:
#   Final answer, citations, evidence chunks, and diagnostics.

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

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
generation_module = load_module("generation_module", "06_generation.py")

load_config = config_helpers.load_config
project_path = config_helpers.project_path


def run_rag_pipeline(
    query: str,
    config: dict[str, Any] | None = None,
    method: str | None = None,
    use_reranker: bool | None = None,
    candidate_top_k: int | None = None,
    final_top_k: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run retrieval, optional reranking, and optional answer generation."""
    if config is None:
        config = load_config()

    retrieval_config = config["retrieval"]
    generation_config = config["generation"]
    reranking_config = config["reranking"]

    selected_method = method or retrieval_config["method"]
    selected_use_reranker = (
        bool(retrieval_config["use_reranker"])
        if use_reranker is None
        else use_reranker
    )
    selected_candidate_top_k = candidate_top_k or int(reranking_config["candidate_top_k"])
    selected_final_top_k = final_top_k or int(generation_config["evidence_top_k"])

    evidence_chunks = generation_module.retrieve_evidence(
        query=query,
        config=config,
        method=selected_method,
        use_reranker=selected_use_reranker,
        candidate_top_k=selected_candidate_top_k,
        final_top_k=selected_final_top_k,
    )
    prompt = generation_module.build_generation_prompt(query, evidence_chunks)

    answer = None
    if not dry_run:
        answer = generation_module.generate_answer(
            query=query,
            evidence_chunks=evidence_chunks,
            model=config["models"]["llm_model"],
            max_output_tokens=int(generation_config["max_output_tokens"]),
            temperature=float(generation_config["temperature"]),
        )

    return {
        "query": query,
        "answer": answer,
        "evidence_chunks": evidence_chunks,
        "prompt": prompt,
        "diagnostics": {
            "method": selected_method,
            "use_reranker": selected_use_reranker,
            "candidate_top_k": selected_candidate_top_k,
            "final_top_k": selected_final_top_k,
            "llm_model": config["models"]["llm_model"],
            "embedding_model": config["models"]["embedding_model"],
            "reranker_model": config["models"]["reranker_model"]
            if selected_use_reranker
            else None,
            "dry_run": dry_run,
        },
    }


def print_pipeline_result(result: dict[str, Any]) -> None:
    """Print structured pipeline output in a readable CLI format."""
    print(f"\nQuery:\n{result['query']}")

    if result["answer"] is not None:
        print("\nAnswer:\n")
        print(result["answer"])
    else:
        print("\nDry run: no LLM call was made.")
        print("\nPrompt:\n")
        print(result["prompt"])

    print("\nEvidence:\n")
    for chunk in result["evidence_chunks"]:
        print(
            "- "
            f"{chunk['chunk_id']} | "
            f"{chunk['document_title']} | "
            f"pages {chunk['page_start']}-{chunk['page_end']} | "
            f"score {chunk['score']:.6f}"
        )

    print("\nDiagnostics:\n")
    for key, value in result["diagnostics"].items():
        print(f"- {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RAG pipeline.")
    parser.add_argument("query", help="User question to answer.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval method. Defaults to config.yaml.",
    )
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--candidate-top-k", type=int, default=None)
    parser.add_argument("--final-top-k", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(project_path(".env"), override=True)
    result = run_rag_pipeline(
        query=args.query,
        method=args.method,
        use_reranker=True if args.use_reranker else None,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        dry_run=args.dry_run,
    )
    print_pipeline_result(result)


if __name__ == "__main__":
    main()
