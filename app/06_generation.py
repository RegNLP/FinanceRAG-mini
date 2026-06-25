# Step 06 - Generation
#
# Role:
#   Generate source-grounded answers from retrieved evidence passages.
#
# Why this exists:
#   The LLM should not answer from memory. It should answer only from retrieved
#   chunks and cite the evidence used.
#
# Input:
#   User question
#   Top evidence chunks
#
# Output:
#   Answer text with citations and limitations.

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def load_module(module_name: str, file_name: str) -> Any:
    """Load another numbered pipeline file.

    Files like 00_config.py and 04_retrieval.py start with numbers, so Python
    cannot import them with normal import statements.
    """
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config_helpers = load_module("config_helpers", "00_config.py")
retrieval_module = load_module("retrieval_module", "04_retrieval.py")
reranking_module = load_module("reranking_module", "05_reranking.py")

load_config = config_helpers.load_config
project_path = config_helpers.project_path
Retriever = retrieval_module.Retriever
Reranker = reranking_module.Reranker


SYSTEM_INSTRUCTIONS = """You are a financial document RAG assistant.
Answer the user's question using only the provided evidence passages.
If the evidence is insufficient, say that the provided evidence is insufficient.
Do not invent facts, calculations, page numbers, or citations.
Do not provide investment advice.
Keep the answer concise.
Cite evidence with chunk IDs in square brackets, for example [NETFLIX_2017_10K_chunk_0045].
"""


def format_evidence(evidence_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    formatted_chunks: list[str] = []

    for chunk in evidence_chunks:
        formatted_chunks.append(
            "\n".join(
                [
                    f"CHUNK_ID: {chunk['chunk_id']}",
                    f"DOCUMENT: {chunk['document_title']}",
                    f"PAGES: {chunk['page_start']}-{chunk['page_end']}",
                    f"RETRIEVAL_METHOD: {chunk['method']}",
                    f"SCORE: {chunk['score']}",
                    "TEXT:",
                    chunk["chunk_text"],
                ]
            )
        )

    return "\n\n---\n\n".join(formatted_chunks)


def build_generation_prompt(query: str, evidence_chunks: list[dict[str, Any]]) -> str:
    """Create the user prompt sent to the LLM."""
    evidence_text = format_evidence(evidence_chunks)

    return f"""User question:
{query}

Evidence passages:
{evidence_text}

Answer format:
Answer:
<concise answer grounded only in the evidence>

Sources:
<chunk IDs used>

Limitations:
<brief note if evidence is incomplete or ambiguous>
"""


def retrieve_evidence(
    query: str,
    config: dict[str, Any],
    method: str,
    use_reranker: bool,
    candidate_top_k: int,
    final_top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve and optionally rerank evidence chunks."""
    retrieval_config = config["retrieval"]

    retriever = Retriever(config)
    candidates = retriever.retrieve(
        query=query,
        method=method,
        bm25_top_k=max(int(retrieval_config["bm25_top_k"]), candidate_top_k),
        dense_top_k=max(int(retrieval_config["dense_top_k"]), candidate_top_k),
        final_top_k=candidate_top_k,
    )
    del retriever
    gc.collect()

    if not use_reranker:
        return candidates[:final_top_k]

    reranker = Reranker(config["models"]["reranker_model"])
    return reranker.rerank(
        query=query,
        candidates=candidates,
        final_top_k=final_top_k,
    )


def generate_answer(
    query: str,
    evidence_chunks: list[dict[str, Any]],
    model: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI to generate a source-grounded answer."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or run with --dry-run."
        )

    client = OpenAI()
    prompt = build_generation_prompt(query, evidence_chunks)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    return str(response)


def print_evidence_summary(evidence_chunks: list[dict[str, Any]]) -> None:
    """Print the evidence chunks used for generation."""
    print("\nEvidence used:\n")
    for chunk in evidence_chunks:
        page_range = f"{chunk['page_start']}-{chunk['page_end']}"
        print(f"- {chunk['chunk_id']} | {chunk['document_title']} | pages {page_range}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a source-grounded RAG answer.")
    parser.add_argument("query", help="User question to answer.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval method. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Rerank retrieved candidates before generation.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        help="Number of first-stage candidates before reranking.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help="Number of evidence chunks sent to generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated prompt without calling OpenAI.",
    )
    args = parser.parse_args()

    load_dotenv(project_path(".env"), override=True)
    config = load_config()

    method = args.method or config["retrieval"]["method"]
    use_reranker = args.use_reranker or bool(config["retrieval"]["use_reranker"])
    candidate_top_k = args.candidate_top_k or int(config["reranking"]["candidate_top_k"])
    final_top_k = args.final_top_k or int(config["generation"]["evidence_top_k"])

    evidence_chunks = retrieve_evidence(
        query=args.query,
        config=config,
        method=method,
        use_reranker=use_reranker,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )

    prompt = build_generation_prompt(args.query, evidence_chunks)

    if args.dry_run:
        print("\nSYSTEM INSTRUCTIONS:\n")
        print(SYSTEM_INSTRUCTIONS)
        print("\nUSER PROMPT:\n")
        print(prompt)
        print_evidence_summary(evidence_chunks)
        return

    answer = generate_answer(
        query=args.query,
        evidence_chunks=evidence_chunks,
        model=config["models"]["llm_model"],
        max_output_tokens=int(config["generation"]["max_output_tokens"]),
        temperature=float(config["generation"]["temperature"]),
    )

    print("\nGenerated answer:\n")
    print(answer)
    print_evidence_summary(evidence_chunks)


if __name__ == "__main__":
    main()
