# Step 05 - Reranking
#
# Role:
#   Rerank retrieved candidate chunks by query-passage relevance.
#
# Why this exists:
#   First-stage retrieval is fast but imperfect. A reranker can compare the
#   question and each candidate passage more carefully before generation.
#
# Input:
#   User question
#   Candidate chunks from 04_retrieval.py
#
# Output:
#   Better ordered evidence chunks.

from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

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

load_config = config_helpers.load_config
Retriever = retrieval_module.Retriever
print_results = retrieval_module.print_results


class Reranker:
    """Rerank candidate chunks with a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        # Lazy import avoids loading cross-encoder internals before dense retrieval.
        # On some local Torch/Transformers stacks, import order can affect stability.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        final_top_k: int,
    ) -> list[dict[str, Any]]:
        """Score query/chunk pairs and return the best candidates."""
        if not candidates:
            return []

        pairs = [(query, candidate["chunk_text"]) for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            updated = candidate.copy()
            updated["retrieval_rank"] = candidate["rank"]
            updated["retrieval_score"] = candidate["score"]
            updated["rerank_score"] = float(score)
            updated["method"] = f"{candidate['method']}+rerank"
            reranked.append(updated)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        for rank, result in enumerate(reranked[:final_top_k], start=1):
            result["rank"] = rank
            result["score"] = result["rerank_score"]

        return reranked[:final_top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve candidates and rerank them.")
    parser.add_argument("query", help="User question to search for.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="First-stage retrieval method. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=None,
        help="Number of first-stage candidates to rerank.",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help="Number of reranked results to return.",
    )
    args = parser.parse_args()

    config = load_config()
    retrieval_config = config["retrieval"]
    reranking_config = config["reranking"]

    method = args.method or retrieval_config["method"]
    candidate_top_k = args.candidate_top_k or int(reranking_config["candidate_top_k"])
    final_top_k = args.final_top_k or int(reranking_config["final_top_k"])

    retriever = Retriever(config)
    candidates = retriever.retrieve(
        query=args.query,
        method=method,
        bm25_top_k=max(int(retrieval_config["bm25_top_k"]), candidate_top_k),
        dense_top_k=max(int(retrieval_config["dense_top_k"]), candidate_top_k),
        final_top_k=candidate_top_k,
    )
    del retriever
    gc.collect()

    reranker = Reranker(config["models"]["reranker_model"])
    reranked_results = reranker.rerank(
        query=args.query,
        candidates=candidates,
        final_top_k=final_top_k,
    )

    print_results(args.query, reranked_results)


if __name__ == "__main__":
    main()
