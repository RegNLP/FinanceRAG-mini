# Step 04 - Retrieval
#
# Role:
#   Retrieve relevant chunks using BM25, dense vector search, or hybrid search.
#
# Why this exists:
#   Retrieval is the "R" in RAG. Given a user question, this step finds the
#   most relevant evidence chunks before any LLM answer is generated.
#
# Input:
#   User question
#   Saved BM25 / FAISS indexes
#   Chunk metadata
#
# Output:
#   Ranked evidence chunks with scores and metadata.

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import faiss
from sentence_transformers import SentenceTransformer


def load_config_helpers() -> tuple[Any, Any]:
    """Load helpers from 00_config.py.

    The file starts with a number, so Python cannot import it with a normal
    import statement. We use importlib here to keep the numbered learning order.
    """
    config_path = Path(__file__).with_name("00_config.py")
    spec = importlib.util.spec_from_file_location("config_helpers", config_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config helpers from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_config, module.project_path


load_config, project_path = load_config_helpers()


def tokenize_for_bm25(text: str) -> list[str]:
    """Use the same tokenizer used while building the BM25 index."""
    return re.findall(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)?", text.lower())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one JSON object per line from a JSONL file."""
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load chunk metadata in index order."""
    metadata = list(iter_jsonl(metadata_path))
    metadata.sort(key=lambda row: row["index_position"])
    return metadata


def load_bm25_payload(bm25_index_path: Path) -> dict[str, Any]:
    """Load BM25 pickle payload."""
    with bm25_index_path.open("rb") as input_file:
        return pickle.load(input_file)


def attach_result_fields(
    metadata: dict[str, Any],
    rank: int,
    score: float,
    method: str,
) -> dict[str, Any]:
    """Attach retrieval details to a chunk metadata record."""
    return {
        "rank": rank,
        "score": float(score),
        "method": method,
        **metadata,
    }


def bm25_search(
    query: str,
    bm25_payload: dict[str, Any],
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Retrieve chunks with BM25 keyword search."""
    bm25 = bm25_payload["bm25"]
    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    top_positions = scores.argsort()[::-1][:top_k]

    return [
        attach_result_fields(
            metadata=metadata[int(position)],
            rank=rank,
            score=float(scores[int(position)]),
            method="bm25",
        )
        for rank, position in enumerate(top_positions, start=1)
    ]


def dense_search(
    query: str,
    faiss_index: faiss.Index,
    embedding_model: SentenceTransformer,
    metadata: list[dict[str, Any]],
    top_k: int,
    normalize_embeddings: bool,
) -> list[dict[str, Any]]:
    """Retrieve chunks with FAISS dense vector search."""
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    scores, positions = faiss_index.search(query_embedding, top_k)

    results: list[dict[str, Any]] = []
    for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1):
        if int(position) < 0:
            continue
        results.append(
            attach_result_fields(
                metadata=metadata[int(position)],
                rank=rank,
                score=float(score),
                method="dense",
            )
        )

    return results


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    final_top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine ranked result lists with Reciprocal Rank Fusion."""
    fused_scores: dict[str, float] = {}
    records_by_chunk_id: dict[str, dict[str, Any]] = {}
    sources_by_chunk_id: dict[str, list[str]] = {}

    for results in result_lists:
        for result in results:
            chunk_id = result["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + result["rank"])
            )
            records_by_chunk_id.setdefault(chunk_id, result)
            sources_by_chunk_id.setdefault(chunk_id, []).append(result["method"])

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )[:final_top_k]

    fused_results: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        result = records_by_chunk_id[chunk_id].copy()
        result["rank"] = rank
        result["score"] = fused_scores[chunk_id]
        result["method"] = "hybrid"
        result["retrieval_sources"] = sorted(set(sources_by_chunk_id[chunk_id]))
        fused_results.append(result)

    return fused_results


class Retriever:
    """Load indexes once and retrieve chunks for user questions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.metadata = load_metadata(project_path(config["paths"]["chunk_metadata_path"]))
        self.bm25_payload = load_bm25_payload(project_path(config["paths"]["bm25_index_path"]))
        self.faiss_index: faiss.Index | None = None
        self.embedding_model: SentenceTransformer | None = None
        self.normalize_embeddings = bool(config["indexing"]["normalize_embeddings"])

    def load_dense_resources(self) -> tuple[faiss.Index, SentenceTransformer]:
        """Load FAISS and the embedding model only when dense search is needed."""
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(str(project_path(self.config["paths"]["faiss_index_path"])))

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.config["models"]["embedding_model"])

        return self.faiss_index, self.embedding_model

    def retrieve(
        self,
        query: str,
        method: str,
        bm25_top_k: int,
        dense_top_k: int,
        final_top_k: int,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks with bm25, dense, or hybrid mode."""
        if method == "bm25":
            return bm25_search(
                query=query,
                bm25_payload=self.bm25_payload,
                metadata=self.metadata,
                top_k=final_top_k,
            )

        if method == "dense":
            faiss_index, embedding_model = self.load_dense_resources()
            return dense_search(
                query=query,
                faiss_index=faiss_index,
                embedding_model=embedding_model,
                metadata=self.metadata,
                top_k=final_top_k,
                normalize_embeddings=self.normalize_embeddings,
            )

        if method == "hybrid":
            faiss_index, embedding_model = self.load_dense_resources()
            bm25_results = bm25_search(
                query=query,
                bm25_payload=self.bm25_payload,
                metadata=self.metadata,
                top_k=bm25_top_k,
            )
            dense_results = dense_search(
                query=query,
                faiss_index=faiss_index,
                embedding_model=embedding_model,
                metadata=self.metadata,
                top_k=dense_top_k,
                normalize_embeddings=self.normalize_embeddings,
            )
            return reciprocal_rank_fusion(
                [bm25_results, dense_results],
                final_top_k=final_top_k,
            )

        raise ValueError(f"Unknown retrieval method: {method}")


def print_results(query: str, results: list[dict[str, Any]]) -> None:
    """Print retrieval results in a readable command-line format."""
    print(f"\nQuery: {query}\n")

    for result in results:
        page_range = f"{result['page_start']}-{result['page_end']}"
        preview = result["chunk_text"].replace("\n", " ")[:500]
        sources = result.get("retrieval_sources", [result["method"]])
        sources_text = ",".join(sources)

        print(f"{result['rank']}. {result['chunk_id']}")
        print(f"   method: {result['method']} ({sources_text})")
        print(f"   score: {result['score']:.6f}")
        print(f"   document: {result['document_title']}")
        print(f"   pages: {page_range}")
        print(f"   preview: {preview}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks for a question.")
    parser.add_argument("query", help="User question to search for.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval method. Defaults to config.yaml.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Final number of results.")
    args = parser.parse_args()

    config = load_config()
    retrieval_config = config["retrieval"]

    method = args.method or retrieval_config["method"]
    final_top_k = args.top_k or int(retrieval_config["final_top_k"])

    retriever = Retriever(config)
    results = retriever.retrieve(
        query=args.query,
        method=method,
        bm25_top_k=int(retrieval_config["bm25_top_k"]),
        dense_top_k=int(retrieval_config["dense_top_k"]),
        final_top_k=final_top_k,
    )
    print_results(args.query, results)


if __name__ == "__main__":
    main()
