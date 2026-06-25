# Step 03 - Indexer
#
# Role:
#   Build and save BM25 and FAISS indexes from document chunks.
#
# Why this exists:
#   Retrieval needs fast search structures. BM25 supports keyword search;
#   FAISS supports semantic vector search over embeddings.
#
# Input:
#   data/chunks/chunks.jsonl
#
# Output:
#   data/indexes/bm25.pkl
#   data/indexes/faiss.index
#   data/indexes/chunk_metadata.jsonl

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one JSON object per line from a JSONL file."""
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25 keyword search."""
    return re.findall(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)?", text.lower())


def load_chunks(chunk_path: Path, limit_chunks: int | None = None) -> list[dict[str, Any]]:
    """Load chunk records from JSONL."""
    chunks: list[dict[str, Any]] = []

    for chunk in iter_jsonl(chunk_path):
        chunks.append(chunk)

        if limit_chunks is not None and len(chunks) >= limit_chunks:
            break

    if not chunks:
        raise ValueError(f"No chunks found in {chunk_path}")

    return chunks


def write_chunk_metadata(chunks: list[dict[str, Any]], metadata_path: Path) -> None:
    """Save chunk metadata in the exact order used by BM25 and FAISS."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("w", encoding="utf-8") as output_file:
        for index_position, chunk in enumerate(chunks):
            metadata = {
                "index_position": index_position,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "source_file": chunk["source_file"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "source_page_ids": chunk["source_page_ids"],
                "chunk_text": chunk["chunk_text"],
                "token_count": chunk["token_count"],
                "char_count": chunk["char_count"],
            }
            output_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def build_bm25_index(chunks: list[dict[str, Any]], bm25_index_path: Path) -> None:
    """Build and save a BM25 keyword index."""
    tokenized_corpus = [
        tokenize_for_bm25(chunk["chunk_text"])
        for chunk in tqdm(chunks, desc="Tokenizing chunks for BM25")
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_index_path.open("wb") as output_file:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "tokenizer": "regex_lowercase_words",
            },
            output_file,
        )


def build_faiss_index(
    chunks: list[dict[str, Any]],
    faiss_index_path: Path,
    embedding_model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> None:
    """Build and save a FAISS dense vector index."""
    texts = [chunk["chunk_text"] for chunk in chunks]
    model = SentenceTransformer(embedding_model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    if normalize_embeddings:
        index = faiss.IndexFlatIP(dimension)
    else:
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path))


def build_indexes(
    chunk_path: Path,
    bm25_index_path: Path,
    faiss_index_path: Path,
    metadata_path: Path,
    embedding_model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    limit_chunks: int | None = None,
    skip_dense: bool = False,
) -> int:
    """Build metadata, BM25, and optionally FAISS indexes."""
    chunks = load_chunks(chunk_path, limit_chunks=limit_chunks)

    write_chunk_metadata(chunks, metadata_path)
    build_bm25_index(chunks, bm25_index_path)

    if not skip_dense:
        build_faiss_index(
            chunks=chunks,
            faiss_index_path=faiss_index_path,
            embedding_model_name=embedding_model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes from chunks.")
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Maximum number of chunks to index. Useful while learning/testing.",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Only build metadata and BM25. Skip embeddings and FAISS.",
    )
    args = parser.parse_args()

    config = load_config()
    chunk_path = project_path(config["paths"]["chunk_path"])
    bm25_index_path = project_path(config["paths"]["bm25_index_path"])
    faiss_index_path = project_path(config["paths"]["faiss_index_path"])
    metadata_path = project_path(config["paths"]["chunk_metadata_path"])
    embedding_model_name = config["models"]["embedding_model"]
    batch_size = int(config["indexing"]["embedding_batch_size"])
    normalize_embeddings = bool(config["indexing"]["normalize_embeddings"])

    chunk_count = build_indexes(
        chunk_path=chunk_path,
        bm25_index_path=bm25_index_path,
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        limit_chunks=args.limit_chunks,
        skip_dense=args.skip_dense,
    )

    print(f"Chunk path: {chunk_path}")
    print(f"Chunks indexed: {chunk_count}")
    print(f"BM25 index path: {bm25_index_path}")
    print(f"FAISS index path: {faiss_index_path}")
    print(f"Chunk metadata path: {metadata_path}")
    print(f"Dense indexing skipped: {args.skip_dense}")


if __name__ == "__main__":
    main()
