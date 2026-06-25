# Step 02 - Chunker
#
# Role:
#   Clean parsed page text and split documents into metadata-rich chunks.
#
# Why this exists:
#   LLMs and embedding models work better with focused passages than with
#   entire long reports. Chunking creates searchable evidence units.
#
# Input:
#   data/parsed_text/documents.jsonl
#
# Output:
#   data/chunks/chunks.jsonl
#
# ID levels:
#   page_id comes from 01_parser.py.
#   chunk_id will uniquely identify each searchable passage.

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import tiktoken
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
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one JSON object per line from a JSONL file."""
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def clean_text(text: str) -> str:
    """Apply light cleanup while keeping the original document wording."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_count(text: str) -> int:
    """Count tokens using the same style of tokenizer used by many OpenAI models."""
    return len(TOKENIZER.encode(text))


def split_long_text(text: str, max_tokens: int) -> list[str]:
    """Split a single oversized text segment into token-sized pieces."""
    token_ids = TOKENIZER.encode(text)
    pieces: list[str] = []

    for start in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[start : start + max_tokens]
        pieces.append(TOKENIZER.decode(piece_ids).strip())

    return [piece for piece in pieces if piece]


def split_page_into_segments(page_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one page into smaller paragraph-like segments."""
    text = clean_text(page_record["text"])
    raw_segments = re.split(r"\n\s*\n", text)

    segments = []
    for raw_segment in raw_segments:
        segment_text = clean_text(raw_segment)
        if not segment_text:
            continue

        segments.append(
            {
                "text": segment_text,
                "page_id": page_record["page_id"],
                "page_number": page_record["page_number"],
            }
        )

    return segments


def make_chunk_record(
    document_id: str,
    document_title: str,
    source_file: str,
    chunk_number: int,
    chunk_text: str,
    source_page_ids: list[str],
    page_numbers: list[int],
) -> dict[str, Any]:
    """Create one chunk record with stable metadata."""
    return {
        "chunk_id": f"{document_id}_chunk_{chunk_number:04d}",
        "document_id": document_id,
        "document_title": document_title,
        "source_file": source_file,
        "page_start": min(page_numbers),
        "page_end": max(page_numbers),
        "source_page_ids": list(dict.fromkeys(source_page_ids)),
        "chunk_text": chunk_text,
        "token_count": token_count(chunk_text),
        "char_count": len(chunk_text),
    }


def chunk_document(
    page_records: list[dict[str, Any]],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Turn one document's page records into searchable chunks."""
    if not page_records:
        return []

    document_id = page_records[0]["document_id"]
    document_title = page_records[0]["document_title"]
    source_file = page_records[0]["source_file"]

    chunks: list[dict[str, Any]] = []
    current_text_parts: list[str] = []
    current_page_ids: list[str] = []
    current_page_numbers: list[int] = []
    chunk_number = 1

    def flush_current() -> None:
        nonlocal current_text_parts
        nonlocal current_page_ids
        nonlocal current_page_numbers
        nonlocal chunk_number

        chunk_text = clean_text("\n\n".join(current_text_parts))
        if not chunk_text:
            current_text_parts = []
            current_page_ids = []
            current_page_numbers = []
            return

        chunks.append(
            make_chunk_record(
                document_id=document_id,
                document_title=document_title,
                source_file=source_file,
                chunk_number=chunk_number,
                chunk_text=chunk_text,
                source_page_ids=current_page_ids,
                page_numbers=current_page_numbers,
            )
        )
        chunk_number += 1

        if chunk_overlap_tokens > 0:
            overlap_ids = TOKENIZER.encode(chunk_text)[-chunk_overlap_tokens:]
            overlap_text = TOKENIZER.decode(overlap_ids).strip()
            last_page_id = current_page_ids[-1]
            last_page_number = current_page_numbers[-1]
            current_text_parts = [overlap_text] if overlap_text else []
            current_page_ids = [last_page_id] if overlap_text else []
            current_page_numbers = [last_page_number] if overlap_text else []
        else:
            current_text_parts = []
            current_page_ids = []
            current_page_numbers = []

    for page_record in page_records:
        for segment in split_page_into_segments(page_record):
            segment_text = segment["text"]
            segment_tokens = token_count(segment_text)

            if segment_tokens > chunk_size_tokens:
                flush_current()

                for piece in split_long_text(segment_text, chunk_size_tokens):
                    chunks.append(
                        make_chunk_record(
                            document_id=document_id,
                            document_title=document_title,
                            source_file=source_file,
                            chunk_number=chunk_number,
                            chunk_text=piece,
                            source_page_ids=[segment["page_id"]],
                            page_numbers=[segment["page_number"]],
                        )
                    )
                    chunk_number += 1
                continue

            candidate_text = clean_text("\n\n".join(current_text_parts + [segment_text]))
            if current_text_parts and token_count(candidate_text) > chunk_size_tokens:
                flush_current()

            current_text_parts.append(segment_text)
            current_page_ids.append(segment["page_id"])
            current_page_numbers.append(segment["page_number"])

    flush_current()
    return chunks


def group_pages_by_document(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group page records by document_id."""
    pages_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        pages_by_document[record["document_id"]].append(record)

    for page_records in pages_by_document.values():
        page_records.sort(key=lambda record: record["page_index"])

    return dict(pages_by_document)


def build_chunks(
    parsed_text_path: Path,
    output_path: Path,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    limit_documents: int | None = None,
) -> tuple[int, int]:
    """Read parsed pages and write chunk records as JSONL."""
    pages_by_document = group_pages_by_document(iter_jsonl(parsed_text_path))
    document_ids = sorted(pages_by_document)

    if limit_documents is not None:
        document_ids = document_ids[:limit_documents]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for document_id in tqdm(document_ids, desc="Chunking documents"):
            chunks = chunk_document(
                pages_by_document[document_id],
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )

            for chunk in chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            chunk_count += len(chunks)

    return len(document_ids), chunk_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create searchable chunks from parsed pages.")
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=None,
        help="Maximum number of parsed documents to chunk. Useful while learning/testing.",
    )
    args = parser.parse_args()

    config = load_config()
    parsed_text_path = project_path(config["paths"]["parsed_text_path"])
    output_path = project_path(config["paths"]["chunk_path"])
    chunk_size_tokens = int(config["chunking"]["chunk_size_tokens"])
    chunk_overlap_tokens = int(config["chunking"]["chunk_overlap_tokens"])

    document_count, chunk_count = build_chunks(
        parsed_text_path=parsed_text_path,
        output_path=output_path,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        limit_documents=args.limit_documents,
    )

    print(f"Parsed text path: {parsed_text_path}")
    print(f"Output path: {output_path}")
    print(f"Chunk size: {chunk_size_tokens} tokens")
    print(f"Chunk overlap: {chunk_overlap_tokens} tokens")
    print(f"Documents chunked: {document_count}")
    print(f"Chunks written: {chunk_count}")


if __name__ == "__main__":
    main()
