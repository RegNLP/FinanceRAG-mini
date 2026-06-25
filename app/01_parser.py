# Step 01 - PDF Parser
#
# Role:
#   Extract page-level text and metadata from financial PDF files.
#
# Why this exists:
#   RAG cannot search PDFs directly. First, we convert each PDF page into a
#   structured JSONL record that later steps can clean, chunk, index, and search.
#
# Input:
#   data/raw_pdfs/*.pdf
#
# Output:
#   data/parsed_text/documents.jsonl
#
# ID levels:
#   document_id: identifies the whole PDF, so it repeats for every page.
#   page_id: identifies one exact page record and must be unique.
#   chunk_id: will be created later in 02_chunker.py.

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm


def load_config_helpers() -> tuple[Any, Any]:
    """Load helpers from 00_config.py.

    The file starts with a number so Python cannot import it with a normal
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


def extract_pdf_pages(pdf_path: Path, raw_pdf_dir: Path) -> list[dict[str, Any]]:
    """Extract text from one PDF and return one record per page."""
    records: list[dict[str, Any]] = []
    document_id = pdf_path.stem

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if not text:
                continue

            records.append(
                {
                    "page_id": f"{document_id}_page_{page_index:04d}",
                    "document_id": document_id,
                    "document_title": pdf_path.stem,
                    "source_file": str(pdf_path.relative_to(raw_pdf_dir.parent.parent)),
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "text": text,
                    "char_count": len(text),
                }
            )

    return records


def parse_pdfs(raw_pdf_dir: Path, output_path: Path, limit: int | None = None) -> int:
    """Parse PDFs from raw_pdf_dir and write page records as JSONL."""
    pdf_paths = sorted(raw_pdf_dir.glob("*.pdf"))

    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for pdf_path in tqdm(pdf_paths, desc="Parsing PDFs"):
            page_records = extract_pdf_pages(pdf_path, raw_pdf_dir)

            for record in page_records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_pages += len(page_records)

    return total_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PDFs into page-level JSONL.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of PDFs to parse. Useful while learning/testing.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Parse all PDFs instead of the configured first-run limit.",
    )
    args = parser.parse_args()

    config = load_config()
    raw_pdf_dir = project_path(config["paths"]["raw_pdf_dir"])
    output_path = project_path(config["paths"]["parsed_text_path"])

    limit = args.limit
    if limit is None and not args.all:
        limit = int(config["parser"]["max_pdfs_for_first_run"])

    total_pages = parse_pdfs(raw_pdf_dir, output_path, limit=limit)

    print(f"Raw PDF directory: {raw_pdf_dir}")
    print(f"Output path: {output_path}")
    print(f"PDF limit: {limit if limit is not None else 'all'}")
    print(f"Page records written: {total_pages}")


if __name__ == "__main__":
    main()
