# Step 08 - Evaluation
#
# Role:
#   Evaluate retrieval quality, answer grounding, and human feedback.
#
# Why this exists:
#   A RAG system must be checked, not only demonstrated. FinanceBench gives us
#   questions, expected answers, and evidence strings for evaluation.
#
# Input:
#   data/evaluation/financebench_open_source.jsonl
#   Retrieved chunks and generated answers
#
# Output:
#   Retrieval metrics, answer checks, and optional human feedback records.

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pytrec_eval
from tqdm import tqdm


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
retrieval_module = load_module("retrieval_module", "04_retrieval.py")

load_config = config_helpers.load_config
project_path = config_helpers.project_path
Retriever = retrieval_module.Retriever


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one JSON object per line from a JSONL file."""
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def load_financebench_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load FinanceBench question records."""
    records: list[dict[str, Any]] = []

    for record in iter_jsonl(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break

    return records


def gold_pages(record: dict[str, Any]) -> set[int]:
    """Extract gold evidence page numbers from a FinanceBench record."""
    pages: set[int] = set()

    for evidence in record.get("evidence", []):
        page_num = evidence.get("evidence_page_num")
        if page_num is None:
            continue
        page = int(page_num)
        pages.add(page)
        # FinanceBench stores PDF/evidence page numbers that can be offset from
        # our one-based parser page_number. Include neighbors for this first
        # retrieval evaluation so page-label differences do not dominate.
        pages.add(page + 1)
        if page > 0:
            pages.add(page - 1)

    return pages


def result_overlaps_gold_page(result: dict[str, Any], pages: set[int]) -> bool:
    """Return True if a retrieved chunk overlaps any gold evidence page."""
    if not pages:
        return False

    retrieved_pages = set(range(int(result["page_start"]), int(result["page_end"]) + 1))
    return bool(retrieved_pages & pages)


def chunk_overlaps_gold_page(chunk: dict[str, Any], pages: set[int]) -> bool:
    """Return True if a metadata chunk overlaps any gold evidence page."""
    if not pages:
        return False

    chunk_pages = set(range(int(chunk["page_start"]), int(chunk["page_end"]) + 1))
    return bool(chunk_pages & pages)


def build_qrels_for_questions(
    questions: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Build pytrec_eval qrels from FinanceBench gold doc/page evidence.

    pytrec_eval evaluates query_id -> document_id relevance labels. Our
    retrieval unit is chunk_id, while FinanceBench labels doc_name and evidence
    page. So we mark chunks as relevant when they belong to the gold document
    and overlap a gold evidence page.
    """
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    for chunk in metadata:
        chunks_by_document.setdefault(chunk["document_id"], []).append(chunk)

    qrels: dict[str, dict[str, int]] = {}
    for question in questions:
        pages = gold_pages(question)
        relevant_chunks: dict[str, int] = {}

        for chunk in chunks_by_document.get(question["doc_name"], []):
            if chunk_overlaps_gold_page(chunk, pages):
                relevant_chunks[chunk["chunk_id"]] = 1

        qrels[question["financebench_id"]] = relevant_chunks

    return qrels


def build_run_entry(results: list[dict[str, Any]], top_k: int) -> dict[str, float]:
    """Build one pytrec_eval run entry while preserving retrieval rank order."""
    return {
        result["chunk_id"]: float(top_k - int(result["rank"]) + 1)
        for result in results
    }


def first_relevant_rank(
    results: list[dict[str, Any]],
    gold_doc_name: str,
    pages: set[int],
) -> int | None:
    """Find the first rank that matches the gold document and evidence page."""
    for result in results:
        same_doc = result["document_id"] == gold_doc_name
        same_page = result_overlaps_gold_page(result, pages)

        if same_doc and same_page:
            return int(result["rank"])

    return None


def first_doc_rank(results: list[dict[str, Any]], gold_doc_name: str) -> int | None:
    """Find the first rank that matches the gold document."""
    for result in results:
        if result["document_id"] == gold_doc_name:
            return int(result["rank"])
    return None


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    retriever: Any,
    method: str,
    top_k: int,
    bm25_top_k: int,
    dense_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Evaluate retrieval against FinanceBench gold doc/page evidence."""
    rows: list[dict[str, Any]] = []
    run: dict[str, dict[str, float]] = {}

    for record in tqdm(questions, desc="Evaluating retrieval"):
        pages = gold_pages(record)
        results = retriever.retrieve(
            query=record["question"],
            method=method,
            bm25_top_k=max(bm25_top_k, top_k),
            dense_top_k=max(dense_top_k, top_k),
            final_top_k=top_k,
        )

        doc_rank = first_doc_rank(results=results, gold_doc_name=record["doc_name"])
        page_rank = first_relevant_rank(
            results=results,
            gold_doc_name=record["doc_name"],
            pages=pages,
        )

        top_result = results[0] if results else {}
        qid = record["financebench_id"]
        run[qid] = build_run_entry(results, top_k=top_k)
        rows.append(
            {
                "financebench_id": qid,
                "company": record["company"],
                "doc_name": record["doc_name"],
                "question": record["question"],
                "answer": record["answer"],
                "gold_pages": sorted(pages),
                "method": method,
                "top_k": top_k,
                "doc_rank": doc_rank,
                "page_rank": page_rank,
                "doc_hit": doc_rank is not None,
                "page_hit": page_rank is not None,
                "doc_reciprocal_rank": 0.0 if doc_rank is None else 1.0 / doc_rank,
                "page_reciprocal_rank": 0.0 if page_rank is None else 1.0 / page_rank,
                "top_chunk_id": top_result.get("chunk_id"),
                "top_document_id": top_result.get("document_id"),
                "top_page_start": top_result.get("page_start"),
                "top_page_end": top_result.get("page_end"),
                "top_score": top_result.get("score"),
            }
        )

    return rows, run


def summarize(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, float]:
    """Summarize hit rates and MRR from per-question rows."""
    summary: dict[str, float] = {}
    total = len(rows)

    if total == 0:
        return {"questions": 0}

    for k in k_values:
        summary[f"doc_hit_rate_at_{k}"] = mean(
            bool(row["doc_rank"] is not None and row["doc_rank"] <= k)
            for row in rows
        )
        summary[f"page_hit_rate_at_{k}"] = mean(
            bool(row["page_rank"] is not None and row["page_rank"] <= k)
            for row in rows
        )

    summary["doc_mrr"] = mean(float(row["doc_reciprocal_rank"]) for row in rows)
    summary["page_mrr"] = mean(float(row["page_reciprocal_rank"]) for row in rows)
    summary["questions"] = float(total)
    return summary


def evaluate_with_pytrec(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Evaluate retrieval with standard TREC-style metrics."""
    metrics = {
        "recall_1",
        "recall_3",
        "recall_5",
        "recall_10",
        "recip_rank",
        "ndcg_cut_10",
        "map_cut_10",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(run)

    if not per_query:
        return {}

    metric_names = sorted(next(iter(per_query.values())).keys())
    return {
        metric: mean(query_scores.get(metric, 0.0) for query_scores in per_query.values())
        for metric in metric_names
    }


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write detailed evaluation rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, float]) -> None:
    """Print evaluation metrics."""
    print("\nRetrieval evaluation summary:\n")
    for key, value in summary.items():
        if key == "questions":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.3f}")


def print_pytrec_summary(summary: dict[str, float]) -> None:
    """Print pytrec_eval metrics."""
    print("\npytrec_eval summary:\n")
    for key, value in sorted(summary.items()):
        print(f"{key}: {value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on FinanceBench questions.")
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default=None,
        help="Retrieval method. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of FinanceBench questions to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum retrieved chunks to evaluate.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="CSV path for detailed evaluation rows.",
    )
    args = parser.parse_args()

    config = load_config()
    retrieval_config = config["retrieval"]
    evaluation_config = config["evaluation"]

    method = args.method or retrieval_config["method"]
    limit = args.limit or int(evaluation_config["default_limit"])
    k_values = [int(k) for k in evaluation_config["retrieval_k_values"]]
    top_k = args.top_k or max(k_values)
    output_path = project_path(args.output_path or evaluation_config["output_path"])

    questions = load_financebench_questions(
        project_path(config["paths"]["financebench_questions_path"]),
        limit=limit,
    )

    retriever = Retriever(config)
    rows, run = evaluate_retrieval(
        questions=questions,
        retriever=retriever,
        method=method,
        top_k=top_k,
        bm25_top_k=int(retrieval_config["bm25_top_k"]),
        dense_top_k=int(retrieval_config["dense_top_k"]),
    )
    qrels = build_qrels_for_questions(questions, retriever.metadata)
    summary = summarize(rows, k_values=k_values)
    pytrec_summary = evaluate_with_pytrec(qrels, run)
    write_results(rows, output_path)

    print_summary(summary)
    print_pytrec_summary(pytrec_summary)
    print(
        "\nqrels relevant chunks: "
        f"{sum(len(relevant_chunks) for relevant_chunks in qrels.values())}"
    )
    print(f"\nDetailed rows written to: {output_path}")


if __name__ == "__main__":
    main()
