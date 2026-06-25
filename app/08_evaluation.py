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
