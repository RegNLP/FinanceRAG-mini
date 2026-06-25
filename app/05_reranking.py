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
