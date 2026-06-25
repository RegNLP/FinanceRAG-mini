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
