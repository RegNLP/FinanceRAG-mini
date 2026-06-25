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
