---
tags: [rag, evaluation]
---

# RAG Evaluation

Evaluation should separately examine retrieval and answer generation. A fluent answer can hide the fact that the correct source was never retrieved.

## Retrieval metrics

Recall at k checks whether a relevant document appears within the first k results. Mean Reciprocal Rank rewards systems that place the first relevant result near the top.

## Small evaluation set

A personal project can begin with twenty manually written questions and their expected source notes. This makes chunking and retrieval experiments measurable.
