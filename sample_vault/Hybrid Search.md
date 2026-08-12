---
tags: [rag, search]
---

# Hybrid Search

Hybrid search combines semantic vector retrieval with lexical keyword retrieval. This is useful because vector search understands related meanings while keyword search handles exact terminology, identifiers, and uncommon names.

## Rank fusion

Reciprocal Rank Fusion combines multiple ranked result lists using their rank positions rather than directly comparing incompatible raw scores.

Hybrid search is a later phase for this project. Phase 1 intentionally begins with dense vector retrieval only.
