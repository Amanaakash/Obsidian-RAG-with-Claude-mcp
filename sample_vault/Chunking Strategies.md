---
tags:
  - rag
  - chunking
---

# Chunking Strategies

Chunking divides a large document into smaller searchable units. Chunk boundaries influence whether retrieval returns enough context without adding irrelevant text.

## Heading-aware chunking

Markdown headings already describe a document's semantic structure. A heading-aware chunker keeps each section together and records the heading path as metadata.

## Chunk size

Chunks that are too small may lose context. Chunks that are too large may contain unrelated subjects. The best size should be selected using retrieval evaluation rather than intuition alone.
