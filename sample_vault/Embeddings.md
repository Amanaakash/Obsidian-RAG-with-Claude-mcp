---
tags: [rag, embeddings]
---

# Embeddings

An embedding is a numeric vector representing aspects of the meaning of text. Semantically related passages tend to be closer together in the embedding space.

## Similarity search

At search time, the user's question is embedded with the same model used for the documents. A vector database compares the query vector with stored vectors and returns nearby chunks.

## Consistency rule

The document index and query must use the same embedding model. Switching models requires rebuilding the vector index because dimensions and vector spaces may differ.
