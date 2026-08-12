---
title: RAG Fundamentals
tags:
  - rag
  - llm
---

# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, retrieves relevant external information before asking a language model to produce an answer. The retrieved passages ground the response in a chosen knowledge source.

## Main pipeline

A basic RAG pipeline loads documents, splits them into chunks, generates embeddings, stores vectors, retrieves relevant chunks, and places those chunks in the language model prompt.

## Why retrieval matters

Retrieval lets an application use private or recently updated knowledge that was not part of the model's original training data. Poor retrieval still produces poor answers, even when the language model is strong.

See also [[Embeddings]] and [[Chunking Strategies]].
