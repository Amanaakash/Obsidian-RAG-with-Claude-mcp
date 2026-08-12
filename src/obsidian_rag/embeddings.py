"""Embedding adapters.

The rest of the project depends on this small interface rather than depending
directly on Ollama. That makes retrieval code easier to test and allows a
different embedding provider to be added later without rewriting the indexer.
"""

from __future__ import annotations

import math
from typing import Protocol

import ollama


DEFAULT_EMBEDDING_MODEL = "all-minilm:22m"


class Embedder(Protocol):
    """The two embedding operations needed by this project."""

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed note chunks before storing them."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a user's search query."""


class OllamaEmbedder:
    """Generate embeddings using an Ollama server running on this computer."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    @staticmethod
    def _response_vectors(response: object) -> list[list[float]]:
        if hasattr(response, "embeddings"):
            return [list(vector) for vector in response.embeddings]
        return [list(vector) for vector in response["embeddings"]]

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        return "input length exceeds the context length" in str(error).casefold()

    @staticmethod
    def _split_text(text: str) -> tuple[str, str]:
        """Split near the middle, preferring whitespace when available."""

        midpoint = len(text) // 2
        left_space = text.rfind(" ", 0, midpoint)
        right_space = text.find(" ", midpoint)
        candidates = [position for position in (left_space, right_space) if position > 0]
        split_at = min(candidates, key=lambda position: abs(position - midpoint)) if candidates else midpoint
        return text[:split_at].strip(), text[split_at:].strip()

    @staticmethod
    def _weighted_mean(
        left: list[float],
        right: list[float],
        left_weight: int,
        right_weight: int,
    ) -> list[float]:
        """Combine sub-chunk embeddings and normalize the resulting vector."""

        total_weight = left_weight + right_weight
        combined = [
            ((left_value * left_weight) + (right_value * right_weight)) / total_weight
            for left_value, right_value in zip(left, right, strict=True)
        ]
        magnitude = math.sqrt(sum(value * value for value in combined)) or 1.0
        return [value / magnitude for value in combined]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = self.client.embed(model=self.model, input=texts, truncate=False)
            return self._response_vectors(response)
        except Exception as error:
            if not self._is_context_length_error(error):
                raise

            # A batch failure does not identify which input was too long.
            # Divide the batch until each document can be handled separately.
            if len(texts) > 1:
                midpoint = len(texts) // 2
                return self._embed(texts[:midpoint]) + self._embed(texts[midpoint:])

            text = texts[0]
            if len(text) < 2:
                raise

            left_text, right_text = self._split_text(text)
            if not left_text or not right_text:
                raise

            left_vector = self._embed([left_text])[0]
            right_vector = self._embed([right_text])[0]
            return [
                self._weighted_mean(
                    left_vector,
                    right_vector,
                    left_weight=len(left_text),
                    right_weight=len(right_text),
                )
            ]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._embed(documents)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]
