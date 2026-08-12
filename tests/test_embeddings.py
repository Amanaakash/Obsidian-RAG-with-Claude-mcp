import math

from obsidian_rag.embeddings import OllamaEmbedder


class LengthLimitedClient:
    """Pretend Ollama rejects any individual input above 20 characters."""

    def embed(self, model: str, input: list[str], truncate: bool) -> dict[str, list[list[float]]]:
        if any(len(text) > 20 for text in input):
            raise RuntimeError("the input length exceeds the context length (status code: 400)")
        return {
            "embeddings": [
                [float(len(text)), float(text.count("rag") + 1)]
                for text in input
            ]
        }


def test_long_input_is_split_and_combined_without_truncation() -> None:
    embedder = OllamaEmbedder()
    embedder.client = LengthLimitedClient()

    vector = embedder.embed_documents(
        ["rag retrieval context that is deliberately longer than twenty characters"]
    )[0]

    assert len(vector) == 2
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
