"""
Embedder — converts Chunks into vector embeddings.

Supports two backends:
  1. OpenAI  — text-embedding-3-small  (best quality, needs API key)
  2. Local   — sentence-transformers   (free, runs on CPU, good quality)

The embedder batches requests automatically to stay within API rate limits
and avoid sending 1000 individual API calls.

Install:
    pip install openai                      # for OpenAI backend
    pip install sentence-transformers       # for local backend
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Literal

from core.chunk import Chunk


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Fill in `chunk.embedding` for every chunk.
        Returns the same list with embeddings attached.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Size of the embedding vector produced by this model."""
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's text-embedding-3-small model.

    Cost: ~$0.02 per million tokens — essentially free for personal use.
    Quality: excellent for RAG retrieval tasks.

    Setup:
        export OPENAI_API_KEY=sk-...        # Mac/Linux
        $env:OPENAI_API_KEY = "sk-..."      # Windows PowerShell

    Usage:
        embedder = OpenAIEmbedder()
        chunks   = embedder.embed_chunks(chunks)
    """

    MODEL     = "text-embedding-3-small"
    DIMENSION = 1536
    BATCH_SIZE = 100   # OpenAI allows up to 2048 inputs per request

    def __init__(self, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        print(f"Embedding {len(chunks)} chunks with {self.MODEL} …")
        batches = self._batch(chunks, self.BATCH_SIZE)

        for batch_num, batch in enumerate(batches, 1):
            texts = [c.text for c in batch]
            print(f"  batch {batch_num}/{len(batches)} ({len(texts)} chunks) …", end=" ")

            response = self._client.embeddings.create(
                model=self.MODEL,
                input=texts,
            )

            for chunk, emb_obj in zip(batch, response.data):
                chunk.embedding = emb_obj.embedding

            print("✓")

            # Avoid rate-limit on large ingestions
            if batch_num < len(batches):
                time.sleep(0.5)

        return chunks

    @staticmethod
    def _batch(items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """
    Runs a sentence-transformers model locally on CPU (or GPU if available).

    No API key needed. Good for development and offline use.

    Recommended model: "all-MiniLM-L6-v2"
      - 384-dimensional embeddings
      - ~80MB download on first run
      - Fast on CPU (~500 chunks/min)

    Usage:
        embedder = LocalEmbedder()                          # default model
        embedder = LocalEmbedder("BAAI/bge-small-en-v1.5") # better quality
        chunks   = embedder.embed_chunks(chunks)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    BATCH_SIZE = 64

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        print(f"Loading local model '{model_name}' (downloads on first run) …")
        self._model      = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim        = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        print(f"Embedding {len(chunks)} chunks locally with '{self._model_name}' …")
        texts = [c.text for c in chunks]

        # encode() handles batching internally
        embeddings = self._model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        print(f"Done. Embedding dimension: {self._dim}")
        return chunks


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_embedder(
    backend: Literal["openai", "local"] = "local",
    **kwargs,
) -> BaseEmbedder:
    """
    Convenience factory.

    Usage:
        embedder = get_embedder("openai")   # needs OPENAI_API_KEY env var
        embedder = get_embedder("local")    # free, runs on your machine
    """
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'openai' or 'local'.")