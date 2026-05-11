"""
Embedder — converts Chunks into vector embeddings.

Supports two backends:
  1. OpenAI    — text-embedding-3-small  (best quality, needs API key)
  2. Local     — fastembed (ONNX-based, free, no torch, ~50MB, runs on CPU)

Switched from sentence-transformers to fastembed for production deployment
because sentence-transformers pulls in torch (~400MB) which exceeds the
512MB memory limit on Render's free tier. fastembed uses ONNX runtime
instead — same quality, fraction of the memory.

Install:
    pip install openai       # for OpenAI backend
    pip install fastembed    # for local backend
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

    MODEL      = "text-embedding-3-small"
    DIMENSION  = 1536
    BATCH_SIZE = 100

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

            if batch_num < len(batches):
                time.sleep(0.5)

        return chunks

    @staticmethod
    def _batch(items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Local backend (fastembed — ONNX, no torch)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """
    Runs embeddings locally using fastembed (ONNX runtime, no torch).

    Why fastembed instead of sentence-transformers?
    - sentence-transformers imports torch (~400MB RAM) — too heavy for
      free cloud tiers (Render free = 512MB limit)
    - fastembed uses ONNX runtime (~50MB RAM) with similar quality

    Default model: "BAAI/bge-small-en-v1.5"
      - 384-dimensional embeddings
      - ~130MB download on first run (cached after that)
      - Fast on CPU

    Usage:
        embedder = LocalEmbedder()
        chunks   = embedder.embed_chunks(chunks)
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    BATCH_SIZE    = 32

    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("Run: pip install fastembed")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        print(f"Loading fastembed model '{model_name}' (downloads on first run) …")
        self._model      = TextEmbedding(model_name=model_name)
        self._model_name = model_name

        # Get dimension by embedding a test string
        test = list(self._model.embed(["test"]))
        self._dim = len(test[0])
        print(f"Model ready. Embedding dimension: {self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        print(f"Embedding {len(chunks)} chunks with '{self._model_name}' …")
        texts = [c.text for c in chunks]

        # fastembed.embed() returns a generator — convert to list
        embeddings = list(self._model.embed(texts))

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        print(f"Done. {len(chunks)} chunks embedded.")
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
        embedder = get_embedder("local")    # free, ONNX-based, no torch
    """
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'openai' or 'local'.")