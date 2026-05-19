"""
Embedder — converts Chunks into vector embeddings.

Supports three backends:
  1. Gemini  — text-embedding-004  (default; API-based, free, zero RAM overhead)
  2. OpenAI  — text-embedding-3-small  (best quality, needs OPENAI_API_KEY)
  3. Local   — fastembed (ONNX-based, no torch — for local dev only)

Why Gemini as default?
  sentence-transformers pulls in torch (~400MB RAM) — too heavy for Render's
  free tier (512MB limit). fastembed requires py-rust-stemmers which needs
  Rust to compile, and Render's build filesystem is read-only so it fails.
  Gemini embeddings are API-based: zero RAM overhead, completely free, and
  use the same GEMINI_API_KEY already set in Render's environment.

Install:
    pip install google-genai>=1.0   # for Gemini backend (default)
    pip install openai               # for OpenAI backend
    pip install fastembed            # for local backend (local dev only)
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
# Gemini backend (default — API-based, free, zero RAM)
# ---------------------------------------------------------------------------

class GeminiEmbedder(BaseEmbedder):
    """
    Uses Google's text-embedding-004 model via the Gemini API.

    Why this is the default for production:
    - Zero RAM overhead (API call, no local model loaded)
    - Completely free under Gemini's generous free tier
    - Uses GEMINI_API_KEY already set in Render environment variables
    - 768-dimensional embeddings — excellent for RAG retrieval

    Setup:
        export GEMINI_API_KEY=...       # Mac/Linux
        $env:GEMINI_API_KEY = "..."     # Windows PowerShell

    Usage:
        embedder = GeminiEmbedder()
        chunks   = embedder.embed_chunks(chunks)
    """

    MODEL      = "text-embedding-004"
    DIMENSION  = 768
    BATCH_SIZE = 100   # Gemini supports up to 100 texts per batch

    def __init__(self, api_key: str | None = None):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Run: pip install google-genai>=1.0")

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Export it or pass api_key= to GeminiEmbedder()."
            )
        self._client = genai.Client(api_key=resolved_key)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        print(f"Embedding {len(chunks)} chunks with Gemini {self.MODEL} …")
        batches = self._batch(chunks, self.BATCH_SIZE)

        for batch_num, batch in enumerate(batches, 1):
            texts = [c.text for c in batch]
            print(f"  batch {batch_num}/{len(batches)} ({len(texts)} chunks) …", end=" ")

            result = self._client.models.embed_content(
                model=self.MODEL,
                contents=texts,
            )

            for chunk, emb in zip(batch, result.embeddings):
                chunk.embedding = emb.values

            print("✓")

            # Gentle rate-limit buffer between batches
            if batch_num < len(batches):
                time.sleep(0.3)

        return chunks

    @staticmethod
    def _batch(items: list, size: int) -> list[list]:
        return [items[i : i + size] for i in range(0, len(items), size)]


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
# Local backend (fastembed — ONNX, no torch) — local dev only
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """
    Runs embeddings locally using fastembed (ONNX runtime, no torch).

    NOTE: Not suitable for Render deployment — fastembed requires
    py-rust-stemmers which needs Rust to compile, and Render's build
    filesystem is read-only. Use GeminiEmbedder for production.

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

        embeddings = list(self._model.embed(texts))

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        print(f"Done. {len(chunks)} chunks embedded.")
        return chunks


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_embedder(
    backend: Literal["gemini", "openai", "local"] = "gemini",
    **kwargs,
) -> BaseEmbedder:
    """
    Convenience factory. Defaults to Gemini for zero-RAM production use.

    Usage:
        embedder = get_embedder()            # Gemini (default, production)
        embedder = get_embedder("openai")    # needs OPENAI_API_KEY
        embedder = get_embedder("local")     # local dev only, needs fastembed
    """
    if backend == "gemini":
        return GeminiEmbedder(**kwargs)
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'gemini', 'openai', or 'local'.")