"""
BM25 Index — keyword-based retrieval over Chunks.

BM25 (Best Match 25) is the gold-standard keyword ranking algorithm
used by Elasticsearch and Solr. It beats plain TF-IDF because it
normalises for document length and applies term saturation.

Why we need it alongside vector search:
- Vector search excels at semantic similarity ("what is RAG?")
- BM25 excels at exact-match queries ("RAGAS score", "text-embedding-3-small")
- Hybrid = best of both worlds

Install: pip install rank-bm25 nltk
"""

from __future__ import annotations

import re
import pickle
import string
from pathlib import Path as FilePath
from typing import Optional

from rank_bm25 import BM25Okapi

from core.chunk import Chunk


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, split on whitespace.
    Keeps numbers (e.g. "gpt4", "top-5") intact.
    No stopword removal — BM25's IDF handles common words naturally.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Builds and queries a BM25 keyword index over a list of Chunks.

    The index is built in-memory and can be saved/loaded from disk
    so you don't need to rebuild it on every run.

    Usage:
        index = BM25Index()
        index.build(chunks)

        results = index.search("retrieval augmented generation", top_k=10)
        # returns list of {"chunk_id", "score", "text", "title", ...}

        # Persist to disk
        index.save("bm25_index.pkl")

        # Load later
        index = BM25Index.load("bm25_index.pkl")
    """

    def __init__(self):
        self._bm25:   Optional[BM25Okapi] = None
        self._chunks: list[Chunk]         = []
        self._corpus: list[list[str]]     = []

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of Chunks."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        print(f"Building BM25 index over {len(chunks)} chunks …")
        self._chunks = chunks
        self._corpus = [tokenise(c.text) for c in chunks]
        self._bm25   = BM25Okapi(self._corpus)
        print(f"BM25 index ready. Avg tokens/chunk: "
              f"{sum(len(t) for t in self._corpus) / len(self._corpus):.0f}")

    # ── Search ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return the top_k chunks ranked by BM25 score.

        Returns list of dicts with keys:
            chunk_id, score, rank, text, title, source_uri, doc_id, …
        """
        self._require_built()

        tokens = tokenise(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)  # numpy array, one score per chunk

        # Get top_k indices sorted by descending score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                continue  # BM25 returns 0 for no match — skip non-matches
            chunk = self._chunks[idx]
            results.append({
                **chunk.to_metadata_dict(),
                "text":  chunk.text,
                "score": score,
                "rank":  rank,
            })

        return results

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the index to disk so you don't rebuild on every run."""
        self._require_built()
        data = {
            "bm25":   self._bm25,
            "chunks": self._chunks,
            "corpus": self._corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a previously saved BM25 index from disk."""
        if not FilePath(path).exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index._bm25   = data["bm25"]
        index._chunks = data["chunks"]
        index._corpus = data["corpus"]
        print(f"BM25 index loaded ← {path} ({len(index._chunks)} chunks)")
        return index

    # ── Helpers ────────────────────────────────────────────────────────

    def _require_built(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build(chunks) first.")

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None