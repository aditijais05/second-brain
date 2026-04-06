"""
Week 4 Pipeline — full hybrid search pipeline.

Builds on Week 3 by replacing the pure vector retriever with a
hybrid retriever (vector + BM25 + RRF fusion).

Usage:
    from ingestion.pipeline import IngestionPipeline
    from week4_pipeline import Week4Pipeline

    docs = IngestionPipeline().run(["notes/", "paper.pdf"])

    pipeline = Week4Pipeline(embedding_backend="local")
    pipeline.index(docs)

    results = pipeline.search("what is retrieval augmented generation?", top_k=5)
    pipeline.print_results(results)

    # See RRF score breakdown
    results = pipeline.search("RAGAS metrics", top_k=5, explain=True)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Literal

from core.document import Document
from core.chunk import Chunk
from chunking.chunker import SemanticChunker, ChunkerConfig
from embeddings.embedder import get_embedder, BaseEmbedder
from embeddings.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever


class Week4Pipeline:
    """
    Full hybrid RAG pipeline: ingest → chunk → embed → index → search.

    Args:
        embedding_backend: "local" (free) or "openai" (better quality)
        chunk_size:        tokens per chunk (default 400)
        chunk_overlap:     overlap tokens (default 80)
        store_mode:        "memory" | "local" | "server"
    """

    def __init__(
        self,
        embedding_backend: Literal["local", "openai"] = "local",
        chunk_size:    int = 400,
        chunk_overlap: int = 80,
        store_mode:    Literal["memory", "local", "server"] = "memory",
        store_path:    str = "./qdrant_data",
    ):
        print(f"Initialising Week4Pipeline (embedder={embedding_backend}, store={store_mode})\n")
        self.embedder  = get_embedder(embedding_backend)
        self.chunker   = SemanticChunker(ChunkerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        self.vector_store = VectorStore(mode=store_mode, dimension=self.embedder.dimension, path=store_path)
        self.bm25_index   = BM25Index()
        self.retriever: HybridRetriever | None = None
        self._chunks:   list[Chunk] = []

    # ── Indexing ───────────────────────────────────────────────────────

    def index(self, docs: list[Document]) -> None:
        """Chunk → embed → store in vector DB + BM25 index."""

        print("── Step 1: Chunking ────────────────────────────────")
        self._chunks = self.chunker.chunk_many(docs)

        print("\n── Step 2: Embedding ───────────────────────────────")
        self._chunks = self.embedder.embed_chunks(self._chunks)

        print("\n── Step 3: Vector store ────────────────────────────")
        self.vector_store.upsert(self._chunks)

        print("\n── Step 4: BM25 index ──────────────────────────────")
        self.bm25_index.build(self._chunks)

        print("\n── Step 5: Building hybrid retriever ───────────────")
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
        )
        print(f"Ready. {self.vector_store.count()} chunks indexed.\n")

    # ── Search ─────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, explain: bool = False) -> list[dict]:
        """Hybrid search over all indexed chunks."""
        if self.retriever is None:
            raise RuntimeError("Call index(docs) before search().")
        return self.retriever.search(query, top_k=top_k, explain=explain)

    # ── Display ────────────────────────────────────────────────────────

    @staticmethod
    def print_results(results: list[dict], query: str = "") -> None:
        """Pretty-print search results to console."""
        if query:
            print(f'\nQuery: "{query}"')
        print(f"{'─' * 60}")
        for r in results:
            print(f"  #{r['rrf_rank']}  [{r['rrf_score']:.5f}]  {r['title']}")
            print(f"       {r['text'][:120].strip()}…")
            print(f"       source: {r['source_uri']}")
            print()