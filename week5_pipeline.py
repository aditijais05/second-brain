"""
Week 5 Pipeline — full RAG pipeline with citation-grounded generation.

Combines:
    Week 1-2: Ingestion (PDF, Markdown, URL)
    Week 3:   Chunking + Embeddings + Vector Store
    Week 4:   Hybrid Search (BM25 + Vector + RRF)
    Week 5:   Citation-grounded generation via Claude

Usage:
    from ingestion.pipeline import IngestionPipeline
    from week5_pipeline import Week5Pipeline

    # Ingest your documents
    docs = IngestionPipeline().run([
        "notes/meeting.md",
        "papers/attention.pdf",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ])

    # Build the full pipeline
    pipeline = Week5Pipeline(embedding_backend="local")
    pipeline.index(docs)

    # Ask a question — get a cited answer
    answer = pipeline.ask("What is the difference between RAG and fine-tuning?")
    print(answer.format_answer())
"""

from __future__ import annotations

from typing import Literal

from core.document import Document
from core.chunk import Chunk
from chunking.chunker import SemanticChunker, ChunkerConfig
from embeddings.embedder import get_embedder
from embeddings.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import CitationGenerator, CitedAnswer


class Week5Pipeline:
    """
    Full end-to-end RAG pipeline: ingest → chunk → embed → index → retrieve → generate.

    Args:
        embedding_backend: "local" (free, no key) or "openai" (better quality)
        chunk_size:        tokens per chunk (default 400)
        chunk_overlap:     overlap between chunks (default 80)
        store_mode:        "memory" | "local" | "server"
        gemini_api_key: your Anthropic API key (or set GEMINI_API_KEY env var)
        top_k:             number of chunks to retrieve per query (default 5)
    """

    def __init__(
        self,
        embedding_backend:  Literal["local", "openai"] = "local",
        chunk_size:         int = 400,
        chunk_overlap:      int = 80,
        store_mode:         Literal["memory", "local", "server"] = "memory",
        store_path:         str = "./qdrant_data",
        gemini_api_key:  str | None = None,
        top_k:              int = 5,
    ):
        print("Initialising Second Brain pipeline…\n")
        self.top_k     = top_k
        self.embedder  = get_embedder(embedding_backend)
        self.chunker   = SemanticChunker(
            ChunkerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        self.vector_store = VectorStore(
            mode=store_mode, dimension=self.embedder.dimension, path=store_path
        )
        self.bm25_index   = BM25Index()
        self.generator    = CitationGenerator(api_key=gemini_api_key)
        self.retriever:   HybridRetriever | None = None
        self._chunks:     list[Chunk] = []

    # ── Indexing ───────────────────────────────────────────────────────

    def index(self, docs: list[Document]) -> None:
        """Full indexing pipeline: chunk → embed → vector store + BM25."""
        print("── Step 1: Chunking ────────────────────────────────")
        self._chunks = self.chunker.chunk_many(docs)

        print("\n── Step 2: Embedding ───────────────────────────────")
        self._chunks = self.embedder.embed_chunks(self._chunks)

        print("\n── Step 3: Vector store ────────────────────────────")
        self.vector_store.upsert(self._chunks)

        print("\n── Step 4: BM25 index ──────────────────────────────")
        self.bm25_index.build(self._chunks)

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
        )
        print(f"\nReady. {self.vector_store.count()} chunks indexed.\n")

    # ── Ask ────────────────────────────────────────────────────────────

    def ask(
        self,
        query:      str,
        top_k:      int  | None = None,
        explain:    bool = False,
        max_tokens: int  = 1024,
    ) -> CitedAnswer:
        """
        Ask a question. Retrieve relevant chunks, generate a cited answer.

        Args:
            query:      Your question in natural language.
            top_k:      Override default number of chunks to retrieve.
            explain:    Print RRF score breakdown for this query.
            max_tokens: Max tokens in Claude's answer.

        Returns:
            CitedAnswer with answer text and grounded source citations.
        """
        if self.retriever is None:
            raise RuntimeError("Call index(docs) before ask().")

        k = top_k or self.top_k

        print(f'Retrieving context for: "{query}"')
        chunks = self.retriever.search(query, top_k=k, explain=explain)
        print(f"Retrieved {len(chunks)} chunks. Generating answer…\n")

        return self.generator.generate(
            query=query,
            chunks=chunks,
            max_tokens=max_tokens,
        )