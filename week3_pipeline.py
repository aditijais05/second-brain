"""
Week 3 Pipeline — chunks documents, embeds them, and stores in Qdrant.

This is the glue that connects ingestion → chunking → embedding → storage.

Usage:
    from ingestion.pipeline import IngestionPipeline
    from week3_pipeline import Week3Pipeline

    # Ingest
    docs = IngestionPipeline().run(["notes/my_note.md", "paper.pdf"])

    # Chunk + embed + store  (local embedder, no API key needed)
    pipeline = Week3Pipeline(embedding_backend="local")
    pipeline.run(docs)

    # Query
    results = pipeline.query("what is retrieval augmented generation?", top_k=5)
    for r in results:
        print(r["score"], r["title"], r["text"][:100])
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


class Week3Pipeline:
    """
    End-to-end: Documents → Chunks → Embeddings → Qdrant.

    Args:
        embedding_backend: "local" (free, no key) or "openai" (better quality)
        chunk_size:        target chunk size in tokens (default 400)
        chunk_overlap:     overlap between chunks in tokens (default 80)
        store_mode:        "memory" | "local" | "server"
    """

    def __init__(
        self,
        embedding_backend: Literal["local", "openai"] = "local",
        chunk_size:   int = 400,
        chunk_overlap: int = 80,
        store_mode:   Literal["memory", "local", "server"] = "memory",
        store_path:   str = "./qdrant_data",
    ):
        print(f"Initialising Week3Pipeline (embedder={embedding_backend}, store={store_mode}) …\n")

        self.embedder = get_embedder(embedding_backend)
        self.chunker  = SemanticChunker(
            ChunkerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        self.store = VectorStore(
            mode=store_mode,
            dimension=self.embedder.dimension,
            path=store_path,
        )

    def run(self, docs: list[Document]) -> list[Chunk]:
        """Full pipeline: chunk → embed → store. Returns all chunks."""

        # 1. Chunk
        print("── Step 1: Chunking ────────────────────────────────")
        chunks = self.chunker.chunk_many(docs)

        # 2. Embed
        print("\n── Step 2: Embedding ───────────────────────────────")
        chunks = self.embedder.embed_chunks(chunks)

        # 3. Store
        print("\n── Step 3: Storing in Qdrant ───────────────────────")
        self.store.upsert(chunks)
        print(f"   Total stored: {self.store.count()} chunks")

        return chunks

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Embed a question and return the top_k most relevant chunks.
        Each result dict contains: text, title, source_uri, score, …
        """
        print(f'\nQuerying: "{question}"')
        # Embed the query using the same model
        query_chunk = Chunk(
            text=question,
            doc_id="query",
            chunk_index=0,
            title="query",
            source_uri="query",
            source_type="query",
        )
        [query_chunk] = self.embedder.embed_chunks([query_chunk])

        results = self.store.search(
            query_vector=query_chunk.embedding,
            top_k=top_k,
        )

        print(f"Found {len(results)} results\n")
        return results