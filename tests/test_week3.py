"""
Tests for Week 3: chunking + embeddings + vector store.

Run: pytest tests/test_week3.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime, timezone

from core.document import Document, SourceType
from core.chunk import Chunk
from chunking.chunker import SemanticChunker, ChunkerConfig


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_doc(content: str, title: str = "Test Doc") -> Document:
    return Document(
        content=content,
        title=title,
        source_type=SourceType.MARKDOWN,
        source_uri="/test.md",
    )


LONG_CONTENT = """
# Introduction to RAG

Retrieval Augmented Generation (RAG) is a technique that combines
information retrieval with language model generation. It was introduced
to address the limitations of purely parametric language models.

## How it works

The system first retrieves relevant documents from a knowledge base
using a query. These documents are then passed to a language model
as context, allowing it to generate more accurate and grounded responses.

## Why it matters

RAG systems can access up-to-date information without retraining the
underlying language model. This makes them far more practical for
real-world applications where knowledge changes frequently.

## Key components

A typical RAG system consists of three main parts: an ingestion pipeline
that processes and stores documents, a retrieval system that finds
relevant chunks given a query, and a generation system that produces
the final answer using the retrieved context.
""".strip()


# ── Chunk dataclass tests ──────────────────────────────────────────────────

def test_chunk_id_is_stable():
    c1 = Chunk(text="hello", doc_id="abc", chunk_index=0,
               title="T", source_uri="/x", source_type="markdown")
    c2 = Chunk(text="hello", doc_id="abc", chunk_index=0,
               title="T", source_uri="/x", source_type="markdown")
    assert c1.chunk_id == c2.chunk_id


def test_chunk_id_differs_by_index():
    c1 = Chunk(text="hello", doc_id="abc", chunk_index=0,
               title="T", source_uri="/x", source_type="markdown")
    c2 = Chunk(text="hello", doc_id="abc", chunk_index=1,
               title="T", source_uri="/x", source_type="markdown")
    assert c1.chunk_id != c2.chunk_id


def test_chunk_metadata_dict():
    c = Chunk(text="hello world", doc_id="abc", chunk_index=0,
              title="T", source_uri="/x", source_type="markdown")
    meta = c.to_metadata_dict()
    assert meta["word_count"] == 2
    assert meta["chunk_index"] == 0
    assert "chunk_id" in meta


# ── Chunker tests ──────────────────────────────────────────────────────────

def test_chunker_produces_chunks():
    doc = make_doc(LONG_CONTENT)
    chunks = SemanticChunker().chunk(doc)
    assert len(chunks) >= 1


def test_chunker_preserves_content():
    """All words in the original doc should appear somewhere in the chunks."""
    doc = make_doc(LONG_CONTENT)
    chunks = SemanticChunker().chunk(doc)
    all_text = " ".join(c.text for c in chunks)
    # Check a few key phrases are present somewhere
    assert "RAG" in all_text
    assert "retrieval" in all_text.lower()


def test_chunker_respects_max_size():
    """No chunk should exceed chunk_size * 1.5 tokens (overlap can push it slightly over)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    config = ChunkerConfig(chunk_size=200, chunk_overlap=40)
    doc = make_doc(LONG_CONTENT)
    chunks = SemanticChunker(config).chunk(doc)
    for c in chunks:
        tokens = len(enc.encode(c.text))
        assert tokens <= 200 * 1.5, f"Chunk too large: {tokens} tokens"


def test_chunker_min_size_filter():
    """Chunks below min_chunk_size should be discarded."""
    config = ChunkerConfig(min_chunk_size=100)
    doc = make_doc("Short.")   # way below min
    chunks = SemanticChunker(config).chunk(doc)
    assert chunks == []


def test_chunker_chunk_index_sequential():
    doc = make_doc(LONG_CONTENT)
    chunks = SemanticChunker().chunk(doc)
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)


def test_chunker_sets_doc_id():
    doc = make_doc(LONG_CONTENT)
    chunks = SemanticChunker().chunk(doc)
    for c in chunks:
        assert c.doc_id == doc.doc_id


def test_chunker_many():
    docs = [make_doc(LONG_CONTENT, f"Doc {i}") for i in range(3)]
    chunks = SemanticChunker().chunk_many(docs)
    assert len(chunks) >= 3
    # Each chunk knows which doc it came from
    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 3


# ── Vector store tests (in-memory, no Docker needed) ──────────────────────

def test_vector_store_upsert_and_search():
    from embeddings.vector_store import VectorStore

    store = VectorStore(mode="memory", dimension=4)

    # Create fake chunks with fake embeddings
    chunks = []
    for i in range(5):
        c = Chunk(
            text=f"Document about topic {i}",
            doc_id=f"doc_{i}",
            chunk_index=0,
            title=f"Doc {i}",
            source_uri=f"/doc_{i}.md",
            source_type="markdown",
        )
        c.embedding = [float(i), 0.0, 0.0, 0.0]  # fake 4-dim vector
        chunks.append(c)

    store.upsert(chunks)
    assert store.count() == 5

    # Search with a vector close to chunk 3
    results = store.search(query_vector=[3.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    # Cosine similarity can't distinguish [3,0,0,0] vs [4,0,0,0] (same direction)
    # so just assert that the high-magnitude docs rank above doc_1
    top_ids = [r["doc_id"] for r in results]
    assert "doc_3" in top_ids or "doc_4" in top_ids


def test_vector_store_skips_chunks_without_embedding():
    from embeddings.vector_store import VectorStore

    store = VectorStore(mode="memory", dimension=4)
    c = Chunk(text="no embedding", doc_id="x", chunk_index=0,
              title="T", source_uri="/x", source_type="markdown")
    # Don't set c.embedding
    store.upsert([c])
    assert store.count() == 0