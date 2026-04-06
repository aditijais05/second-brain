"""
Tests for the ingestion pipeline.

Run: pytest tests/test_ingestion.py -v
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile, os

from core.document import Document, SourceType
from ingestion.markdown_ingestor import MarkdownIngestor
from ingestion.pipeline import IngestionPipeline


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_md_file(tmp_path):
    """Create a temp markdown file with frontmatter."""
    content = """---
title: "Test Note"
author: "Alice"
date: 2024-03-01
tags: [rag, testing]
---

# Introduction

This is a test document about RAG systems.
It has multiple paragraphs so we can test chunking later.

## Details

More content here about vector databases and embeddings.
"""
    f = tmp_path / "test_note.md"
    f.write_text(content)
    return f


@pytest.fixture
def tmp_md_no_frontmatter(tmp_path):
    f = tmp_path / "plain.md"
    f.write_text("# Just a plain note\n\nNo frontmatter here.")
    return f


# ── Document model tests ───────────────────────────────────────────────────

def test_document_id_is_stable():
    d1 = Document(content="hello", title="T", source_type=SourceType.MARKDOWN, source_uri="/a/b.md")
    d2 = Document(content="hello", title="T", source_type=SourceType.MARKDOWN, source_uri="/a/b.md")
    assert d1.doc_id == d2.doc_id, "Same source_uri+title must produce same doc_id"


def test_document_id_differs_for_different_sources():
    d1 = Document(content="x", title="T", source_type=SourceType.MARKDOWN, source_uri="/a.md")
    d2 = Document(content="x", title="T", source_type=SourceType.MARKDOWN, source_uri="/b.md")
    assert d1.doc_id != d2.doc_id


def test_metadata_dict_keys():
    d = Document(content="hello world", title="T", source_type=SourceType.PDF, source_uri="/doc.pdf")
    meta = d.to_metadata_dict()
    required_keys = {"doc_id", "title", "source_type", "source_uri", "word_count", "created_at"}
    assert required_keys.issubset(meta.keys())


def test_word_count():
    d = Document(content="one two three four five", title="T",
                 source_type=SourceType.MARKDOWN, source_uri="/x.md")
    assert d.word_count == 5


# ── Markdown ingestor tests ────────────────────────────────────────────────

def test_markdown_ingestor_basic(tmp_md_file):
    ingestor = MarkdownIngestor()
    docs = ingestor.ingest(tmp_md_file)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.title   == "Test Note"
    assert doc.author  == "Alice"
    assert "rag"       in doc.tags
    assert "testing"   in doc.tags
    assert doc.source_type == SourceType.MARKDOWN
    assert "RAG systems" in doc.content


def test_markdown_ingestor_no_frontmatter(tmp_md_no_frontmatter):
    docs = MarkdownIngestor().ingest(tmp_md_no_frontmatter)
    assert len(docs) == 1
    assert docs[0].title == "plain"   # falls back to filename stem
    assert docs[0].tags  == []


def test_markdown_ingestor_folder(tmp_path):
    for i in range(3):
        (tmp_path / f"note_{i}.md").write_text(f"# Note {i}\n\nContent {i}")
    docs = MarkdownIngestor().ingest_folder(tmp_path)
    assert len(docs) == 3


def test_markdown_ingestor_empty_file(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("")
    docs = MarkdownIngestor().ingest(f)
    assert docs == []


def test_markdown_ingestor_invalid_path():
    with pytest.raises(FileNotFoundError):
        MarkdownIngestor().ingest("/does/not/exist.md")


# ── Pipeline deduplication tests ──────────────────────────────────────────

def test_pipeline_deduplication(tmp_md_file):
    pipeline = IngestionPipeline()
    # Ingest the same file twice
    docs = pipeline.run([tmp_md_file, tmp_md_file])
    assert len(docs) == 1, "Duplicate source should be deduped"


def test_pipeline_reset_seen(tmp_md_file):
    pipeline = IngestionPipeline()
    docs1 = pipeline.run([tmp_md_file])
    pipeline.reset_seen()
    docs2 = pipeline.run([tmp_md_file])
    assert len(docs1) == len(docs2) == 1


def test_pipeline_unknown_extension(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c")
    pipeline = IngestionPipeline()
    # Should log error and return 0 docs, not raise
    docs = pipeline.run([f])
    assert docs == []