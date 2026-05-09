"""
Core Document model — the unified data structure that flows through
every stage of the Second Brain pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import hashlib
import json


class SourceType(str, Enum):
    PDF      = "pdf"
    MARKDOWN = "markdown"
    URL      = "url"
    NOTION   = "notion"
    KINDLE   = "kindle"


@dataclass
class Document:
    """
    A single ingested document before chunking.

    Every ingestor must produce a list of Documents.
    Downstream stages (chunker, embedder, store) only ever see Documents.
    """

    # ── Content ──────────────────────────────────────────────────────────
    content: str                          # Raw extracted text
    title:   str                          # Human-readable title

    # ── Provenance ───────────────────────────────────────────────────────
    source_type: SourceType
    source_uri:  str                      # File path, URL, Notion page id, …
    created_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Optional metadata ────────────────────────────────────────────────
    author:   Optional[str]  = None
    tags:     list[str]      = field(default_factory=list)
    language: str            = "en"
    extra:    dict           = field(default_factory=dict)  # source-specific extras

    # ── Computed ─────────────────────────────────────────────────────────
    doc_id: str = field(init=False)

    def __post_init__(self):
        # Stable ID: hash of (source_uri + title) so re-ingesting same file
        # gives the same ID → easy deduplication later.
        payload = f"{self.source_uri}::{self.title}"
        self.doc_id = hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ── Helpers ───────────────────────────────────────────────────────────
    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_metadata_dict(self) -> dict:
        """Return a flat dict safe to store alongside a vector embedding."""
        return {
            "doc_id":      self.doc_id,
            "title":       self.title,
            "source_type": self.source_type.value,
            "source_uri":  self.source_uri,
            "author":      self.author or "",
            "tags":        json.dumps(self.tags),
            "language":    self.language,
            "created_at":  self.created_at.isoformat(),
            "word_count":  self.word_count,
        }

    def __repr__(self) -> str:
        return (
            f"Document(id={self.doc_id!r}, title={self.title!r}, "
            f"source={self.source_type.value}, words={self.word_count})"
        )