"""
Chunk — the unit of retrieval.

A Document gets split into many Chunks. Each Chunk carries enough
metadata to reconstruct its provenance (which doc, which position)
and to be stored in a vector database.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class Chunk:
    """
    A single retrievable piece of a Document.

    Attributes:
        text        — the raw text for this chunk
        doc_id      — ID of the parent Document
        chunk_index — position within the document (0-based)
        title       — parent document title (copied for convenience)
        source_uri  — parent document source path/URL
        source_type — pdf / markdown / url / …
        start_word  — approximate word offset in original document
        embedding   — filled in by the Embedder (None until then)
        metadata    — any extra key-value pairs (page number, tags, …)
    """

    text:        str
    doc_id:      str
    chunk_index: int
    title:       str
    source_uri:  str
    source_type: str

    start_word:  int           = 0
    embedding:   Optional[list[float]] = field(default=None, repr=False)
    metadata:    dict          = field(default_factory=dict)

    # Computed
    chunk_id: str = field(init=False)

    def __post_init__(self):
        payload = f"{self.doc_id}::{self.chunk_index}"
        self.chunk_id = hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_metadata_dict(self) -> dict:
        """Flat dict safe to store alongside a vector embedding in Qdrant."""
        return {
            "chunk_id":    self.chunk_id,
            "doc_id":      self.doc_id,
            "chunk_index": self.chunk_index,
            "title":       self.title,
            "source_uri":  self.source_uri,
            "source_type": self.source_type,
            "start_word":  self.start_word,
            "word_count":  self.word_count,
            **self.metadata,
        }

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id!r}, doc={self.doc_id!r}, "
            f"idx={self.chunk_index}, words={self.word_count}, "
            f"text={preview!r}…)"
        )