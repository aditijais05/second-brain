"""
Chunker — splits Documents into retrieval-sized Chunks.

Strategy: semantic boundary chunking.
- Split on paragraph breaks and markdown headings first (natural boundaries).
- Then enforce a max token size by splitting oversized paragraphs further.
- Apply overlap so context is not lost at chunk edges.

This beats fixed-size chunking because it keeps related sentences together
and avoids cutting mid-sentence, which degrades retrieval quality.

Install: pip install tiktoken
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from core.document import Document
from core.chunk import Chunk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    """Tune these for your use case."""

    # Target chunk size in tokens (not words — tokens are model-native)
    chunk_size:    int = 400

    # Overlap between consecutive chunks (preserves cross-boundary context)
    chunk_overlap: int = 80

    # Hard minimum — discard chunks smaller than this (usually noise)
    min_chunk_size: int = 30

    # Encoding to use for token counting (cl100k = GPT-4 / text-embedding-3)
    encoding_name: str = "cl100k_base"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Splits a Document into overlapping Chunks at semantic boundaries.

    Pipeline:
        1. Split text into paragraphs (blank lines / headings).
        2. Merge small paragraphs into chunks up to `chunk_size` tokens.
        3. When a paragraph exceeds `chunk_size`, split it at sentence
           boundaries, then by token count as a last resort.
        4. Add `chunk_overlap` tokens of the previous chunk as a prefix
           to each new chunk so retrieval doesn't miss cross-boundary info.

    Usage:
        chunker = SemanticChunker()
        chunks  = chunker.chunk(document)
        chunks  = chunker.chunk_many(documents)
    """

    def __init__(self, config: ChunkerConfig | None = None):
        self.cfg = config or ChunkerConfig()
        self._enc = tiktoken.get_encoding(self.cfg.encoding_name)

    # ── Public ─────────────────────────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one Document into Chunks."""
        paragraphs = self._split_paragraphs(doc.content)
        raw_chunks = self._merge_paragraphs(paragraphs)
        raw_chunks = self._apply_overlap(raw_chunks)

        chunks = []
        word_offset = 0
        for idx, text in enumerate(raw_chunks):
            if self._token_count(text) < self.cfg.min_chunk_size:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    title=doc.title,
                    source_uri=doc.source_uri,
                    source_type=doc.source_type.value,
                    start_word=word_offset,
                    metadata={
                        "author":   doc.author or "",
                        "tags":     doc.tags,
                        "language": doc.language,
                    },
                )
            )
            word_offset += len(text.split())

        return chunks

    def chunk_many(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a list of Documents. Returns flat list of all Chunks."""
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk(doc)
            all_chunks.extend(doc_chunks)
            print(f"  chunked {doc.title!r:50s} → {len(doc_chunks)} chunks")
        print(f"\nTotal chunks: {len(all_chunks)}")
        return all_chunks

    # ── Step 1: paragraph splitting ────────────────────────────────────

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split on:
        - Blank lines (standard paragraph break)
        - Markdown headings (# / ## / ###)
        - [Page N] markers (injected by PdfIngestor)
        """
        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on blank lines OR markdown headings OR page markers
        parts = re.split(
            r"\n{2,}|(?=^#{1,3} )|(?=^\[Page \d+\])",
            text,
            flags=re.MULTILINE,
        )

        # Clean up each part
        paragraphs = [p.strip() for p in parts if p.strip()]
        return paragraphs

    # ── Step 2: merge small paragraphs into target-size chunks ─────────

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Greedily merge consecutive paragraphs until we'd exceed chunk_size.
        Oversized single paragraphs are split at sentence boundaries.
        """
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._token_count(para)

            # Single paragraph is already too large — split it first
            if para_tokens > self.cfg.chunk_size:
                # Flush current buffer first
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts, current_tokens = [], 0
                # Split the big paragraph
                chunks.extend(self._split_large_paragraph(para))
                continue

            # Adding this paragraph would overflow → flush and start fresh
            if current_tokens + para_tokens > self.cfg.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_tokens = [], 0

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remainder
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _split_large_paragraph(self, text: str) -> list[str]:
        """Split a paragraph that exceeds chunk_size at sentence boundaries."""
        # Split at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self._token_count(sentence)

            # Single sentence too long — hard split by tokens
            if s_tokens > self.cfg.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                    current, current_tokens = [], 0
                chunks.extend(self._hard_split(sentence))
                continue

            if current_tokens + s_tokens > self.cfg.chunk_size and current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0

            current.append(sentence)
            current_tokens += s_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by raw token count."""
        tokens = self._enc.encode(text)
        size   = self.cfg.chunk_size
        return [
            self._enc.decode(tokens[i : i + size])
            for i in range(0, len(tokens), size)
        ]

    # ── Step 3: add overlap ────────────────────────────────────────────

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Prefix each chunk with the last `chunk_overlap` tokens of the
        previous chunk. This ensures retrieval doesn't miss context that
        straddles a chunk boundary.
        """
        if self.cfg.chunk_overlap == 0 or len(chunks) < 2:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._enc.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.cfg.chunk_overlap :]
            overlap_text = self._enc.decode(overlap_tokens).strip()
            result.append(overlap_text + "\n\n" + chunks[i])

        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text))