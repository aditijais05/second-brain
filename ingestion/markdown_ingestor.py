"""
Markdown Ingestor — parses .md and .txt files, extracts YAML frontmatter.

Install: pip install python-frontmatter
"""

from __future__ import annotations
from pathlib import Path

from datetime import datetime, timezone
from typing import Union

import frontmatter  # python-frontmatter

from core.document import Document, SourceType
from ingestion.base import BaseIngestor


class MarkdownIngestor(BaseIngestor):
    """
    Parses markdown files and extracts YAML frontmatter metadata.

    Frontmatter example:
        ---
        title: "My Note"
        author: "Alice"
        date: 2024-01-15
        tags: [rag, llm, notes]
        ---

    If frontmatter is absent, title defaults to filename and
    date defaults to file modification time.

    Usage:
        ingestor = MarkdownIngestor()
        docs = ingestor.ingest("notes/my_note.md")

        # Ingest entire folder
        docs = ingestor.ingest_folder("notes/")
    """

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}

    def validate(self, source: Union[str, Path]) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {path.suffix}")

    def ingest(self, source: Union[str, Path]) -> list[Document]:
        self.validate(source)
        path = Path(source)

        self._log(f"Parsing {path.name} …")
        post = frontmatter.load(str(path))

        content = post.content.strip()
        if not content:
            self._log(f"Skipping empty file: {path.name}")
            return []

        # ── Metadata from frontmatter or file system ──────────────────
        title  = post.get("title") or path.stem
        author = post.get("author") or None
        tags   = self._normalise_tags(post.get("tags", []))
        lang   = post.get("language") or "en"

        raw_date   = post.get("date") or post.get("created_at")
        created_at = self._parse_date(raw_date) or self._mtime(path)

        self._log(f"Loaded {len(content.split())} words — '{title}'")

        return [
            Document(
                content=content,
                title=title,
                source_type=SourceType.MARKDOWN,
                source_uri=str(path.resolve()),
                author=author,
                tags=tags,
                language=lang,
                created_at=created_at,
                extra={"frontmatter": dict(post.metadata)},
            )
        ]

    def ingest_folder(
        self,
        folder: Union[str, Path],
        recursive: bool = True,
    ) -> list[Document]:
        """Ingest all supported markdown files in a directory."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        pattern = "**/*" if recursive else "*"
        files = [
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        self._log(f"Found {len(files)} files in {folder}")
        docs = []
        for f in sorted(files):
            try:
                docs.extend(self.ingest(f))
            except Exception as e:
                self._log(f"⚠ Skipped {f.name}: {e}")
        return docs

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_tags(raw) -> list[str]:
        if isinstance(raw, list):
            return [str(t).strip().lower() for t in raw]
        if isinstance(raw, str):
            return [t.strip().lower() for t in raw.split(",") if t.strip()]
        return []

    @staticmethod
    def _parse_date(raw) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(raw), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)