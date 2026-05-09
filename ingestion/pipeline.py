"""
Ingestion Pipeline — unified entry point that routes sources to the
correct ingestor and returns a deduplicated list of Documents.

Usage:
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    docs = pipeline.run([
        "notes/meeting.md",
        "papers/attention.pdf",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ])
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from core.document import Document
from ingestion.pdf_ingestor      import PdfIngestor
from ingestion.markdown_ingestor import MarkdownIngestor
from ingestion.url_ingestor      import UrlIngestor


Source = Union[str, Path]


class IngestionPipeline:
    """
    Routes each source to the right ingestor automatically.

    Deduplication:
    - Documents with the same `doc_id` (hash of source_uri + title)
      are silently dropped on re-ingestion.
    - This lets you re-run the pipeline incrementally without
      creating duplicates in your vector store.
    """

    def __init__(self):
        self._pdf      = PdfIngestor()
        self._markdown = MarkdownIngestor()
        self._url      = UrlIngestor()
        self._seen_ids: set[str] = set()

    # ── Public ─────────────────────────────────────────────────────────

    def run(self, sources: list[Source]) -> list[Document]:
        """
        Ingest a mixed list of file paths and URLs.
        Returns deduplicated Documents ready for the chunker.
        """
        all_docs: list[Document] = []

        for source in sources:
            try:
                docs = self._dispatch(source)
                new  = self._deduplicate(docs)
                all_docs.extend(new)
                print(f"✓ {source!r:60s} → {len(new)} doc(s)")
            except Exception as e:
                print(f"✗ {source!r:60s} → ERROR: {e}")

        print(f"\nIngestion complete: {len(all_docs)} documents total.")
        return all_docs

    def run_folder(self, folder: Union[str, Path], recursive: bool = True) -> list[Document]:
        """Ingest all markdown/PDF files inside a directory."""
        folder = Path(folder)
        sources = [
            f for f in (folder.rglob("*") if recursive else folder.glob("*"))
            if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}
        ]
        return self.run(sources)

    def reset_seen(self) -> None:
        """Clear dedup cache — forces re-ingestion of all sources on next run."""
        self._seen_ids.clear()

    # ── Private ────────────────────────────────────────────────────────

    def _dispatch(self, source: Source) -> list[Document]:
        s = str(source)
        if s.startswith("http://") or s.startswith("https://"):
            return self._url.ingest(s)

        path = Path(s)
        ext  = path.suffix.lower()

        if ext == ".pdf":
            return self._pdf.ingest(path)
        if ext in {".md", ".markdown", ".txt"}:
            return self._markdown.ingest(path)

        raise ValueError(
            f"Cannot ingest {path.name!r}: unknown extension '{ext}'. "
            f"Supported: .pdf, .md, .markdown, .txt, http(s)://"
        )

    def _deduplicate(self, docs: list[Document]) -> list[Document]:
        unique = []
        for doc in docs:
            if doc.doc_id not in self._seen_ids:
                self._seen_ids.add(doc.doc_id)
                unique.append(doc)
            else:
                print(f"  [dedup] Skipping duplicate: {doc.doc_id} ({doc.title!r})")
        return unique