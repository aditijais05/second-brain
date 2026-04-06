"""
PDF Ingestor — extracts text and metadata from PDF files using PyMuPDF.

Install: pip install pymupdf
"""

from __future__ import annotations
from pathlib import Path

from datetime import datetime, timezone
from typing import Union

import fitz  # PyMuPDF

from core.document import Document, SourceType
from ingestion.base import BaseIngestor


class PdfIngestor(BaseIngestor):
    """
    Extracts full text + metadata from a PDF file.

    Strategy:
    - Use PyMuPDF's `get_text("blocks")` to preserve paragraph structure.
    - Preserve page numbers in content so downstream chunker can reference them.
    - Pull author/title from PDF metadata when available.

    Usage:
        ingestor = PdfIngestor()
        docs = ingestor.ingest("research_paper.pdf")
    """

    def __init__(self, include_page_numbers: bool = True):
        self.include_page_numbers = include_page_numbers

    def validate(self, source: Union[str, Path]) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {path.suffix}")

    def ingest(self, source: Union[str, Path]) -> list[Document]:
        self.validate(source)
        path = Path(source)

        self._log(f"Opening {path.name} …")
        doc = fitz.open(str(path))

        # ── Extract metadata ──────────────────────────────────────────
        meta   = doc.metadata or {}
        title  = meta.get("title") or path.stem
        author = meta.get("author") or None

        raw_date = meta.get("creationDate", "")
        created_at = self._parse_pdf_date(raw_date)

        # ── Extract text page by page ──────────────────────────────────
        pages_text = []
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,…)

            # Sort top-to-bottom, left-to-right (reading order)
            blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))

            page_content = "\n\n".join(
                b[4].strip() for b in blocks if b[4].strip()
            )

            if not page_content:
                continue  # skip empty/image-only pages

            if self.include_page_numbers:
                pages_text.append(f"[Page {page_num}]\n{page_content}")
            else:
                pages_text.append(page_content)

        full_text = "\n\n".join(pages_text)
        doc.close()

        self._log(f"Extracted {len(full_text.split())} words from {path.name}")

        return [
            Document(
                content=full_text,
                title=title,
                source_type=SourceType.PDF,
                source_uri=str(path.resolve()),
                author=author,
                created_at=created_at,
                extra={"page_count": len(doc), "pdf_meta": meta},
            )
        ]

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdf_date(raw: str) -> datetime:
        """Parse PDF date string like D:20231015120000+00'00' → datetime."""
        try:
            cleaned = raw.replace("D:", "").replace("'", "")[:14]
            return datetime.strptime(cleaned, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)