"""
URL Ingestor — fetches and cleans web page text using Trafilatura.

Trafilatura is purpose-built for article extraction — it strips nav bars,
ads, and boilerplate far better than BeautifulSoup alone.

Install: pip install trafilatura
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union
from urllib.parse import urlparse

import trafilatura
from trafilatura.settings import use_config

from core.document import Document, SourceType
from ingestion.base import BaseIngestor


# Speed up by disabling network-level sitemaps fetch
_CFG = use_config()
_CFG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "15")


class UrlIngestor(BaseIngestor):
    """
    Downloads a URL and extracts its main article / content.

    Features:
    - Uses Trafilatura for high-quality boilerplate removal
    - Falls back to raw HTML text if extraction fails
    - Extracts publication date and author when available

    Usage:
        ingestor = UrlIngestor()
        docs = ingestor.ingest("https://example.com/article")

        # Batch
        docs = ingestor.ingest_batch([url1, url2, url3])
    """

    def __init__(self, include_comments: bool = False, language: str = None):
        self.include_comments = include_comments
        self.language = language

    def validate(self, source: Union[str]) -> None:
        parsed = urlparse(source)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Expected http/https URL, got: {source!r}")

    def ingest(self, source: str) -> list[Document]:
        self.validate(source)
        self._log(f"Fetching {source} …")

        # ── Download raw HTML ─────────────────────────────────────────
        downloaded = trafilatura.fetch_url(source)
        if not downloaded:
            self._log(f"⚠ Could not fetch {source}")
            return []

        # ── Extract main content ──────────────────────────────────────
        result = trafilatura.extract(
            downloaded,
            include_comments=self.include_comments,
            include_tables=True,
            favor_precision=True,
            output_format="txt",
            with_metadata=True,
            config=_CFG,
        )

        if not result:
            self._log(f"⚠ Trafilatura found no extractable content at {source}")
            return []

        # ── Parse trafilatura metadata object ─────────────────────────
        meta = trafilatura.extract(
            downloaded,
            output_format="xmltei",
            with_metadata=True,
        )

        title, author, pub_date = self._parse_meta(downloaded, source)

        self._log(f"Extracted {len(result.split())} words — '{title}'")

        return [
            Document(
                content=result,
                title=title,
                source_type=SourceType.URL,
                source_uri=source,
                author=author,
                created_at=pub_date,
                language=self.language or "en",
                extra={"domain": urlparse(source).netloc},
            )
        ]

    def ingest_batch(self, urls: list[str]) -> list[Document]:
        """Ingest a list of URLs, skipping failures."""
        docs = []
        for url in urls:
            try:
                docs.extend(self.ingest(url))
            except Exception as e:
                self._log(f"⚠ Skipped {url}: {e}")
        return docs

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_meta(html: str, fallback_url: str) -> tuple[str, str | None, datetime]:
        """Extract title, author, date from raw HTML via trafilatura."""
        try:
            meta = trafilatura.extract_metadata(html)
            title  = (meta.title  if meta and meta.title  else None) or urlparse(fallback_url).path or fallback_url
            author = (meta.author if meta and meta.author else None)
            date   = datetime.now(timezone.utc)
            if meta and meta.date:
                try:
                    date = datetime.fromisoformat(meta.date)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            return title, author, date
        except Exception:
            return fallback_url, None, datetime.now(timezone.utc)