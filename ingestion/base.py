"""
Base class that every ingestor must implement.
Keeps the pipeline uniform regardless of source type.
"""

from __future__ import annotations
from pathlib import Path

from abc import ABC, abstractmethod
from typing import Union
from core.document import Document


class BaseIngestor(ABC):
    """
    All ingestors share this interface:

        ingestor = PdfIngestor()
        docs: list[Document] = ingestor.ingest("path/or/url")

    Subclasses implement `_load()` to return raw text + metadata,
    and may override `validate()` for source-specific checks.
    """

    @abstractmethod
    def ingest(self, source: Union[str, Path]) -> list[Document]:
        """Parse source and return a list of Document objects."""
        ...

    def validate(self, source: Union[str, Path]) -> None:
        """Raise ValueError if source is invalid for this ingestor."""
        pass

    def _log(self, msg: str) -> None:
        # Swap for logging.getLogger(__name__) in production
        print(f"[{self.__class__.__name__}] {msg}")