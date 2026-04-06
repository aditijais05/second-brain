"""
Generator -- produces citation-grounded answers using Google Gemini (free API).

Setup:
    1. Get a free key at https://aistudio.google.com -> Get API Key
    2. pip install google-genai
    3. $env:GEMINI_API_KEY = "your-key-here"   (PowerShell)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CitedSource:
    """A single source chunk cited in an answer."""
    source_num:  int
    chunk_id:    str
    doc_id:      str
    title:       str
    source_uri:  str
    text:        str


@dataclass
class CitedAnswer:
    """
    The final output of the generation step.

    answer        -- answer text with inline [source_N] citations
    cited_sources -- only sources actually cited in the answer
    all_sources   -- all retrieved chunks passed as context
    query         -- the original user question
    model         -- the Gemini model used
    """
    answer:        str
    cited_sources: list[CitedSource]
    all_sources:   list[dict]
    query:         str
    model:         str = "gemini-2.0-flash"

    def format_answer(self) -> str:
        """Return answer + a References section, ready to display."""
        lines = [self.answer, "", "-" * 48, "References"]
        for s in self.cited_sources:
            lines.append(f"  [{s.source_num}] {s.title}")
            lines.append(f"       {s.source_uri}")
            preview = s.text[:120].strip()
            lines.append(f"       \"{preview}...\"")
        return "\n".join(lines)

    def cited_indices(self) -> list[int]:
        """Return the source numbers that appear in the answer."""
        return sorted({s.source_num for s in self.cited_sources})


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. Answer the user's question "
    "using ONLY the sources provided below.\n\n"
    "Rules:\n"
    "- Cite every factual claim with [source_N] immediately after the claim.\n"
    "- If a claim is supported by multiple sources, cite all: [source_1][source_3].\n"
    "- If the sources lack enough info, say so clearly.\n"
    "- Do NOT make up information not found in the sources.\n"
    "- Write in clear, concise prose.\n"
    "- Do not mention 'the sources say' -- just cite inline naturally."
)


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered source block for the prompt."""
    lines = ["SOURCES:"]
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown")
        text  = chunk.get("text",  "").strip()
        lines.append(f"\n[source_{i}] -- {title}")
        lines.append(text)
    return "\n".join(lines)


def build_user_message(query: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return f"{context}\n\n---\n\nQuestion: {query}"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CitationGenerator:
    """
    Generates citation-grounded answers using the Google Gemini API (free tier).

    Free tier: 15 requests/min, 1500 requests/day.

    Models:
        "gemini-2.0-flash"  -- fast, free, great quality (default)
        "gemini-1.5-flash"  -- fallback

    Usage:
        generator = CitationGenerator()
        answer = generator.generate(
            query="What is retrieval augmented generation?",
            chunks=retrieved_chunks,
        )
        print(answer.format_answer())
    """

    MODEL      = "gemini-2.0-flash"
    MAX_TOKENS = 1024

    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        self.model   = model or self.MODEL
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key found. Set $env:GEMINI_API_KEY or pass api_key=..."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._types  = types

    def generate(
        self,
        query:      str,
        chunks:     list[dict],
        max_tokens: int = MAX_TOKENS,
    ) -> CitedAnswer:
        """
        Generate a citation-grounded answer from retrieved chunks.

        Args:
            query:      The user's question.
            chunks:     Retrieved chunks (dicts from HybridRetriever.search()).
            max_tokens: Max tokens in the response.

        Returns:
            CitedAnswer with answer text, cited sources, and metadata.
        """
        if not chunks:
            return CitedAnswer(
                answer="I could not find any relevant sources to answer this question.",
                cited_sources=[],
                all_sources=[],
                query=query,
                model=self.model,
            )

        response = self._client.models.generate_content(
            model=self.model,
            contents=build_user_message(query, chunks),
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=max_tokens,
            ),
        )

        answer_text = response.text
        cited       = self._extract_citations(answer_text, chunks)

        return CitedAnswer(
            answer=answer_text,
            cited_sources=cited,
            all_sources=chunks,
            query=query,
            model=self.model,
        )

    # ── Private ────────────────────────────────────────────────────────

    def _extract_citations(
        self,
        answer_text: str,
        chunks:      list[dict],
    ) -> list[CitedSource]:
        """Parse [source_N] markers and map back to chunks."""
        cited_nums = set(
            int(m) for m in re.findall(r"\[source_(\d+)\]", answer_text)
        )
        cited_sources = []
        for num in sorted(cited_nums):
            idx = num - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                cited_sources.append(CitedSource(
                    source_num=num,
                    chunk_id=chunk.get("chunk_id", ""),
                    doc_id=chunk.get("doc_id", ""),
                    title=chunk.get("title", "Unknown"),
                    source_uri=chunk.get("source_uri", ""),
                    text=chunk.get("text", ""),
                ))
        return cited_sources