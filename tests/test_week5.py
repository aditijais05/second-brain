"""
Tests for Week 5: citation generator.

These tests mock the Gemini API so no API key is needed.

Run: pytest tests/test_week5.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from generation.generator import (
    CitationGenerator,
    CitedAnswer,
    CitedSource,
    build_context_block,
    build_user_message,
    SYSTEM_PROMPT,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "chunk_id":   "chunk_001",
        "doc_id":     "doc_1",
        "title":      "RAG Overview",
        "source_uri": "/notes/rag.md",
        "text":       "Retrieval Augmented Generation (RAG) combines a retrieval system with a language model to produce grounded answers.",
        "rrf_score":  0.032,
        "rrf_rank":   1,
    },
    {
        "chunk_id":   "chunk_002",
        "doc_id":     "doc_2",
        "title":      "Vector Databases",
        "source_uri": "/notes/vectors.md",
        "text":       "Vector databases like Qdrant store embeddings and support fast approximate nearest neighbour search.",
        "rrf_score":  0.028,
        "rrf_rank":   2,
    },
    {
        "chunk_id":   "chunk_003",
        "doc_id":     "doc_3",
        "title":      "BM25 Explained",
        "source_uri": "/notes/bm25.md",
        "text":       "BM25 is a probabilistic ranking function that scores documents based on term frequency and inverse document frequency.",
        "rrf_score":  0.021,
        "rrf_rank":   3,
    },
]

MOCK_ANSWER = (
    "RAG combines a retrieval system with a language model [source_1]. "
    "It often uses vector databases for semantic search [source_2]. "
    "Some systems also use BM25 for keyword matching [source_3]."
)


def make_generator(answer_text: str = MOCK_ANSWER) -> CitationGenerator:
    """Create a CitationGenerator with a fully mocked Gemini client."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
        with patch("google.genai.Client"):
            gen = CitationGenerator()

    # Mock _client.models.generate_content to return a fake response
    mock_response      = MagicMock()
    mock_response.text = answer_text
    gen._client        = MagicMock()
    gen._client.models.generate_content.return_value = mock_response
    return gen


# ── Prompt builder tests ───────────────────────────────────────────────────

def test_build_context_block_numbers_sources():
    block = build_context_block(SAMPLE_CHUNKS)
    assert "[source_1]" in block
    assert "[source_2]" in block
    assert "[source_3]" in block


def test_build_context_block_includes_text():
    block = build_context_block(SAMPLE_CHUNKS)
    assert "Retrieval Augmented Generation" in block
    assert "Qdrant" in block


def test_build_context_block_includes_title():
    block = build_context_block(SAMPLE_CHUNKS)
    assert "RAG Overview" in block
    assert "Vector Databases" in block


def test_build_user_message_includes_question():
    msg = build_user_message("What is RAG?", SAMPLE_CHUNKS)
    assert "What is RAG?" in msg
    assert "[source_1]" in msg


def test_system_prompt_mentions_citation():
    assert "[source_N]" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()


# ── CitationGenerator tests ────────────────────────────────────────────────

def test_generator_returns_cited_answer():
    gen    = make_generator()
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    assert isinstance(result, CitedAnswer)


def test_generator_answer_text_preserved():
    gen    = make_generator()
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    assert result.answer == MOCK_ANSWER


def test_generator_extracts_all_citations():
    gen    = make_generator()
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    cited_nums = {s.source_num for s in result.cited_sources}
    assert cited_nums == {1, 2, 3}


def test_generator_maps_citations_to_correct_chunks():
    gen        = make_generator()
    result     = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    source_map = {s.source_num: s for s in result.cited_sources}
    assert source_map[1].chunk_id == "chunk_001"
    assert source_map[2].chunk_id == "chunk_002"
    assert source_map[3].chunk_id == "chunk_003"


def test_generator_cited_sources_have_correct_titles():
    gen    = make_generator()
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    titles = {s.title for s in result.cited_sources}
    assert "RAG Overview" in titles
    assert "Vector Databases" in titles


def test_generator_empty_chunks_returns_gracefully():
    gen    = make_generator()
    result = gen.generate("What is RAG?", [])
    assert isinstance(result, CitedAnswer)
    assert result.cited_sources == []
    assert "could not find" in result.answer.lower()


def test_generator_partial_citation():
    """If the model only cites source_1, only that appears in cited_sources."""
    gen    = make_generator(answer_text="RAG is great [source_1].")
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    assert len(result.cited_sources) == 1
    assert result.cited_sources[0].source_num == 1


def test_generator_all_sources_always_populated():
    gen    = make_generator()
    result = gen.generate("What is RAG?", SAMPLE_CHUNKS)
    assert len(result.all_sources) == len(SAMPLE_CHUNKS)


def test_generator_calls_gemini_api():
    """Verify the Gemini client is actually called with the right model."""
    gen    = make_generator()
    gen.generate("What is RAG?", SAMPLE_CHUNKS)
    gen._client.models.generate_content.assert_called_once()
    call_kwargs = gen._client.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-2.0-flash"


# ── CitedAnswer tests ──────────────────────────────────────────────────────

def test_cited_answer_format_answer():
    answer = CitedAnswer(
        answer="RAG is useful [source_1].",
        cited_sources=[
            CitedSource(
                source_num=1,
                chunk_id="chunk_001",
                doc_id="doc_1",
                title="RAG Overview",
                source_uri="/notes/rag.md",
                text="RAG combines retrieval with generation.",
            )
        ],
        all_sources=SAMPLE_CHUNKS,
        query="What is RAG?",
    )
    formatted = answer.format_answer()
    assert "RAG is useful [source_1]." in formatted
    assert "References" in formatted
    assert "RAG Overview" in formatted


def test_cited_answer_cited_indices():
    answer = CitedAnswer(
        answer="Text [source_2] and [source_4].",
        cited_sources=[
            CitedSource(2, "c2", "d2", "T2", "/t2", "text"),
            CitedSource(4, "c4", "d4", "T4", "/t4", "text"),
        ],
        all_sources=[],
        query="q",
    )
    assert answer.cited_indices() == [2, 4]