"""
Tests for Week 4: BM25 index, RRF fusion, hybrid retriever.

Run: pytest tests/test_week4.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.chunk import Chunk
from retrieval.bm25_index import BM25Index, tokenise
from retrieval.rrf import reciprocal_rank_fusion


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_chunk(text: str, idx: int, doc_id: str = "doc_1") -> Chunk:
    return Chunk(
        text=text,
        doc_id=doc_id,
        chunk_index=idx,
        title=f"Doc {doc_id}",
        source_uri=f"/{doc_id}.md",
        source_type="markdown",
    )


CHUNKS = [
    make_chunk("Retrieval augmented generation combines search with LLMs.", 0),
    make_chunk("Vector databases store embeddings for semantic search.", 1),
    make_chunk("BM25 is a keyword ranking algorithm used in Elasticsearch.", 2),
    make_chunk("Chunking splits documents into smaller retrievable pieces.", 3),
    make_chunk("RAGAS is a framework for evaluating RAG pipelines.", 4),
]


# ── Tokeniser tests ────────────────────────────────────────────────────────

def test_tokenise_lowercases():
    assert tokenise("Hello World") == ["hello", "world"]

def test_tokenise_strips_punctuation():
    assert "." not in tokenise("Hello World.")

def test_tokenise_keeps_numbers():
    tokens = tokenise("GPT-4 scored 95%")
    assert any("4" in t for t in tokens)

def test_tokenise_empty():
    assert tokenise("") == []


# ── BM25 index tests ───────────────────────────────────────────────────────

def test_bm25_builds_successfully():
    index = BM25Index()
    index.build(CHUNKS)
    assert index.is_built
    assert len(index) == 5


def test_bm25_returns_relevant_result():
    index = BM25Index()
    index.build(CHUNKS)
    results = index.search("BM25 keyword ranking", top_k=3)
    assert len(results) >= 1
    # chunk about BM25 should rank first
    assert results[0]["chunk_id"] == CHUNKS[2].chunk_id


def test_bm25_search_ragas():
    index = BM25Index()
    index.build(CHUNKS)
    results = index.search("RAGAS evaluation framework", top_k=3)
    assert results[0]["chunk_id"] == CHUNKS[4].chunk_id


def test_bm25_returns_empty_for_no_match():
    index = BM25Index()
    index.build(CHUNKS)
    results = index.search("zzzznonexistenttermzzzz", top_k=5)
    assert results == []   # all scores 0, all skipped


def test_bm25_raises_if_not_built():
    index = BM25Index()
    with pytest.raises(RuntimeError):
        index.search("something")


def test_bm25_raises_on_empty_chunks():
    with pytest.raises(ValueError):
        BM25Index().build([])


def test_bm25_result_has_required_keys():
    index = BM25Index()
    index.build(CHUNKS)
    results = index.search("retrieval augmented generation", top_k=1)
    assert len(results) == 1
    for key in ("chunk_id", "doc_id", "title", "text", "score", "rank"):
        assert key in results[0], f"Missing key: {key}"


def test_bm25_save_and_load(tmp_path):
    index = BM25Index()
    index.build(CHUNKS)
    save_path = str(tmp_path / "bm25.pkl")
    index.save(save_path)

    loaded = BM25Index.load(save_path)
    assert len(loaded) == len(index)

    # Results should be identical after save/load
    r1 = index.search("retrieval", top_k=3)
    r2 = loaded.search("retrieval", top_k=3)
    assert [r["chunk_id"] for r in r1] == [r["chunk_id"] for r in r2]


# ── RRF tests ──────────────────────────────────────────────────────────────

def test_rrf_merges_two_lists():
    list_a = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    list_b = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]
    fused = reciprocal_rank_fusion(list_a, list_b)
    ids = [r["chunk_id"] for r in fused]
    assert set(ids) == {"a", "b", "c"}


def test_rrf_boosts_items_in_both_lists():
    # "b" appears in both lists → should score higher than "a" or "c"
    list_a = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    list_b = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    fused = reciprocal_rank_fusion(list_a, list_b)
    scores = {r["chunk_id"]: r["rrf_score"] for r in fused}
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rrf_assigns_rank():
    list_a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    fused = reciprocal_rank_fusion(list_a)
    ranks = [r["rrf_rank"] for r in fused]
    assert ranks == [1, 2]


def test_rrf_single_list():
    items = [{"chunk_id": str(i)} for i in range(5)]
    fused = reciprocal_rank_fusion(items)
    assert len(fused) == 5
    # Order should be preserved (rank 1 gets highest RRF score)
    assert fused[0]["chunk_id"] == "0"


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([], [])
    assert fused == []


def test_rrf_score_formula():
    """Verify RRF scores match the formula: 1/(k+rank)."""
    k = 60
    list_a = [{"chunk_id": "a"}]   # rank 1 in list_a
    fused = reciprocal_rank_fusion(list_a, k=k)
    expected = 1.0 / (k + 1)
    assert abs(fused[0]["rrf_score"] - expected) < 1e-9