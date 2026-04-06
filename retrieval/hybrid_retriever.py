"""
Hybrid Retriever — combines vector search and BM25 using RRF.
"""

from __future__ import annotations

from embeddings.embedder import BaseEmbedder
from embeddings.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.rrf import reciprocal_rank_fusion, explain_rrf
from core.chunk import Chunk


class HybridRetriever:

    def __init__(
        self,
        vector_store:      VectorStore,
        bm25_index:        BM25Index,
        embedder:          BaseEmbedder,
        vector_candidates: int = 20,
        bm25_candidates:   int = 20,
        rrf_k:             int = 60,
    ):
        self.vector_store      = vector_store
        self.bm25_index        = bm25_index
        self.embedder          = embedder
        self.vector_candidates = vector_candidates
        self.bm25_candidates   = bm25_candidates
        self.rrf_k             = rrf_k

    def search(self, query: str, top_k: int = 5, explain: bool = False) -> list[dict]:
        query_embedding = self._embed_query(query)

        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=self.vector_candidates,
        )
        for r in vector_results:
            r["vector_score"] = r.pop("score", 0.0)

        bm25_results = self.bm25_index.search(query=query, top_k=self.bm25_candidates)
        for r in bm25_results:
            r["bm25_score"] = r.pop("score", 0.0)

        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=self.rrf_k)

        if explain:
            explain_rrf(vector_results, bm25_results, k=self.rrf_k, top_k=top_k)

        return fused[:top_k]

    def bm25_only(self, query: str, top_k: int = 5) -> list[dict]:
        return self.bm25_index.search(query, top_k=top_k)

    def _embed_query(self, query: str) -> list[float]:
        placeholder = Chunk(
            text=query, doc_id="__query__", chunk_index=0,
            title="query", source_uri="query", source_type="query",
        )
        [embedded] = self.embedder.embed_chunks([placeholder])
        return embedded.embedding