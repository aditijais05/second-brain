"""
Vector Store — persists Chunks + embeddings in Qdrant.

Qdrant runs locally via Docker (no cloud account needed):
    docker run -p 6333:6333 qdrant/qdrant

Or use the in-memory mode for development (no Docker needed):
    store = VectorStore(mode="memory")

Install: pip install qdrant-client
"""

from __future__ import annotations

from typing import Literal
from core.chunk import Chunk


class VectorStore:
    """
    Thin wrapper around Qdrant that stores and retrieves Chunks by vector.

    Modes:
        "memory" — in-process, data lost on exit. Great for development.
        "local"  — persists to disk at `path`. No Docker needed.
        "server" — connects to a running Qdrant instance (Docker / cloud).

    Usage:
        # Development (no setup needed)
        store = VectorStore(mode="memory", dimension=384)
        store.upsert(chunks)
        results = store.search("what is attention?", query_vector=[...], top_k=5)

        # Persistent local storage
        store = VectorStore(mode="local", path="./qdrant_data", dimension=384)

        # Docker / production
        store = VectorStore(mode="server", host="localhost", port=6333, dimension=1536)
    """

    COLLECTION = "second_brain"

    def __init__(
        self,
        mode: Literal["memory", "local", "server"] = "memory",
        dimension: int = 384,
        path: str = "./qdrant_data",
        host: str = "localhost",
        port: int = 6333,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError("Run: pip install qdrant-client")

        self._PointStruct = PointStruct
        self.dimension    = dimension

        if mode == "memory":
            self._client = QdrantClient(":memory:")
        elif mode == "local":
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(host=host, port=port)

        # Create collection if it doesn't exist
        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION not in existing:
            from qdrant_client.models import Distance, VectorParams
            self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{self.COLLECTION}' (dim={dimension})")

    # ── Write ──────────────────────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings. Skips chunks without embeddings."""
        points = []
        skipped = 0
        for chunk in chunks:
            if chunk.embedding is None:
                skipped += 1
                continue
            points.append(
                self._PointStruct(
                    id=self._chunk_id_to_int(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload=chunk.to_metadata_dict() | {"text": chunk.text},
                )
            )

        if skipped:
            print(f"  ⚠ Skipped {skipped} chunks with no embedding")

        if not points:
            print("  Nothing to upsert.")
            return

        # Qdrant recommends batches of 100 for large uploads
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self.COLLECTION,
                points=points[i : i + batch_size],
            )

        print(f"  Upserted {len(points)} chunks into Qdrant ✓")

    # ── Read ───────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Return the top_k most similar chunks to a query vector.
        Each result is a dict with 'text', 'score', and all chunk metadata.
        """
        hits = self._client.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        return [
            {**hit.payload, "score": hit.score}
            for hit in hits
        ]

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._client.count(collection_name=self.COLLECTION).count

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        """Qdrant needs integer IDs. Convert hex chunk_id → int."""
        return int(chunk_id, 16)