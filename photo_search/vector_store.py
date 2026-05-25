"""Qdrant vector store wrapper.

We keep the public surface deliberately small: ensure_collection / upsert /
search / count. That makes it easy to later swap Qdrant out for pgvector or
something else without touching the indexer or the API layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from photo_search import config


@dataclass
class SearchHit:
    """A single search result, decoupled from the underlying client's types."""

    id: str | int
    score: float
    payload: dict[str, Any]


@dataclass
class VectorStore:
    """Thin wrapper around qdrant-client for our photo collection."""

    url: str = config.QDRANT_URL
    collection: str = config.COLLECTION_NAME
    dim: int = 512  # CLIP ViT-B/32 default. Override at construction if needed.

    client: QdrantClient = field(init=False)

    def __post_init__(self) -> None:
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def upsert(self, points: list[PointStruct]) -> None:
        """Upsert a batch of points. Re-running with the same ids overwrites."""
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int = 10) -> list[SearchHit]:
        """Return the top_k most similar points to the query vector.

        Uses qdrant-client's `query_points` (the `search` method was removed in
        recent versions). `query_points` is the unified entry point that also
        supports filters and hybrid search.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
        )
        return [
            SearchHit(id=r.id, score=r.score, payload=r.payload or {})
            for r in response.points
        ]

    def count(self) -> int:
        """Exact count of points in the collection. Handy for sanity checks."""
        return self.client.count(collection_name=self.collection, exact=True).count
