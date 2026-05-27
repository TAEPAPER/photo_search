"""Text -> top_k photos.

Run as:
    uv run python -m photo_search.search "a photo of a beach"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from photo_search.embedding import CLIPEmbedder
from photo_search.vector_store import SearchHit, VectorStore


@dataclass
class PhotoSearcher:
    """Holds the heavy objects (CLIP model + Qdrant client) and exposes search().

    Construct once, reuse for every query. This will be the object FastAPI keeps
    around for the lifetime of the process.
    """

    embedder: CLIPEmbedder
    store: VectorStore

    @classmethod
    def default(cls) -> "PhotoSearcher":
        """Construct with default config (CLIP base + local Qdrant)."""
        embedder = CLIPEmbedder()
        store = VectorStore(dim=embedder.embedding_dim)
        return cls(embedder=embedder, store=store)

    def search(
        self,
        query: str,
        min_score: float | None = None,
        limit: int = 1000,
    ) -> list[SearchHit]:
        """Embed the query text and return all photos above `min_score`.

        With `min_score=None` you get the full ranked list up to `limit`.
        With `min_score=0.22` you get only matches Qdrant deems relevant.
        """
        # embed_texts returns shape (N, D); take the single query vector and
        # move it to a plain Python list for the JSON-ish wire format.
        vector = self.embedder.embed_texts([query])[0].cpu().tolist()
        return self.store.search(vector, limit=limit, score_threshold=min_score)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "a photo of a beach"
    print(f"Query: {query!r}")

    searcher = PhotoSearcher.default()
    hits = searcher.search(query, min_score=0.22)

    if not hits:
        print("No results above the score threshold. Try rephrasing or lower the threshold.")
        return

    print(f"\n{len(hits)} result(s):")
    for rank, hit in enumerate(hits, start=1):
        filename = hit.payload.get("filename", "?")
        print(f"  {rank}. {hit.score:+.4f}  {filename}")


if __name__ == "__main__":
    main()
