"""Index photos from a folder into the vector store.

Run as:
    uv run python -m photo_search.indexer

Re-running is safe: each file's ID is a stable hash of its absolute path, so
re-indexing the same files just overwrites the existing points (no duplicates).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from itertools import batched
from pathlib import Path

from PIL import Image
from qdrant_client.models import PointStruct
from tqdm import tqdm

from photo_search import config
from photo_search.embedding import CLIPEmbedder
from photo_search.vector_store import VectorStore


def find_images(root: Path) -> list[Path]:
    """Recursively find image files we know how to read."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def stable_id(path: Path) -> str:
    """Deterministic UUID from a file's absolute path.

    Same path -> same UUID, so re-indexing the same file is an overwrite, not
    a duplicate. Qdrant point IDs must be int or UUID — we go with UUID.
    """
    digest = hashlib.md5(str(path.resolve()).encode()).hexdigest()
    return str(uuid.UUID(digest))


def build_payload(path: Path) -> dict:
    """Metadata to attach to each point.

    Keep this small — heavy stuff (the image bytes themselves) stays on disk;
    the payload only needs to be enough to render and re-locate the file.
    """
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "indexed_at": time.time(),
    }


def index_photos(
    photos_dir: Path | None = None,
    batch_size: int | None = None,
) -> int:
    """Index all photos under photos_dir into the vector store.

    Returns the number of photos indexed in this run.
    """
    photos_dir = photos_dir or config.PHOTOS_DIR
    batch_size = batch_size or config.BATCH_SIZE

    paths = find_images(photos_dir)
    if not paths:
        print(f"No images found under {photos_dir}")
        return 0
    print(f"Found {len(paths)} image(s) under {photos_dir}")

    # Heavy objects: construct once, reuse for the whole run.
    embedder = CLIPEmbedder()
    store = VectorStore(dim=embedder.embedding_dim)
    store.ensure_collection()

    indexed = 0
    batches = list(batched(paths, batch_size))
    for batch in tqdm(batches, desc="Indexing", unit="batch"):
        # 1. Load images
        images = [Image.open(p).convert("RGB") for p in batch]
        # 2. Embed (on GPU/MPS), then move back to CPU as plain Python lists
        #    so qdrant-client can serialize them.
        vectors = embedder.embed_images(images).cpu().tolist()
        # 3. Build Qdrant points
        points = [
            PointStruct(id=stable_id(p), vector=v, payload=build_payload(p))
            for p, v in zip(batch, vectors)
        ]
        # 4. Upsert
        store.upsert(points)
        indexed += len(points)

    total = store.count()
    print(f"Indexed {indexed} photo(s) into '{store.collection}' (collection total: {total})")
    return indexed


if __name__ == "__main__":
    index_photos()
