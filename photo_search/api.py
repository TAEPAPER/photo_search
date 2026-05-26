"""FastAPI app exposing photo search over HTTP.

Run with:
    uv run uvicorn photo_search.api:app --reload --port 8000

Then:
    curl http://localhost:8000/health
    curl 'http://localhost:8000/search?q=a+photo+of+food&top_k=5'
    open http://localhost:8000/docs      # Swagger UI
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from photo_search.search import PhotoSearcher


# --- Lifespan -----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construct heavy objects (CLIP, Qdrant client) once at startup.

    Anything before `yield` runs at startup; anything after runs at shutdown.
    """
    app.state.searcher = PhotoSearcher.default()
    yield
    # No explicit teardown needed.


app = FastAPI(title="photo-search", lifespan=lifespan)


def get_searcher(request: Request) -> PhotoSearcher:
    """FastAPI dependency: returns the singleton searcher from app.state."""
    return request.app.state.searcher


# --- Response schemas ---------------------------------------------------------

class PhotoHit(BaseModel):
    id: str
    score: float
    filename: str
    path: str


class SearchResponse(BaseModel):
    query: str
    hits: list[PhotoHit]


# --- Routes -------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 once the model has finished loading."""
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, description="Natural language query"),
    top_k: int = Query(default=10, ge=1, le=50),
    searcher: PhotoSearcher = Depends(get_searcher),
) -> SearchResponse:
    """Text -> top-K matching photos."""
    hits = searcher.search(q, top_k=top_k)
    return SearchResponse(
        query=q,
        hits=[
            PhotoHit(
                id=str(h.id),
                score=h.score,
                filename=h.payload.get("filename", "?"),
                path=h.payload.get("path", ""),
            )
            for h in hits
        ],
    )


@app.get("/photo/{photo_id}")
def get_photo(
    photo_id: str,
    max_size: int = Query(
        default=1024, ge=64, le=4096,
        description="Resize so the longer side is at most this many pixels.",
    ),
    searcher: PhotoSearcher = Depends(get_searcher),
):
    """Stream the photo by stable ID, re-encoded as JPEG for browser compatibility.

    Why re-encode:
    - Browsers can't render HEIC/HEIF (iPhone's default) natively. JPEG works everywhere.
    - Downscaling keeps payloads small for snappy thumbnail rendering in the UI.

    The original file on disk is never modified — we re-encode on the fly.
    HEIC reading is enabled by `pillow_heif.register_heif_opener()`, which is
    already called when `photo_search.embedding` is imported.
    """
    # TODO: move this into vector_store.py once we have more retrievals.
    points = searcher.store.client.retrieve(
        collection_name=searcher.store.collection,
        ids=[photo_id],
    )
    if not points:
        raise HTTPException(status_code=404, detail="Photo not found")

    payload = points[0].payload or {}
    path = Path(payload.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=410, detail="Photo file no longer on disk")

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size))  # in-place, preserves aspect ratio
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
