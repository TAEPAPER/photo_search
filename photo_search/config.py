"""Project-wide configuration constants.

Kept as plain module-level constants for now. If we later want to drive these
from environment variables or a .env file, this is the one place to change.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------

# photo_search/config.py -> photo_search/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHOTOS_DIR = PROJECT_ROOT / "data" / "photos"

# --- CLIP model --------------------------------------------------------------

# Larger CLIP variant with patch size 14 (vs 32 on the base model).
# - Embedding dim: 768 (vs 512 on base) — collection must be recreated.
# - Download: ~1.7GB (one-time)
# - Indexing: ~2-3x slower per image; query latency only mildly higher
# - Quality: noticeably better at fine-grained discrimination
MODEL_NAME = "openai/clip-vit-large-patch14"

# --- Qdrant ------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "photos"

# --- Indexing ----------------------------------------------------------------

# Number of images processed in a single CLIP forward pass.
# Tuned for Apple Silicon + 16GB RAM. Larger = faster but more memory pressure.
BATCH_SIZE = 16

# Image file extensions we'll try to read. Lowercase, with the leading dot.
SUPPORTED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png",
    ".heic", ".heif",
    ".webp", ".bmp", ".tiff",
})
