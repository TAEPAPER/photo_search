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

MODEL_NAME = "openai/clip-vit-base-patch32"

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
