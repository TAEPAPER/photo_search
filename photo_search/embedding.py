"""CLIP-based image and text embeddings.

Wraps Hugging Face's CLIPModel + CLIPProcessor into a small reusable class.
The model is loaded once at construction time and reused for every call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from transformers import CLIPModel, CLIPProcessor

# Register HEIC/HEIF reader once at import time so PIL.Image.open() handles
# iPhone photos without any extra ceremony at the call site.
register_heif_opener()


@dataclass
class CLIPEmbedder:
    """Loads a CLIP model and produces L2-normalized embeddings.

    Image and text embeddings live in the same vector space, so the dot product
    of an image embedding with a text embedding equals their cosine similarity.

    The model is heavy (~600MB on disk, sizeable in memory). Construct one
    instance and reuse it for both indexing and querying.
    """

    model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None  # None -> auto-detect

    # Filled in __post_init__. `init=False` keeps them out of the constructor.
    model: CLIPModel = field(init=False)
    processor: CLIPProcessor = field(init=False)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = self._pick_device()
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()  # inference mode

    @staticmethod
    def _pick_device() -> str:
        """Prefer Apple Silicon GPU, then CUDA, then CPU."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def embedding_dim(self) -> int:
        """Output dimension of the joint image/text embedding space.

        Read from the model config rather than hardcoded so swapping the
        backbone (e.g. to a Large variant with 768-dim embeddings) just works.
        """
        return self.model.config.projection_dim

    def embed_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """Embed a batch of PIL images.

        Returns an L2-normalized tensor of shape (N, embedding_dim).
        """
        images = list(images)
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def embed_texts(self, texts: Iterable[str]) -> torch.Tensor:
        """Embed a batch of texts.

        Returns an L2-normalized tensor of shape (N, embedding_dim).
        """
        texts = list(texts)
        with torch.no_grad():
            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True
            ).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def embed_image_paths(self, paths: Iterable[Path]) -> torch.Tensor:
        """Convenience: read images from disk then embed them.

        Note: this loads every image into memory at once. Fine for tens of
        photos; for a few hundred we'll want to batch in chunks (next step).
        """
        images = [Image.open(p).convert("RGB") for p in paths]
        return self.embed_images(images)
