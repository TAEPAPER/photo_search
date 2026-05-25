"""Explore CLIP: embed one image, embed a few texts, compute cosine similarity.

This is a throwaway exploration script — its only job is to make sure the
full pipeline (load model -> read image -> embed -> compare with text) works
end to end on this machine before we wire it into the real package.
"""
from pathlib import Path

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from transformers import CLIPModel, CLIPProcessor

# Teach PIL how to read HEIC/HEIF (iPhone) images.
# Must be called before any Image.open() on a HEIC file.
register_heif_opener()

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_PATH = Path("data/photos/sample.jpg")

# A few candidate captions to compare the image against.
# Feel free to edit these to match what you expect your photo to be.
TEXT_CANDIDATES = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a beach",
    "a photo of food",
    "a photo of a person",
    "a photo of a building",
    "a photo of a car",
    "a photo of mountains",
]


def pick_device() -> str:
    """Prefer Apple Silicon GPU (MPS), fall back to CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    device: str,
) -> torch.Tensor:
    """Run an image through CLIP. Returns an L2-normalized embedding of shape (1, D)."""
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def embed_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: str,
) -> torch.Tensor:
    """Run a list of texts through CLIP. Returns L2-normalized embeddings of shape (N, D)."""
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()  # inference mode

    print(f"Loading image: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"  size: {image.size[0]} x {image.size[1]}")

    image_features = embed_image(model, processor, image, device)
    print(f"Image embedding shape: {tuple(image_features.shape)}")

    text_features = embed_texts(model, processor, TEXT_CANDIDATES, device)
    print(f"Text embeddings shape: {tuple(text_features.shape)}")

    # Cosine similarity = dot product of L2-normalized vectors.
    # (1, D) @ (D, N) -> (1, N) -> (N,)
    scores = (image_features @ text_features.T).squeeze(0)

    print("\nRanking (higher = more similar to the image):")
    ranked = sorted(zip(TEXT_CANDIDATES, scores.tolist()), key=lambda pair: -pair[1])
    for text, score in ranked:
        print(f"  {score:+.4f}  {text}")


if __name__ == "__main__":
    main()
