import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
import faiss
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEBOOK = ROOT / "features" / "sift_codebook_k512.npy"
CODEBOOK_ENV = "SIFT_CODEBOOK_PATH"


@lru_cache(maxsize=1)
def _load_sift():
    sift = cv2.SIFT_create()
    return sift


@lru_cache(maxsize=1)
def _load_codebook() -> np.ndarray:
    env_path = os.getenv(CODEBOOK_ENV)
    path = Path(env_path) if env_path else DEFAULT_CODEBOOK
    if not path.exists():
        raise FileNotFoundError(
            f"SIFT codebook not found at {path}. "
            "Generate it with src/build_sift_codebook.py or set SIFT_CODEBOOK_PATH."
        )
    codebook = np.load(path).astype(np.float32)
    if codebook.ndim != 2:
        raise ValueError(f"Expected codebook shape (k, 128), got {codebook.shape}")
    if codebook.shape[1] != 128:
        raise ValueError(
            f"Unexpected descriptor dimension {codebook.shape[1]}, SIFT uses 128."
        )
    return codebook


@lru_cache(maxsize=1)
def _load_assignment_index() -> faiss.IndexFlatL2:
    codebook = _load_codebook()
    index = faiss.IndexFlatL2(codebook.shape[1])
    index.add(codebook)
    return index


def _to_gray(image_or_path: Union[Image.Image, str, Path]) -> np.ndarray:
    if isinstance(image_or_path, Image.Image):
        img = image_or_path.convert("L")
    else:
        img = Image.open(image_or_path).convert("L")
    return np.asarray(img)


def extract_sift_bow(image_or_path: Union[Image.Image, str, Path]) -> np.ndarray:
    """
    Compute a Bag-of-Visual-Words histogram using SIFT descriptors.
    Returns a normalized (L2) vector of shape (1, vocabulary_size).
    """
    gray = _to_gray(image_or_path)
    sift = _load_sift()
    _, descriptors = sift.detectAndCompute(gray, None)

    codebook = _load_codebook()
    vocab_size = codebook.shape[0]

    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, vocab_size), dtype=np.float32)

    index = _load_assignment_index()
    _, assignments = index.search(descriptors.astype(np.float32), k=1)

    hist = np.bincount(assignments.ravel(), minlength=vocab_size).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist.reshape(1, -1)


__all__ = ["extract_sift_bow", "DEFAULT_CODEBOOK", "CODEBOOK_ENV"]
