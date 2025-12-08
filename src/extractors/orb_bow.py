import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
import faiss
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODEBOOK = ROOT / "features" / "orb_codebook_k512.npy"
CODEBOOK_ENV = "ORB_CODEBOOK_PATH"


@lru_cache(maxsize=1)
def _load_orb():
    # nfeatures=1000 to gather sufficient keypoints.
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    return orb


@lru_cache(maxsize=1)
def _load_codebook() -> np.ndarray:
    env_path = os.getenv(CODEBOOK_ENV)
    path = Path(env_path) if env_path else DEFAULT_CODEBOOK
    if not path.exists():
        raise FileNotFoundError(
            f"ORB codebook not found at {path}. "
            "Generate it with src/build_orb_codebook.py or set ORB_CODEBOOK_PATH."
        )
    codebook = np.load(path).astype(np.float32)
    if codebook.ndim != 2:
        raise ValueError(f"Expected codebook shape (k, d), got {codebook.shape}")
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


def extract_orb_bow(image_or_path: Union[Image.Image, str, Path]) -> np.ndarray:
    gray = _to_gray(image_or_path)
    orb = _load_orb()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    codebook = _load_codebook()
    vocab_size, dim = codebook.shape

    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, vocab_size), dtype=np.float32)

    descriptors = descriptors.astype(np.float32)
    if descriptors.shape[1] != dim:
        raise ValueError(
            f"Descriptor dimension mismatch: expected {dim}, got {descriptors.shape[1]}"
        )

    index = _load_assignment_index()
    _, assignments = index.search(descriptors, k=1)

    hist = np.bincount(assignments.ravel(), minlength=vocab_size).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist.reshape(1, -1)


__all__ = ["extract_orb_bow", "DEFAULT_CODEBOOK", "CODEBOOK_ENV"]
