import argparse
import pathlib
from typing import List

import faiss
import numpy as np
import pandas as pd
from PIL import Image

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "database" / "db_train.csv"
DEFAULT_OUTPUT = ROOT / "features" / "sift_codebook_k512.npy"


def collect_descriptors(
    db_path: pathlib.Path,
    max_per_image: int,
    max_descriptors: int,
) -> np.ndarray:
    if not db_path.exists():
        raise FileNotFoundError(f"Database CSV not found: {db_path}")

    df = pd.read_csv(db_path)
    if "image" not in df.columns:
        raise ValueError("CSV must contain an 'image' column.")

    sift = cv2.SIFT_create()
    descriptors: List[np.ndarray] = []

    total = 0
    for _, row in df.iterrows():
        img_path = ROOT / row["image"]
        with Image.open(img_path) as img:
            gray = np.asarray(img.convert("L"))

        _, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            continue

        if max_per_image > 0:
            desc = desc[:max_per_image]

        descriptors.append(desc.astype(np.float32))
        total += len(desc)

        if total >= max_descriptors:
            break

    if not descriptors:
        raise RuntimeError("No SIFT descriptors collected. Check your images.")

    stacked = np.vstack(descriptors)
    if len(stacked) > max_descriptors:
        stacked = stacked[:max_descriptors]
    return stacked


def build_codebook(
    descriptors: np.ndarray, vocab_size: int, niter: int, seed: int
) -> np.ndarray:
    d = descriptors.shape[1]
    kmeans = faiss.Kmeans(d=d, k=vocab_size, niter=niter, seed=seed, verbose=True)
    kmeans.train(descriptors)
    return kmeans.centroids.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIFT Bag-of-Words codebook using faiss.Kmeans."
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=DEFAULT_DB,
        help="CSV with image paths for codebook training (default: database/db_train.csv).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="File to store the codebook (default: features/sift_codebook_k512.npy).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Number of visual words in the vocabulary (default: 512).",
    )
    parser.add_argument(
        "--max-descriptors",
        type=int,
        default=200_000,
        help="Maximum total SIFT descriptors to collect (default: 200000).",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=1000,
        help="Maximum descriptors per image (default: 1000, 0 for unlimited).",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=25,
        help="Number of k-means iterations (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for k-means (default: 123).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptors = collect_descriptors(
        args.db, args.max_per_image, args.max_descriptors
    )
    vocab = build_codebook(descriptors, args.vocab_size, args.niter, args.seed)

    args.output.parent.mkdir(exist_ok=True)
    np.save(args.output, vocab)
    print(
        f"Codebook with {vocab.shape[0]} visual words saved to {args.output} "
        f"(dimension {vocab.shape[1]})."
    )


if __name__ == "__main__":
    main()
