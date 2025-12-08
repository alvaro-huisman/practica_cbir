import argparse
import pathlib
import sys
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_extractor(name: str):
    if name == "color_histogram":
        from src.extractors.color_histogram import extract_color_histogram

        return extract_color_histogram
    if name == "sift_bow":
        from src.extractors.sift_bow import extract_sift_bow

        return extract_sift_bow
    if name == "orb_bow":
        from src.extractors.orb_bow import extract_orb_bow

        return extract_orb_bow
    if name == "resnet50":
        from src.extractors.resnet import extract_resnet50

        return extract_resnet50
    if name == "efficientnet_b0":
        from src.extractors.efficientnet import extract_efficientnet_b0

        return extract_efficientnet_b0

    raise ValueError(f"Extractor '{name}' is not supported.")


def average_precision(relevance: np.ndarray) -> float:
    """Compute Average Precision given a binary relevance array."""
    if relevance.sum() == 0:
        return 0.0

    cumulative_hits = np.cumsum(relevance)
    precision = cumulative_hits / (np.arange(len(relevance)) + 1)
    return float((precision * relevance).sum() / relevance.sum())


def evaluate(
    extractor_name: str,
    train_csv: pathlib.Path,
    test_csv: pathlib.Path,
    index_path: pathlib.Path,
    ks: List[int],
) -> Dict[str, float]:
    extractor = _get_extractor(extractor_name)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_labels = train_df["label"].values
    ks = sorted(set(ks))
    max_k = ks[-1]

    hit_counts = {k: 0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    ap_scores: List[float] = []

    for _, row in test_df.iterrows():
        image_path = ROOT / row["image"]
        with Image.open(image_path) as img:
            query_vec = extractor(img)

        distances, indices = index.search(np.float32(query_vec), k=max_k)
        retrieved_labels = train_labels[indices[0]]
        relevance = (retrieved_labels == row["label"]).astype(np.float32)

        for k in ks:
            topk_relevance = relevance[:k]
            hit_counts[k] += 1 if topk_relevance.any() else 0
            precision_sums[k] += topk_relevance.sum() / k

        ap_scores.append(average_precision(relevance))

    num_queries = len(test_df)
    metrics = {}
    for k in ks:
        metrics[f"hit_rate@{k}"] = hit_counts[k] / num_queries
        metrics[f"precision@{k}"] = precision_sums[k] / num_queries
    metrics["mAP"] = float(np.mean(ap_scores))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval metrics using a FAISS index."
    )
    parser.add_argument(
        "--extractor",
        default="color_histogram",
        help="Name of the extractor used to build the index (default: color_histogram).",
    )
    parser.add_argument(
        "--train-csv",
        type=pathlib.Path,
        default=ROOT / "database" / "db_train.csv",
        help="CSV used to build the index (default: database/db_train.csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=pathlib.Path,
        default=ROOT / "database" / "db_test.csv",
        help="CSV containing held-out queries (default: database/db_test.csv).",
    )
    parser.add_argument(
        "--index",
        type=pathlib.Path,
        default=ROOT / "faiss_indexes" / "color_histogram_db_train_l2.index",
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of k values for retrieval metrics (default: 1 5 10).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        args.extractor,
        args.train_csv,
        args.test_csv,
        args.index,
        args.k,
    )
    print("Retrieval metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
