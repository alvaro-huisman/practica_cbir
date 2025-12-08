import argparse
import pathlib
import sys

import faiss
import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features"
FAISS_DIR = ROOT / "faiss_indexes"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_features(features_path: pathlib.Path) -> np.ndarray:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features = np.load(features_path)
    if features.ndim != 2:
        raise ValueError(
            f"Expected features to have shape (n_samples, dim), got {features.shape}"
        )

    return features.astype(np.float32)


def build_index(
    extractor_name: str,
    features_path: pathlib.Path,
    db_path: pathlib.Path,
    metric: str = "l2",
) -> pathlib.Path:
    features = load_features(features_path)
    n_samples, dim = features.shape

    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    elif metric == "cosine":
        faiss.normalize_L2(features)
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("Metric must be either 'l2' or 'cosine'")

    index.add(features)

    FAISS_DIR.mkdir(exist_ok=True)
    index_stem = f"{features_path.stem}_{metric}"
    index_path = FAISS_DIR / f"{index_stem}.index"
    faiss.write_index(index, str(index_path))

    # Persist metadata mapping (faiss id -> image path, label)
    if not db_path.exists():
        raise FileNotFoundError(f"Database CSV not found at {db_path}")

    db = pd.read_csv(db_path)
    metadata_path = FAISS_DIR / f"{index_stem}.csv"
    db.to_csv(metadata_path, index=False)

    print(f"Stored {n_samples} vectors of dim {dim} using metric '{metric}'.")
    print(f"FAISS index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")

    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from features.")
    parser.add_argument(
        "--extractor",
        default="color_histogram",
        help="Name of the extractor whose features were saved in features/<name>.npy",
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=ROOT / "database" / "db_train.csv",
        help="CSV used for the FAISS index metadata (default: database/db_train.csv).",
    )
    parser.add_argument(
        "--features",
        type=pathlib.Path,
        help=(
            "Path to the .npy feature file. "
            "If omitted, uses features/<extractor>_<dbstem>.npy"
        ),
    )
    parser.add_argument(
        "--metric",
        default="l2",
        choices=("l2", "cosine"),
        help="Distance metric for FAISS (default: l2)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = args.features
    if features_path is None:
        features_path = FEATURES_DIR / f"{args.extractor}_{args.db.stem}.npy"
    build_index(args.extractor, features_path, args.db, metric=args.metric)


if __name__ == "__main__":
    main()
