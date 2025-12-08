import argparse
import pathlib
import sys
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_extractor(name: str) -> Callable[[Image.Image], np.ndarray]:
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


def build_features(extractor_name: str, db_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    if not db_path.exists():
        raise FileNotFoundError(f"Database CSV not found at {db_path}")

    db = pd.read_csv(db_path)
    if "image" not in db.columns:
        raise ValueError("db.csv must contain an 'image' column with image paths.")

    extractor = _get_extractor(extractor_name)
    features = []

    for idx, image_rel_path in enumerate(db["image"], start=1):
        image_path = ROOT / image_rel_path
        with Image.open(image_path) as img:
            feature = extractor(img)

        features.append(feature)

    features_array = np.vstack(features)

    FEATURES_DIR.mkdir(exist_ok=True)
    np.save(out_path, features_array)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate feature vectors for images listed in database/db.csv."
    )
    parser.add_argument(
        "--extractor",
        default="color_histogram",
        help="Name of the feature extractor to use (default: color_histogram)",
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=ROOT / "database" / "db_train.csv",
        help="CSV listing images to encode (default: database/db_train.csv).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        help=(
            "Optional path for the output .npy file. "
            "If omitted, saves to features/<extractor>_<dbstem>.npy"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        stem = args.db.stem
        out_path = FEATURES_DIR / f"{args.extractor}_{stem}.npy"
    out_path = build_features(args.extractor, args.db, out_path)
    print(f"Features saved to {out_path}")


if __name__ == "__main__":
    main()
