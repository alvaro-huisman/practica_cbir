import argparse
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_CSV = ROOT / "database" / "db.csv"


def split_dataset(
    db_path: pathlib.Path, train_count: int, test_count: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database CSV not found at {db_path}")

    df = pd.read_csv(db_path)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("db.csv must contain 'image' and 'label' columns.")

    train_parts = []
    test_parts = []

    rng = np.random.default_rng(seed)

    for label, group in df.groupby("label"):
        if len(group) < train_count + test_count:
            raise ValueError(
                f"Class '{label}' has {len(group)} samples, "
                f"needs at least {train_count + test_count}."
            )

        shuffled = group.sample(frac=1.0, random_state=rng.integers(0, 1_000_000))

        selected_train = shuffled.iloc[:train_count]
        selected_test = shuffled.iloc[train_count : train_count + test_count]

        train_parts.append(selected_train)
        test_parts.append(selected_test)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split database/db.csv into class-balanced train/test CSV files."
        )
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=110,
        help="Number of samples per class for the training split (default: 110).",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=15,
        help="Number of samples per class for the test split (default: 15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=DB_CSV,
        help="Path to the CSV containing all images (default: database/db.csv).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = split_dataset(
        args.db, args.train_count, args.test_count, args.seed
    )

    train_path = args.db.parent / "db_train.csv"
    test_path = args.db.parent / "db_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"Saved {len(train_df)} train samples to {train_path} "
        f"and {len(test_df)} test samples to {test_path}."
    )


if __name__ == "__main__":
    main()
