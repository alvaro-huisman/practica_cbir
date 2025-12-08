import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
DB_DIR = ROOT / "database"

TRAIN_CSV = DB_DIR / "db_train.csv"
TEST_CSV = DB_DIR / "db_test.csv"
FULL_CSV = DB_DIR / "db.csv"


def _process_split(split_name: str, csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'image' and 'label' columns.")

    new_rows = []
    for _, row in df.iterrows():
        rel_path = Path(row["image"])
        label = row["label"]
        filename = rel_path.name

        src = ROOT / rel_path
        dest = DATASET_DIR / label / split_name / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if src.resolve() != dest.resolve():
            if not src.exists():
                raise FileNotFoundError(f"Source image not found: {src}")
            shutil.move(str(src), str(dest))

        new_rel = dest.relative_to(ROOT).as_posix()
        new_rows.append((new_rel, label))

    new_df = pd.DataFrame(new_rows, columns=["image", "label"])
    new_df.to_csv(csv_path, index=False)
    return new_df


def main():
    train_df = _process_split("train", TRAIN_CSV)
    test_df = _process_split("test", TEST_CSV)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df.to_csv(FULL_CSV, index=False)
    print(f"Updated train/test CSVs and regenerated {FULL_CSV}")


if __name__ == "__main__":
    main()
