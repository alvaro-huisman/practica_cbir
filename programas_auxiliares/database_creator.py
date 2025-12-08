import csv, pathlib

root = pathlib.Path("dataset")
with open("database/db.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])
    for class_dir in sorted(root.iterdir()):
        if class_dir.is_dir():
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    rel_path = image_path.as_posix()
                    writer.writerow([rel_path, class_dir.name])
