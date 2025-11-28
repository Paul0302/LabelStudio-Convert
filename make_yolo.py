import os
import random
import shutil
from pathlib import Path

# ======= Setting =======
ROOT_DIR = Path("output")   # Your original data root directory
OUT_DIR = Path("DataSet/YOLO")     # Output 5-fold directory
NUM_FOLDS = 5

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # Required image extensions
RANDOM_SEED = 42
# ======================


def collect_yolo_pairs(root_dir: Path):
    """
    For this structure:
    output/
      seq1/
        images/
        labels/
      seq2/
        images/
        labels/
    Collect (img_path, label_path, rel_path).
    rel_path is the path relative to images/, used to keep the correspondence between images/ and labels/ when copying later.
    """
    pairs = []

    # Scan each subfolder under output/
    for sub in root_dir.iterdir():
        if not sub.is_dir():
            continue

        img_root = sub / "images"
        lbl_root = sub / "labels"

        if not img_root.exists() or not lbl_root.exists():
            # If this subfolder is not a standard YOLO structure, skip it
            print(f"[Warning] {sub} does not have images/ or labels/, skipping")
            continue

        # Recursively scan all images under images/
        for dirpath, _, filenames in os.walk(img_root):
            dirpath = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in IMG_EXTS:
                    continue

                img_path = dirpath / fname
                # Path relative to images/, e.g., "abc/xyz.jpg"
                rel_path = img_path.relative_to(img_root)

                # Corresponding label should be under labels/ with the same path, but with .txt extension
                label_path = (lbl_root / rel_path).with_suffix(".txt")

                if label_path.exists():
                    pairs.append((img_path, label_path, rel_path))
                else:
                    print(f"[Warning] Label file not found: {label_path}")

    return pairs


def split_counts(n, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def copy_pair(img_path: Path, label_path: Path, rel_path: Path,
              dst_img_root: Path, dst_lbl_root: Path):
    """
    Copy a pair of (img, label) to the target location, keeping the relative paths between images/ and labels/ consistent.
    For example:
      rel_path = "seq1/abc.jpg"
      images/train/seq1/abc.jpg
      labels/train/seq1/abc.txt
    """
    dst_img = dst_img_root / rel_path
    dst_lbl = (dst_lbl_root / rel_path).with_suffix(".txt")

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, dst_img)
    shutil.copy2(label_path, dst_lbl)


def main():
    pairs = collect_yolo_pairs(ROOT_DIR)
    print(f"Total found {len(pairs)} (image, label) pairs")

    if not pairs:
        print("No data found, please check ROOT_DIR setting and folder structure (should be output/xxx/images + labels).")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n=== Creating fold {fold} ===")
        rng = random.Random(RANDOM_SEED + fold)
        rng.shuffle(pairs)

        n = len(pairs)
        n_train, n_val, n_test = split_counts(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:n_train + n_val]
        test_pairs  = pairs[n_train + n_val:]

        print(f"fold {fold} - train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

        fold_root = OUT_DIR / f"fold_{fold}"

        # Typical YOLO structure
        img_train_root = fold_root / "images" / "train"
        img_val_root   = fold_root / "images" / "val"
        img_test_root  = fold_root / "images" / "test"

        lbl_train_root = fold_root / "labels" / "train"
        lbl_val_root   = fold_root / "labels" / "val"
        lbl_test_root  = fold_root / "labels" / "test"

        # train
        for img_path, label_path, rel_path in train_pairs:
            copy_pair(img_path, label_path, rel_path, img_train_root, lbl_train_root)

        # val
        for img_path, label_path, rel_path in val_pairs:
            copy_pair(img_path, label_path, rel_path, img_val_root, lbl_val_root)

        # test
        for img_path, label_path, rel_path in test_pairs:
            copy_pair(img_path, label_path, rel_path, img_test_root, lbl_test_root)

        print(f"fold {fold} completed, data stored in: {fold_root}")

    print("\nAll 5 folds created!")


if __name__ == "__main__":
    main()
