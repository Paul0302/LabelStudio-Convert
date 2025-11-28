import os
import random
import shutil
from pathlib import Path

# ======= 設定區 =======
ROOT_DIR = Path("output")   # 你的原始資料根目錄
OUT_DIR = Path("DataSet/YOLO")     # 產出的 5-fold 目錄
NUM_FOLDS = 5

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 需要的圖片副檔名
RANDOM_SEED = 42
# ======================


def collect_yolo_pairs(root_dir: Path):
    """
    針對這種結構：
    output/
      seq1/
        images/
        labels/
      seq2/
        images/
        labels/
    去收集 (img_path, label_path, rel_path)。
    rel_path 是相對於 images/ 的路徑，用來之後 copy 時保持 images/ 與 labels/ 對應。
    """
    pairs = []

    # 掃描 output/ 底下每個子資料夾
    for sub in root_dir.iterdir():
        if not sub.is_dir():
            continue

        img_root = sub / "images"
        lbl_root = sub / "labels"

        if not img_root.exists() or not lbl_root.exists():
            # 如果這個子資料夾不是標準 YOLO 結構就略過
            print(f"[提示] {sub} 沒有 images/ 或 labels/，略過")
            continue

        # 遞迴掃描 images/ 底下所有圖片
        for dirpath, _, filenames in os.walk(img_root):
            dirpath = Path(dirpath)
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in IMG_EXTS:
                    continue

                img_path = dirpath / fname
                # 相對於 images/ 的路徑，例如 "abc/xyz.jpg"
                rel_path = img_path.relative_to(img_root)

                # 對應的 label 應該在 labels/ 底下同樣路徑，只是副檔名改成 .txt
                label_path = (lbl_root / rel_path).with_suffix(".txt")

                if label_path.exists():
                    pairs.append((img_path, label_path, rel_path))
                else:
                    print(f"[警告] 找不到標註檔: {label_path}")

    return pairs


def split_counts(n, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例必須加總為 1"
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def copy_pair(img_path: Path, label_path: Path, rel_path: Path,
              dst_img_root: Path, dst_lbl_root: Path):
    """
    把一組 (img, label) 複製到目標位置，並保持 images/ 與 labels/ 的相對路徑一致。
    例如：
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
    print(f"總共找到 {len(pairs)} 筆 (image, label) pair")

    if not pairs:
        print("沒有找到任何資料，請確認 ROOT_DIR 設定與資料夾結構（是否為 output/xxx/images + labels）。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n=== 建立 fold {fold} ===")
        rng = random.Random(RANDOM_SEED + fold)
        rng.shuffle(pairs)

        n = len(pairs)
        n_train, n_val, n_test = split_counts(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        train_pairs = pairs[:n_train]
        val_pairs   = pairs[n_train:n_train + n_val]
        test_pairs  = pairs[n_train + n_val:]

        print(f"fold {fold} - train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

        fold_root = OUT_DIR / f"fold_{fold}"

        # YOLO 典型結構
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

        print(f"fold {fold} 完成，資料存放在: {fold_root}")

    print("\n全部 5 個 fold 建立完成！")


if __name__ == "__main__":
    main()
