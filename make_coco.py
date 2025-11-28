import json
from pathlib import Path
import argparse
import shutil
import random


def find_annotation_file(split_dir: Path) -> Path | None:
    """
    嘗試在 split 資料夾裡找到標註檔。
    依序嘗試的檔名：
      - annotation.json
      - annotations.json
      - annotations_train.json / annotations_test.json
    """
    candidates = [
        split_dir / "annotation.json",
        split_dir / "annotations.json",
        split_dir / "annotations_train.json",
        split_dir / "annotations_test.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def merge_coco_kfold(
    input_root: str | Path,
    output_root: str | Path,
    seed: int = 42,
):
    """
    input_root: coco/ （底下有多個子資料夾，每個子資料夾有 5 個 fold，各 fold 下有 train/test）
    output_root: coco_merged/ （會依 fold 合併成 5 個 train+test）
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"找不到輸入根目錄: {input_root}")

    # 1. 找出所有「子資料夾」：例如 video_1, video_2, ...
    sub_datasets = [d for d in input_root.iterdir() if d.is_dir()]
    if not sub_datasets:
        raise RuntimeError(f"{input_root} 底下沒有任何子資料夾可用。")

    print("[INFO] 發現以下子資料夾（視為不同影片資料集）:")
    for d in sub_datasets:
        print("   -", d.name)

    # 2. 假設所有 sub_dataset 的 fold 結構一樣，先從第一個抓 fold 名稱
    first_sub = sub_datasets[0]
    fold_dirs = [d for d in first_sub.iterdir() if d.is_dir()]
    if not fold_dirs:
        raise RuntimeError(f"{first_sub} 底下沒找到任何 fold 資料夾。")

    fold_names = sorted([d.name for d in fold_dirs])
    print("[INFO] 偵測到 fold 名稱:", fold_names)

    random.seed(seed)

    # 3. 依 fold 名稱逐一處理
    for fold_name in fold_names:
        print(f"\n[INFO] 處理 fold: {fold_name}")

        # 對 train / test 兩個 split 分別做合併
        for split in ["train", "test"]:
            print(f"  [INFO] 合併 split: {split}")

            # COCO 結構
            merged_images = []
            merged_annotations = []
            merged_categories = None  # 直接用第一個讀到的

            image_id_counter = 1
            ann_id_counter = 1

            # 用來避免同一張 image 重複加入（理論上不會發生）
            seen_new_file_names = set()

            # 這個 fold+split 的輸出資料夾
            out_split_dir = output_root / fold_name / split
            out_img_dir = out_split_dir / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)

            # 4. 逐個 sub_dataset 合併
            for sub_ds in sub_datasets:
                # 假設結構： input_root/sub_ds/fold_name/split/
                split_dir = sub_ds / fold_name / split
                if not split_dir.exists():
                    print(f"    [WARN] {split_dir} 不存在，略過。")
                    continue

                ann_path = find_annotation_file(split_dir)
                if ann_path is None:
                    print(f"    [WARN] {split_dir} 找不到 annotation.json 類型檔案，略過。")
                    continue

                print(f"    [INFO] 從 {ann_path} 讀取 COCO 標註")

                with ann_path.open("r", encoding="utf-8") as f:
                    coco = json.load(f)

                images = coco.get("images", [])
                annotations = coco.get("annotations", [])
                categories = coco.get("categories", [])

                if merged_categories is None:
                    merged_categories = categories
                else:
                    # 簡單檢查一下類別是否一致（可選）
                    if len(merged_categories) != len(categories):
                        print("    [WARN] 類別數量不同，請確定所有子資料集的 categories 一致。")

                # 建一個 map：舊 image_id -> 新 image_id
                id_map = {}

                # 來源影像資料夾
                src_img_dir = split_dir / "images"

                for img in images:
                    old_id = img["id"]
                    old_file_name = img["file_name"]

                    # 為了避免不同子資料夾的檔名撞在一起，加上前綴
                    new_file_name = f"{sub_ds.name}___{old_file_name}"

                    # 避免重複
                    if new_file_name in seen_new_file_names:
                        # 理論上不該發生，如果發生就略過或改名
                        print(f"    [WARN] 合併時檔名衝突: {new_file_name}，略過這張影像。")
                        continue

                    seen_new_file_names.add(new_file_name)

                    new_img = dict(img)
                    new_img["id"] = image_id_counter
                    new_img["file_name"] = new_file_name

                    merged_images.append(new_img)
                    id_map[old_id] = image_id_counter

                    # 複製影像檔
                    src_img_path = src_img_dir / old_file_name
                    dst_img_path = out_img_dir / new_file_name
                    if src_img_path.exists():
                        shutil.copy2(src_img_path, dst_img_path)
                    else:
                        print(f"    [WARN] 找不到影像檔 {src_img_path}，略過複製。")

                    image_id_counter += 1

                # 處理 annotations
                for ann in annotations:
                    old_img_id = ann["image_id"]
                    if old_img_id not in id_map:
                        # 這張 annotation 對應的 image 沒有成功加入
                        continue

                    new_ann = dict(ann)
                    new_ann["id"] = ann_id_counter
                    new_ann["image_id"] = id_map[old_img_id]

                    merged_annotations.append(new_ann)
                    ann_id_counter += 1

            # 如果這個 split 完全沒資料，就略過
            if not merged_images:
                print(f"  [WARN] fold={fold_name}, split={split} 沒有任何影像被合併，略過輸出。")
                continue

            # 5. 寫出合併後的 COCO JSON
            merged_coco = {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": merged_categories if merged_categories is not None else [],
            }

            out_ann_path = out_split_dir / f"annotations_{split}.json"
            with out_ann_path.open("w", encoding="utf-8") as f:
                json.dump(merged_coco, f, ensure_ascii=False, indent=2)

            print(f"  [INFO] fold={fold_name}, split={split} 合併完成：")
            print(f"         images = {len(merged_images)}, annotations = {len(merged_annotations)}")
            print(f"         輸出影像資料夾: {out_img_dir}")
            print(f"         輸出標註檔: {out_ann_path}")

    print("\n[INFO] 所有 fold 的 train/test 合併完成！")


def main():
    parser = argparse.ArgumentParser(
        description="合併多個子資料夾底下的 k-fold COCO 資料集，依 fold 產生 5 組 train+test。"
    )
    parser.add_argument(
        "--input-root",
        default="./DataSet/COCO",
        help="原始 COCO k-fold 根目錄，例如 ./coco",
    )
    parser.add_argument(
        "--output-root",
        default="./DataSet/Dome_Train_Test",
        help="合併後的輸出根目錄，例如 ./coco_merged",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子（目前只用於潛在隨機處理的地方，這裡其實主要是 deterministic）。",
    )

    args = parser.parse_args()
    merge_coco_kfold(
        input_root=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
