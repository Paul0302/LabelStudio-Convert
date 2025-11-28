import json
from pathlib import Path
import argparse
import shutil
import random


def find_annotation_file(split_dir: Path) -> Path | None:
    """
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
    input_root: coco/ （Every sub folder have 5 個 fold，each fold have train/test）
    output_root: coco_merged/ （train+test）
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Cant find the root: {input_root}")

    # 1. Find all "subdirectories": e.g., video_1, video_2, ...
    sub_datasets = [d for d in input_root.iterdir() if d.is_dir()]
    if not sub_datasets:
        raise RuntimeError(f"No subdirectories found under {input_root}.")

    print("[INFO] Found the following subdirectories (considered as different video datasets):")
    for d in sub_datasets:
        print("   -", d.name)

    # 2. Detect fold names (assuming the fold structure is the same for each sub-dataset)
    first_sub = sub_datasets[0]
    fold_dirs = [d for d in first_sub.iterdir() if d.is_dir()]
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {first_sub}.")
    fold_names = sorted([d.name for d in fold_dirs])
    print("[INFO] Detected fold names:", fold_names)

    random.seed(seed)

    # 3. Process each fold name
    for fold_name in fold_names:
        print(f"\n[INFO] Processing fold: {fold_name}")

        # Merge train/test splits separately
        for split in ["train", "test"]:
            print(f"  [INFO] Merging split: {split}")

            # COCO structure
            merged_images = []
            merged_annotations = []
            merged_categories = None  # Use the first one read

            image_id_counter = 1
            ann_id_counter = 1

            # To avoid adding the same image multiple times (should not happen in theory)
            seen_new_file_names = set()

            # Output directory for this fold+split
            out_split_dir = output_root / fold_name / split
            out_img_dir = out_split_dir / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)

            # 4. Merge each sub_dataset
            for sub_ds in sub_datasets:
                # Assume structure: input_root/sub_ds/fold_name/split/
                split_dir = sub_ds / fold_name / split
                if not split_dir.exists():
                    print(f"    [WARN] {split_dir} does not exist, skipping.")
                    continue

                ann_path = find_annotation_file(split_dir)
                if ann_path is None:
                    print(f"    [WARN] {split_dir} Can't find annotation.json type file, skipping.")
                    continue

                print(f"    [INFO] Reading COCO annotations from {ann_path}")

                with ann_path.open("r", encoding="utf-8") as f:
                    coco = json.load(f)

                images = coco.get("images", [])
                annotations = coco.get("annotations", [])
                categories = coco.get("categories", [])

                if merged_categories is None:
                    merged_categories = categories
                else:
                    # Simple check if categories are consistent (optional)
                    if len(merged_categories) != len(categories):
                        print("    [WARN] Different number of categories, please ensure all sub-datasets have consistent categories.")

                # Build a map: old image_id -> new image_id
                id_map = {}

                # Source image directory
                src_img_dir = split_dir / "images"

                for img in images:
                    old_id = img["id"]
                    old_file_name = img["file_name"]

                    # To avoid filename collisions from different subfolders, add a prefix
                    new_file_name = f"{sub_ds.name}___{old_file_name}"

                    # Avoid duplicates
                    if new_file_name in seen_new_file_names:
                        # This theoretically shouldn't happen; if it does, skip or rename
                        print(f"    [WARN] Filename collision during merge: {new_file_name}, skipping this image.")
                        continue

                    seen_new_file_names.add(new_file_name)

                    new_img = dict(img)
                    new_img["id"] = image_id_counter
                    new_img["file_name"] = new_file_name

                    merged_images.append(new_img)
                    id_map[old_id] = image_id_counter

                    # Copy image file
                    src_img_path = src_img_dir / old_file_name
                    dst_img_path = out_img_dir / new_file_name
                    if src_img_path.exists():
                        shutil.copy2(src_img_path, dst_img_path)
                    else:
                        print(f"    [WARN] Image file not found {src_img_path}, skipping copy.")

                    image_id_counter += 1

                # Process annotations
                for ann in annotations:
                    old_img_id = ann["image_id"]
                    if old_img_id not in id_map:
                        # This annotation's image was not successfully added, skip it
                        continue

                    new_ann = dict(ann)
                    new_ann["id"] = ann_id_counter
                    new_ann["image_id"] = id_map[old_img_id]

                    merged_annotations.append(new_ann)
                    ann_id_counter += 1

            # If this split has no data, skip it
            if not merged_images:
                print(f"  [WARN] fold={fold_name}, split={split} has no images merged, skipping output.")
                continue

            # 5. Write merged COCO JSON
            merged_coco = {
                "images": merged_images,
                "annotations": merged_annotations,
                "categories": merged_categories if merged_categories is not None else [],
            }

            out_ann_path = out_split_dir / f"annotations_{split}.json"
            with out_ann_path.open("w", encoding="utf-8") as f:
                json.dump(merged_coco, f, ensure_ascii=False, indent=2)

            print(f"  [INFO] fold={fold_name}, split={split} merge completed:")
            print(f"         images = {len(merged_images)}, annotations = {len(merged_annotations)}")
            print(f"         output image directory: {out_img_dir}")
            print(f"         output annotation file: {out_ann_path}")

    print("\n[INFO] All folds' train/test merge completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge k-fold COCO datasets from multiple subfolders, producing 5 sets of train+test per fold."
    )
    parser.add_argument(
        "--input-root",
        default="./DataSet/COCO",
        help="Original COCO k-fold root directory, e.g., ./coco",
    )
    parser.add_argument(
        "--output-root",
        default="./DataSet/Dome_Train_Test",
        help="Output root directory after merging, e.g., ./coco_merged",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (currently only used for potential random processing, mainly deterministic here).",
    )

    args = parser.parse_args()
    merge_coco_kfold(
        input_root=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
