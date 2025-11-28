import json
from pathlib import Path
import argparse
import math
import random
import shutil

import cv2


def load_task_from_json(json_path: str, task_id: int | None = None):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        tasks = [data]
    else:
        tasks = data

    if not tasks:
        raise ValueError("No any task in json file")

    if task_id is None:
        print("Undefined task_id，use the first task ID from json")
        return tasks[0]

    for t in tasks:
        if t.get("id") == task_id:
            return t

    raise ValueError(f"Can't find Task id={task_id} from json")


def get_ls_timeline_info(task: dict):
    annotations = task.get("annotations", [])
    if not annotations:
        return None, None, None

    results = annotations[0].get("result", [])
    for res in results:
        if res.get("type") != "videorectangle":
            continue
        val = res.get("value", {})
        frames_count = val.get("framesCount")
        duration = val.get("duration")
        if frames_count and duration and duration > 0:
            ls_fps = frames_count / duration
            return frames_count, duration, ls_fps

    return None, None, None


def build_frame_boxes_from_task(task: dict, class_id: int = 0):
    frame_boxes: dict[int, list[tuple[int, float, float, float, float]]] = {}

    annotations = task.get("annotations", [])
    if not annotations:
        print("Error：No annotations in current task")
        return frame_boxes

    ann = annotations[0]
    results = ann.get("result", [])

    for res in results:
        if res.get("type") != "videorectangle":
            continue

        val = res.get("value", {})
        seq = val.get("sequence", [])
        if not seq:
            continue

        for s in seq:
            if not s.get("enabled", True):
                continue
            if s.get("outside", False):
                continue

            f = int(s["frame"])
            x = float(s["x"])
            y = float(s["y"])
            w = float(s["width"])
            h = float(s["height"])

            cx = (x + w / 2.0) / 100.0
            cy = (y + h / 2.0) / 100.0
            ww = w / 100.0
            hh = h / 100.0

            frame_boxes.setdefault(f, []).append((class_id, cx, cy, ww, hh))

    return frame_boxes


def export_video_to_yolo_frames_time_aligned(
    video_path: str,
    json_path: str,
    task_id: int | None,
    out_root: str,
    class_id: int = 0,
):

    task = load_task_from_json(json_path, task_id=task_id)
    print(f"Use task id={task.get('id')}，file_upload={task.get('file_upload')}")

    # 取得 Label Studio 的 timeline 資訊
    frames_count, duration, ls_fps = get_ls_timeline_info(task)
    if not frames_count or not duration or not ls_fps:
        print("Error: Can't get framesCount/duration from JSON, fallback to direct frame index alignment.")
        frames_count = None
        duration = None
        ls_fps = None
    else:
        print(
            f"Label Studio: framesCount={frames_count}, "
            f"duration={duration:.3f}s, timeline fps≈{ls_fps:.3f}"
        )

    # 整理每一幀的框
    frame_boxes = build_frame_boxes_from_task(task, class_id=class_id)
    if not frame_boxes:
        print("Error：No any bounding box（frame_boxes is empty）。")
    else:
        print(f"Total annotation frames from label studio : {len(frame_boxes)} frames")

    out_root = Path(out_root)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: fps≈{video_fps:.3f}, metadata frame count={total_frames_meta}")
    if not video_fps or video_fps <= 1e-3:
        print("Warning: OpenCV can't get video FPS, will assume fps = Label Studio's ls_fps or 30")
        if ls_fps:
            video_fps = ls_fps
        else:
            video_fps = 30.0

    frame_idx = 0  # OpenCV's 0-based frame index
    decoded_frames = 0
    saved_frames = 0
    used_ls_frames = set()
    last_ls_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / video_fps  # time of this frame (seconds)
        frame_idx += 1
        decoded_frames += 1

        if ls_fps and frames_count:
            # Use time alignment to calculate Label Studio frame index
            ls_frame_float = t * ls_fps + 1  # LS frame usually starts from 1
            ls_frame = int(round(ls_frame_float))
            # Clamp to valid range
            ls_frame = max(1, min(frames_count, ls_frame))
        else:
            # No timeline info, fallback to simple alignment
            ls_frame = frame_idx

        # If multiple video frames map to the same Label Studio frame, keep only the first
        if ls_frame == last_ls_frame:
            continue
        last_ls_frame = ls_frame

        if ls_frame not in frame_boxes:
            continue

        name = f"frame_{ls_frame:06d}"
        img_path = img_dir / f"{name}.jpg"
        lbl_path = lbl_dir / f"{name}.txt"

        cv2.imwrite(str(img_path), frame)

        with lbl_path.open("w", encoding="utf-8") as f:
            for (cls_id, cx, cy, ww, hh) in frame_boxes[ls_frame]:
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                ww = max(0.0, min(1.0, ww))
                hh = max(0.0, min(1.0, hh))
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")

        saved_frames += 1
        used_ls_frames.add(ls_frame)

    cap.release()

    print(f"Actual decoded frames: {decoded_frames}")
    print(f"Actual YOLO frames generated: {saved_frames}")
    if frames_count:
        print(
            f"Label Studio frames with annotations: {len(frame_boxes)}, "
            f"actually used frames: {len(used_ls_frames)}"
        )
    print(f"Image path: {img_dir}")
    print(f"Labels path: {lbl_dir}")

    return img_dir, lbl_dir


def export_video_to_coco_time_aligned(
    video_path: str,
    json_path: str,
    task_id: int | None,
    out_root: str,
    category_id: int = 1,
    category_name: str = "Dog",
):
    """
    From video + Label Studio JSON (interpolated) generate COCO dataset:
      out_root/images/*.jpg
      out_root/annotations_coco.json

    Time alignment logic is the same as YOLO version:
      t = n / video_fps → ls_frame = round(t * ls_fps) + 1
    """

    # 1) Load Label Studio task
    task = load_task_from_json(json_path, task_id=task_id)
    print(f"[COCO] use task id={task.get('id')}, file_upload={task.get('file_upload')}")

    # 2) Get Label Studio timeline info
    frames_count, duration, ls_fps = get_ls_timeline_info(task)
    if not frames_count or not duration or not ls_fps:
        print("[COCO] Warning: can't get framesCount/duration from JSON, fallback to direct frame index alignment.")
        frames_count = None
        duration = None
        ls_fps = None
    else:
        print(
            f"[COCO] Label Studio: framesCount={frames_count}, "
            f"duration={duration:.3f}s, timeline fps≈{ls_fps:.3f}"
        )

    # 3) Boxes for each LS frame
    frame_boxes = build_frame_boxes_from_task(task, class_id=category_id)
    if not frame_boxes:
        print("[COCO] Warning: no available boxes (frame_boxes is empty).")

    # 4) Prepare output directory
    out_root = Path(out_root)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 5) Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[COCO] Video: fps≈{video_fps:.3f}, metadata frames={total_frames_meta}")

    if not video_fps or video_fps <= 1e-3:
        print("[COCO] Warning: OpenCV can't get video FPS, will assume fps = Label Studio's ls_fps or 30")
        if ls_fps:
            video_fps = ls_fps
        else:
            video_fps = 30.0

    # 6) COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "object",
            }
        ],
    }
    ann_id = 1
    img_id = 1

    frame_idx = 0
    decoded_frames = 0
    saved_frames = 0
    used_ls_frames = set()
    last_ls_frame = None

    # Read the first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Cannot read any frames from the video.")
    h, w = first_frame.shape[:2]
    # Because the first frame also needs to be processed, reset the cap position to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / video_fps
        frame_idx += 1
        decoded_frames += 1

        if ls_fps and frames_count:
            ls_frame_float = t * ls_fps + 1
            ls_frame = int(round(ls_frame_float))
            ls_frame = max(1, min(frames_count, ls_frame))
        else:
            ls_frame = frame_idx

        # Avoid multiple video frames matching the same LS frame
        if ls_frame == last_ls_frame:
            continue
        last_ls_frame = ls_frame

        # If this LS frame has no annotations, skip it
        if ls_frame not in frame_boxes:
            continue

        name = f"frame_{ls_frame:06d}"
        img_path = img_dir / f"{name}.jpg"
        cv2.imwrite(str(img_path), frame)

        # COCO image entry
        image_entry = {
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        }
        coco["images"].append(image_entry)

        # Corresponding bbox: convert from normalized cx,cy,w,h to x,y,w,h (pixels)
        for (cls_id, cx, cy, ww, hh) in frame_boxes[ls_frame]:
            box_w = ww * w
            box_h = hh * h
            center_x = cx * w
            center_y = cy * h

            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0

            # Simple clamp
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            box_w = max(0.0, min(float(w) - x1, box_w))
            box_h = max(0.0, min(float(h) - y1, box_h))

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [x1, y1, box_w, box_h],  # COCO: [x, y, w, h]
                "area": float(box_w * box_h),
                "iscrowd": 0,
            }
            coco["annotations"].append(ann)
            ann_id += 1

        saved_frames += 1
        used_ls_frames.add(ls_frame)
        img_id += 1

    cap.release()

    # 7) 寫出 COCO JSON
    coco_path = out_root / "annotations_coco.json"
    with coco_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[COCO] Actual decoded frames: {decoded_frames}")
    print(f"[COCO] Actual images with annotations: {saved_frames}")
    print(f"[COCO] Images at: {img_dir}")
    print(f"[COCO] COCO annotations at: {coco_path}")

    return img_dir, coco_path


def load_yolo_labels(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append((x1, y1, x2, y2, cls))
    return boxes


def frames_to_video(
    images_dir: Path,
    labels_dir: Path,
    output_video: Path,
    fps: float = 30.0,
    class_name: str = "Dog",
):
    """
    Generate a preview video with bounding boxes using YOLO txt files
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}, will output images without boxes")

    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if not img_files:
        raise RuntimeError(f"No jpg/png images found in {images_dir}.")

    sample = cv2.imread(str(img_files[0]))
    if sample is None:
        raise RuntimeError(f"Cannot read image: {img_files[0]}")
    h, w = sample.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    total = len(img_files)
    print(f"Preview video: Writing {total} frames to: {output_video}")
    print(f"Video size: {w}x{h}, FPS: {fps}")
    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue

        txt_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_labels(txt_path, img_w=w, img_h=h)

        for (x1, y1, x2, y2, cls) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name} ({cls})"
            cv2.putText(
                img,
                label_text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(img)

        if idx % 50 == 0 or idx == total:
            print(f"Writing {idx}/{total} frames")

    writer.release()
    print("Preview video (YOLO)")


def kfold_split_yolo_frames(
    out_root: str | Path,
    k: int = 5,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    out_root = Path(out_root)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"[kfold YOLO] Images directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"[kfold YOLO] Labels directory not found: {lbl_dir}")

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if not img_files:
        raise RuntimeError(f"[kfold YOLO] No images found in {img_dir}.")

    print(f"[kfold YOLO] Found {len(img_files)} images, preparing {k}-fold split.")

    n_total = len(img_files)

    total_ratio = train_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"[kfold YOLO] train_ratio + test_ratio must be 1, got {total_ratio}")

    n_train = int(round(n_total * train_ratio))
    n_test = n_total - n_train

    print(f"[kfold YOLO] Target images per fold: train={n_train}, test={n_test}")
    random.seed(seed)

    for fold in range(1, k + 1):
        print(f"[kfold YOLO] Generating fold {fold}/{k} ...")
        indices = list(range(n_total))
        random.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        assert len(train_indices) + len(test_indices) == n_total

        fold_root = out_root / f"kfold_{fold}"
        for split_name, split_indices in [
            ("train", train_indices),
            ("test", test_indices),
        ]:
            split_img_dir = fold_root / split_name / "images"
            split_lbl_dir = fold_root / split_name / "labels"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for idx in split_indices:
                src_img = img_files[idx]
                src_lbl = lbl_dir / (src_img.stem + ".txt")

                dst_img = split_img_dir / src_img.name
                dst_lbl = split_lbl_dir / (src_img.stem + ".txt")

                shutil.copy2(src_img, dst_img)

                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

        print(f"[kfold YOLO] Fold {fold} completed, output at: {fold_root}")

    print("[kfold YOLO] All folds generated!")


def kfold_split_coco(
    out_root: str | Path,
    coco_filename: str = "annotations_coco.json",
    k: int = 5,
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    對單一影片匯出的 COCO dataset 做 k-fold 切分：
      - Origin:
          out_root/images/*.jpg
          out_root/annotations_coco.json
      - Output:
          out_root/coco_kfold_1/train/images, annotations_train.json
                                test/images,  annotations_test.json
          ...
    """
    out_root = Path(out_root)
    img_dir = out_root / "images"
    coco_path = out_root / coco_filename

    if not img_dir.exists():
        raise FileNotFoundError(f"[kfold COCO] Images directory not found: {img_dir}")
    if not coco_path.exists():
        raise FileNotFoundError(f"[kfold COCO] COCO JSON not found: {coco_path}")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not images:
        raise RuntimeError("[kfold COCO] No images found in COCO JSON.")

    print(f"[kfold COCO] Found {len(images)} images to split, k={k}.")

    n_total = len(images)

    total_ratio = train_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"[kfold COCO] train_ratio + test_ratio must be 1, got {total_ratio}")

    n_train = int(round(n_total * train_ratio))
    n_test = n_total - n_train

    print(f"[kfold COCO] Target images per fold: train={n_train}, test={n_test}")

    random.seed(seed)

    # Build image_id -> annotations cache
    ann_by_img = {}
    for ann in annotations:
        img_id = ann["image_id"]
        ann_by_img.setdefault(img_id, []).append(ann)

    for fold in range(1, k + 1):
        print(f"[kfold COCO] Generating fold {fold}/{k} ...")

        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        assert len(train_indices) + len(test_indices) == n_total

        fold_root = out_root / f"coco_kfold_{fold}"

        for split_name, split_indices in [
            ("train", train_indices),
            ("test", test_indices),
        ]:
            split_img_dir = fold_root / split_name / "images"
            split_img_dir.mkdir(parents=True, exist_ok=True)

            split_images = []
            split_annotations = []

            for idx in split_indices:
                img_info = images[idx]
                split_images.append(img_info)

                img_id = img_info["id"]
                anns = ann_by_img.get(img_id, [])
                split_annotations.extend(anns)

                src_img = img_dir / img_info["file_name"]
                dst_img = split_img_dir / img_info["file_name"]
                if src_img.exists():
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"[kfold COCO] Warning: Image not found {src_img}, skipping copy.")

            split_coco = {
                "images": split_images,
                "annotations": split_annotations,
                "categories": categories,
            }

            anno_path = fold_root / split_name / f"annotations_{split_name}.json"
            anno_path.parent.mkdir(parents=True, exist_ok=True)
            with anno_path.open("w", encoding="utf-8") as f:
                json.dump(split_coco, f, ensure_ascii=False, indent=2)

            print(f"[kfold COCO] {split_name} JSON: {anno_path}")

        print(f"[kfold COCO] Fold {fold} completed, output at: {fold_root}")

    print("[kfold COCO] All folds generated!")


def frames_to_video_coco(
    images_dir: Path,
    coco_json_path: Path,
    output_video: Path,
    fps: float = 30.0,
):
    """
    Generate a preview video with bounding boxes using COCO annotations
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"[preview COCO] Images directory not found: {images_dir}")
    if not coco_json_path.exists():
        raise FileNotFoundError(f"[preview COCO] COCO JSON not found: {coco_json_path}")

    with coco_json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # category_id -> name
    cat_name = {c["id"]: c.get("name", f"cls{c['id']}") for c in categories}

    # file_name -> boxes
    boxes_by_file = {}
    for ann in annotations:
        img_id = ann["image_id"]
        # image_id -> file_name mapping
        # First build id -> file_name map
    id_to_file = {img["id"]: img["file_name"] for img in images}
    for ann in annotations:
        img_id = ann["image_id"]
        file_name = id_to_file.get(img_id)
        if not file_name:
            continue
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = bbox
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)
        cls = ann.get("category_id", 0)
        boxes_by_file.setdefault(file_name, []).append((x1, y1, x2, y2, cls))

    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if not img_files:
        raise RuntimeError(f"[preview COCO] No jpg/png images found in {images_dir}.")

    sample = cv2.imread(str(img_files[0]))
    if sample is None:
        raise RuntimeError(f"[preview COCO] Unable to read image: {img_files[0]}")
    h, w = sample.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    total = len(img_files)
    print(f"[preview COCO] Preview video: Writing {total} frames to {output_video}")
    print(f"[preview COCO] Video size: {w}x{h}, FPS: {fps}")

    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[preview COCO] Warning: Unable to read {img_path}, skipping.")
            continue

        boxes = boxes_by_file.get(img_path.name, [])

        for (x1, y1, x2, y2, cls) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label_text = cat_name.get(cls, f"cls{cls}")
            cv2.putText(
                img,
                label_text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(img)

        if idx % 50 == 0 or idx == total:
            print(f"[preview COCO] Writing {idx}/{total} frames")

    writer.release()
    print("[preview COCO]   output!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO / COCO dataset (time-aligned) from Label Studio Video JSON + video, with optional k-fold splitting and preview video."
    )
    parser.add_argument(
        "--json",
        default=r"",
        help="Path to Label Studio exported JSON (recommended to use the version exported with interpolate_key_frames).",
    )
    parser.add_argument(
        "--video",
        default=r"",
        help="Path to the original video (.mp4, etc.), used to extract frames.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=6,
        help="Task ID to process. If not specified, the first task in the JSON is used.",
    )
    parser.add_argument(
        "--out-root",
        default=r"",
        help="Output root directory for the dataset, will automatically create images/ and labels/ etc.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS for the preview video (default 30).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="If set, do not generate preview video, only generate dataset.",
    )
    parser.add_argument(
        "--export-format",
        choices=["yolo", "coco", "both"],
        default="coco",
        help="Output format: yolo / coco / both (default yolo).",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help=">0 to perform k-fold splitting on the exported dataset for this video (train:test=8:2). For example, 5 means 5-fold.",
    )
    parser.add_argument(
        "--kfold-seed",
        type=int,
        default=42,
        help="Random seed for k-fold splitting (fixed for reproducibility).",
    )

    args = parser.parse_args()

    img_dir = None
    lbl_dir = None
    coco_img_dir = None
    coco_path = None

    # YOLO output
    if args.export_format in ["yolo", "both"]:
        img_dir, lbl_dir = export_video_to_yolo_frames_time_aligned(
            video_path=args.video,
            json_path=args.json,
            task_id=args.task_id,
            out_root=args.out_root,
            class_id=0,
        )

        # YOLO k-fold
        if args.kfold > 0:
            kfold_split_yolo_frames(
                out_root=args.out_root,
                k=args.kfold,
                train_ratio=0.8,
                test_ratio=0.2,
                seed=args.kfold_seed,
            )

        # YOLO preview video
        if not args.no_preview and img_dir is not None and lbl_dir is not None:
            out_root_path = Path(args.out_root)
            preview_path = out_root_path / "preview_yolo_with_boxes.mp4"
            frames_to_video(
                images_dir=img_dir,
                labels_dir=lbl_dir,
                output_video=preview_path,
                fps=args.fps,
                class_name="Dog",
            )
            print(f"YOLO preview video path: {preview_path}")

    # COCO output
    if args.export_format in ["coco", "both"]:
        coco_img_dir, coco_path = export_video_to_coco_time_aligned(
            video_path=args.video,
            json_path=args.json,
            task_id=args.task_id,
            out_root=args.out_root,
            category_id=1,
            category_name="Dog",
        )
        print(f"COCO annotation file: {coco_path}")

        # COCO k-fold
        if args.kfold > 0:
            kfold_split_coco(
                out_root=args.out_root,
                coco_filename=coco_path.name,
                k=args.kfold,
                train_ratio=0.8,
                test_ratio=0.2,
                seed=args.kfold_seed,
            )

        # COCO preview video (using original COCO images + annotations)
        if not args.no_preview and coco_img_dir is not None and coco_path is not None:
            out_root_path = Path(args.out_root)
            preview_coco_path = out_root_path / "preview_coco_with_boxes.mp4"
            frames_to_video_coco(
                images_dir=coco_img_dir,
                coco_json_path=coco_path,
                output_video=preview_coco_path,
                fps=args.fps,
            )
            print(f"COCO preview video path: {preview_coco_path}")


if __name__ == "__main__":
    main()
