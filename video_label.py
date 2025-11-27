import json
from pathlib import Path
import argparse
import math

import cv2


def load_task_from_json(json_path: str, task_id: int | None = None):
    """
    從 Label Studio 匯出的 JSON 讀出指定的 task。
    - 如果 JSON 是 list，多個 task：用 task_id 找
    - 如果 JSON 只有一個 task，就直接用
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        tasks = [data]
    else:
        tasks = data

    if not tasks:
        raise ValueError("JSON 裡沒有任何 task。")

    if task_id is None:
        print("未指定 task_id，使用 JSON 中的第一個 task。")
        return tasks[0]

    for t in tasks:
        if t.get("id") == task_id:
            return t

    raise ValueError(f"在 JSON 裡找不到 id={task_id} 的 task。")


def get_ls_timeline_info(task: dict):
    """
    從 task 的第一個 videorectangle 結構中取出
    framesCount、duration，並計算 Label Studio 的 time-line fps。
    """
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
    """
    從一個 Label Studio Video task 的 annotations 裡面，
    把所有 videorectangle 的 sequence 整理成「每一幀」要輸出的 YOLO box。

    這裡假設你用的是「已插值的匯出」（interpolated JSON），
    sequence 裡每一筆都是一個時間點（通常接近每幀一次）。
    """
    frame_boxes: dict[int, list[tuple[int, float, float, float, float]]] = {}

    annotations = task.get("annotations", [])
    if not annotations:
        print("警告：這個 task 沒有 annotations。")
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

            # Label Studio: x,y,w,h = 百分比（0~100）
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
    """
    從影片 + Label Studio JSON（已插值）產生 YOLO dataset：
      out_root/images/*.jpg
      out_root/labels/*.txt

    ★ 重點：用「時間對齊」方式把影片幀對映到 Label Studio 的 frame index
      - 影片 fps 可能是 ~30
      - Label Studio timeline fps 可能是 25（framesCount/duration）
      - 這裡用 t = n / video_fps → ls_frame = round(t * ls_fps) + 1 做映射
    """
    task = load_task_from_json(json_path, task_id=task_id)
    print(f"使用 task id={task.get('id')}，file_upload={task.get('file_upload')}")

    # 取得 Label Studio 的 timeline 資訊
    frames_count, duration, ls_fps = get_ls_timeline_info(task)
    if not frames_count or not duration or not ls_fps:
        print("警告：無法從 JSON 取得 framesCount/duration，將退回直接以 frame index 對齊。")
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
        print("警告：沒有任何可用的框（frame_boxes 為空）。")
    else:
        print(f"共有 {len(frame_boxes)} 個 Label Studio frame 有標註。")

    out_root = Path(out_root)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法打開影片：{video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片: fps≈{video_fps:.3f}, metadata 幀數={total_frames_meta}")

    if not video_fps or video_fps <= 1e-3:
        print("警告：OpenCV 無法取得影片 FPS，將假設 fps = Label Studio 的 ls_fps 或 30")
        if ls_fps:
            video_fps = ls_fps
        else:
            video_fps = 30.0

    frame_idx = 0  # OpenCV 的 0-based 幀索引
    decoded_frames = 0
    saved_frames = 0
    used_ls_frames = set()
    last_ls_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / video_fps  # 這一幀的時間（秒）
        frame_idx += 1
        decoded_frames += 1

        if ls_fps and frames_count:
            # 用時間對齊計算 Label Studio 的 frame index
            ls_frame_float = t * ls_fps + 1  # LS frame 通常從 1 開始
            ls_frame = int(round(ls_frame_float))
            # 限制在合法範圍
            ls_frame = max(1, min(frames_count, ls_frame))
        else:
            # 沒有 timeline 資訊時，退回簡單對齊
            ls_frame = frame_idx

        # 如果多個影片幀對到同一個 Label Studio frame，只保留第一個
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

    print(f"實際成功解碼幀數: {decoded_frames}")
    print(f"實際產生 YOLO frame 數: {saved_frames}")
    if frames_count:
        print(
            f"有標註的 Label Studio frame 數: {len(frame_boxes)}，"
            f"其中實際用到的 frame: {len(used_ls_frames)}"
        )
    print(f"影像在: {img_dir}")
    print(f"標註在: {lbl_dir}")

    return img_dir, lbl_dir


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
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 資料夾: {images_dir}")
    if not labels_dir.exists():
        print(f"警告: 找不到 labels 資料夾: {labels_dir}，將只輸出影像不畫框")

    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    if not img_files:
        raise RuntimeError(f"{images_dir} 裡面沒有 jpg/png 圖片。")

    sample = cv2.imread(str(img_files[0]))
    if sample is None:
        raise RuntimeError(f"無法讀取影像: {img_files[0]}")
    h, w = sample.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    total = len(img_files)
    print(f"預覽影片: 共 {total} 張 frame 要寫入: {output_video}")
    print(f"影片尺寸: {w}x{h}, FPS: {fps}")

    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 無法讀取 {img_path}，略過。")
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
            print(f"寫入 {idx}/{total} frames")

    writer.release()
    print("預覽影片輸出完成！")


def main():
    parser = argparse.ArgumentParser(
        description="從 Label Studio Video JSON + 影片 產生 YOLO dataset（時間對齊），並輸出帶框預覽影片。"
    )
    parser.add_argument(
        "--json",
        default=r"./project3_interpolated.json",
        help="Label Studio 匯出的 JSON 路徑（建議用 interpolate_key_frames 匯出的版本）",
    )
    parser.add_argument(
        "--video",
        default=r"./video/DJI_20251031120850_0005_D.mp4",
        help="原始影片路徑（.mp4 等），會用來擷取 frame",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=3086,
        help="要處理的 task id（例如 3086）。若不指定則用 JSON 中第一個 task。",
    )
    parser.add_argument(
        "--out-root",
        default=r"./output/20251031120850_0005_D",
        help="輸出 YOLO 資料集的根資料夾，會自動建立 images/ 和 labels/。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="預覽影片的 FPS（預設 30）。",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="設定此 flag 則不產生預覽影片，只產生 YOLO frames。",
    )

    args = parser.parse_args()

    img_dir, lbl_dir = export_video_to_yolo_frames_time_aligned(
        video_path=args.video,
        json_path=args.json,
        task_id=args.task_id,
        out_root=args.out_root,
        class_id=0,
    )

    if not args.no_preview:
        out_root = Path(args.out_root)
        preview_path = out_root / "preview_with_boxes.mp4"
        frames_to_video(
            images_dir=img_dir,
            labels_dir=lbl_dir,
            output_video=preview_path,
            fps=args.fps,
            class_name="Dog",
        )
        print(f"預覽影片路徑：{preview_path}")


if __name__ == "__main__":
    main()
