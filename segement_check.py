import cv2
import os
from pathlib import Path
import argparse


def load_yolo_labels(txt_path, img_w, img_h):
    """
    讀取單一 YOLO txt 的標註，轉成 (x1, y1, x2, y2, class_id) 的 list（pixel 座標）
    格式: class cx cy w h（都是 0~1）
    """
    boxes = []
    if not txt_path.exists():
        return boxes

    with open(txt_path, "r", encoding="utf-8") as f:
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
    images_dir,
    labels_dir,
    output_video,
    fps=30,
    draw_label=True,
    class_name="Dog",
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_video = Path(output_video)

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 資料夾: {images_dir}")
    if not labels_dir.exists():
        print(f"警告: 找不到 labels 資料夾: {labels_dir}，將只輸出影像不畫框")

    # 收集所有圖片檔案
    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not img_files:
        raise RuntimeError(f"{images_dir} 裡面沒有 jpg/png 圖片。")

    # 先讀第一張決定影像大小
    sample = cv2.imread(str(img_files[0]))
    if sample is None:
        raise RuntimeError(f"無法讀取影像: {img_files[0]}")
    h, w = sample.shape[:2]

    # 建立 VideoWriter
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    total = len(img_files)
    print(f"共 {total} 張 frame，要輸出成影片: {output_video}")
    print(f"影片尺寸: {w}x{h}, FPS: {fps}")

    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 無法讀取 {img_path}，略過。")
            continue

        # 對應的 txt 檔名（frame_000001.jpg → frame_000001.txt）
        txt_path = labels_dir / (img_path.stem + ".txt")

        boxes = load_yolo_labels(txt_path, img_w=w, img_h=h)

        # 畫框
        for (x1, y1, x2, y2, cls) in boxes:
            # 畫框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if draw_label:
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
    print("影片輸出完成！")


def main():
    parser = argparse.ArgumentParser(
        description="把 YOLO 標註的 frame + txt 再組成帶框影片，用來檢查標註是否正確。"
    )
    parser.add_argument(
        "--images-dir",
        default=r"./output/20251031120850_0005_D/images",
        help="frame 影像所在資料夾，例如 ./LabelData_Yolo/xxx/images",
    )
    parser.add_argument(
        "--labels-dir",
        default=r"./output/20251031120850_0005_D/labels",
        help="YOLO txt 標註所在資料夾，例如 ./LabelData_Yolo/xxx/labels",
    )
    parser.add_argument(
        "--output-video",
        default=r"./output_video/20251031120850_0005_D/check_video.mp4",
        help="輸出影片檔路徑，例如 ./check_xxx.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="輸出影片的 FPS（預設 30）",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="Dog",
        help="顯示在畫面上的類別名稱（預設 Dog）",
    )
    args = parser.parse_args()

    frames_to_video(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_video=args.output_video,
        fps=args.fps,
        class_name=args.class_name,
    )


if __name__ == "__main__":
    main()
