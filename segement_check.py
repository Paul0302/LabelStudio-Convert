import cv2
import os
from pathlib import Path
import argparse


def load_yolo_labels(txt_path, img_w, img_h):
    """
    Read YOLO txt annotations and convert them to a list of (x1, y1, x2, y2, class_id) in pixel coordinates.
    Format: class cx cy w h (all normalized between 0 and 1)
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
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        print(f"Warning: Labels directory not found: {labels_dir}, will output video without boxes")

    # Collect all image files   
    img_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not img_files:
        raise RuntimeError(f"No jpg/png images found in {images_dir}.")

    # Read the first image to determine video size
    sample = cv2.imread(str(img_files[0]))
    if sample is None:
        raise RuntimeError(f"Cannot read image: {img_files[0]}")
    h, w = sample.shape[:2]

    # Create VideoWriter
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    total = len(img_files)
    print(f"Total {total} frames, outputting video to: {output_video}")
    print(f"Video size: {w}x{h}, FPS: {fps}")

    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Cannot read {img_path}, skipping.")
            continue

        # Corresponding txt filename (frame_000001.jpg → frame_000001.txt)
        txt_path = labels_dir / (img_path.stem + ".txt")

        boxes = load_yolo_labels(txt_path, img_w=w, img_h=h)

        # Draw boxes
        for (x1, y1, x2, y2, cls) in boxes:
            # Draw box
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
            print(f"Written {idx}/{total} frames")

    writer.release()
    print("Video output completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine YOLO annotated frames and txt files into a video with bounding boxes for annotation verification."
    )
    parser.add_argument(
        "--images-dir",
        default=r"./output/20251031120850_0005_D/images",
        help="Directory containing frame images, e.g., ./LabelData_Yolo/xxx/images",
    )
    parser.add_argument(
        "--labels-dir",
        default=r"./output/20251031120850_0005_D/labels",
        help="Directory containing YOLO txt annotations, e.g., ./LabelData_Yolo/xxx/labels",
    )
    parser.add_argument(
        "--output-video",
        default=r"./output_video/20251031120850_0005_D/check_video.mp4",
        help="Output video file path, e.g., ./check_xxx.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default 30)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="Dog",
        help="Class name to display on the video (default Dog)",
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
