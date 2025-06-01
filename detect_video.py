import argparse
import os
import json
import cv2
from ultralytics import YOLO

def detect_objects(frame, model, conf_thresh=0.3):
    """
    Perform inference on a single frame, filter out low-confidence detections,
    and ignore “person” boxes whose area is below a certain fraction of the image area.

    Returns a list of dicts, each containing:
      {
        'class': str,      # class name
        'conf':  float,    # confidence score
        'bbox':  [x1, y1, x2, y2]
      }
    """
    results = model(frame)[0]
    detections = []

    img_h, img_w = frame.shape[:2]
    img_area = img_h * img_w

    # Define the minimum box-area ratio (relative to whole image) for “person”
    MIN_PERSON_AREA_RATIO = 0.01  # e.g., 1% of image area

    boxes = results.boxes.xyxy
    classes = results.boxes.cls
    confs = results.boxes.conf

    for box, cls, conf in zip(boxes, classes, confs):
        conf = float(conf)
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        class_name = results.names[int(cls)]

        # Compute width, height, and area of this bounding box
        width = x2 - x1
        height = y2 - y1
        box_area = width * height

        # If this is a "person" and the box area is too small relative to image, skip it
        if class_name == 'person':
            if (box_area / img_area) < MIN_PERSON_AREA_RATIO:
                continue

        detections.append({
            'class': class_name,
            'conf':  conf,
            'bbox':  [x1, y1, x2, y2]
        })

    return detections

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8x on a video and save detections to JSON.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (e.g., /path/to/Scene_1.mp4)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where frames/ and content/ folders will be created"
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Save every Nth frame (default: 5)"
    )

    args = parser.parse_args()
    video_path = args.video
    output_base = args.output_dir
    conf_thresh = args.conf_thresh
    frame_interval = args.frame_interval

    # Verify input video exists
    if not os.path.isfile(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return

    # Load YOLOv8x (detection-only) model
    model = YOLO('yolov8x.pt')

    # Create output directories: <output_base>/frames/ and <output_base>/content/
    frames_dir = os.path.join(output_base, "frames")
    content_dir = os.path.join(output_base, "content")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    detections = {}
    frame_idx = 0

    print(f"\n=== Processing video: {video_path} ===")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detection on this frame
        objs = detect_objects(frame, model, conf_thresh=conf_thresh)

        # Save every Nth frame and record detections
        if frame_idx % frame_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, fname), frame)
            detections[fname] = objs

    cap.release()

    # Write all detections to JSON
    json_path = os.path.join(content_dir, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(detections)} frames and detections to:\n"
          f"  Frames → {frames_dir}\n"
          f"  JSON   → {json_path}")

if __name__ == "__main__":
    main()
