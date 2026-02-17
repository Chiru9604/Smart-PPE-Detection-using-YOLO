"""
End-to-end safety vest and helmet detection on video.
Reads video from videos/, runs person + safety model, classifies SAFE / PARTIAL / UNSAFE,
writes annotated video to outputs/, saves evidence (snapshots + CSV) for UNSAFE/PARTIAL.
Model classes: 0 = No Vest, 1 = Helmet, 2 = Vest.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .evidence import EvidenceManager


# Status and colors (BGR)
SAFE, PARTIAL, UNSAFE = "SAFE", "PARTIAL", "UNSAFE"
COLOR_SAFE = (0, 255, 0)      # Green
COLOR_PARTIAL_VEST = (0, 255, 255)   # Yellow
COLOR_PARTIAL_HELMET = (255, 255, 0) # Cyan
COLOR_UNSAFE = (0, 0, 255)    # Red

PERSON_MODEL = "yolov8s.pt"
SAFETY_MODEL_REL = "weights/best.pt"
DETECTION_CONFIDENCE = 0.45
PERSON_PADDING = 15
HELMET_DEBOUNCE_SECONDS = 3.0
SHOW_PREVIEW = False
OUTPUT_FPS_MIN, OUTPUT_FPS_MAX = 24.0, 60.0
OUTPUT_FPS_DEFAULT = 25.0
PROCESS_EVERY_N_FRAMES = 2   # Every 2nd frame: faster, small accuracy trade-off
PERSON_IMG_SIZE = 640        # Higher res for better person/helmet detection
SAFETY_IMG_SIZE = 640        # Higher res for better helmet/vest detection


def get_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_safety_model_path(root: Path) -> Path | None:
    env_path = os.environ.get("MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
    default = root / SAFETY_MODEL_REL
    return default if default.exists() else None


def get_video_path(root: Path) -> Path | None:
    default = root / "videos" / "input.mp4"
    if default.exists():
        return default
    videos_dir = root / "videos"
    if videos_dir.exists():
        mp4s = sorted(videos_dir.glob("*.mp4"))
        if mp4s:
            return mp4s[0]
    return default


def classify_person(crop, safety_model, conf_thresh: float, imgsz: int | None = None, half: bool = False) -> tuple[str, float, bool, bool]:
    """
    Run safety model on person crop. Returns (status, confidence, has_helmet, has_vest).
    Classes: 0 = No Vest, 1 = Helmet, 2 = Vest.
    imgsz: override SAFETY_IMG_SIZE for live/speed. half: FP16 inference when True.
    """
    if crop.size == 0 or min(crop.shape[:2]) < 20:
        return UNSAFE, 0.0, False, False
    sz = imgsz if imgsz is not None else SAFETY_IMG_SIZE
    kwargs = {"conf": conf_thresh, "imgsz": sz, "verbose": False}
    if half:
        kwargs["half"] = True
    results = safety_model(crop, **kwargs)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return UNSAFE, 0.0, False, False
    has_helmet = False
    has_vest = False
    max_conf = 0.0
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())
        max_conf = max(max_conf, conf)
        if cls_id == 1:
            has_helmet = True
        elif cls_id == 2:
            has_vest = True
    if has_helmet and has_vest:
        return SAFE, max_conf, True, True
    if has_helmet or has_vest:
        return PARTIAL, max_conf, has_helmet, has_vest
    return UNSAFE, max_conf, False, False


def process_frame(
    frame,
    person_model,
    safety_model,
    conf_thresh: float,
    person_imgsz: int | None = None,
    safety_imgsz: int | None = None,
    half: bool = False,
) -> list[dict]:
    """Detect persons, run safety model per crop, return list of {box, status, conf, has_helmet, has_vest}."""
    p_sz = person_imgsz if person_imgsz is not None else PERSON_IMG_SIZE
    s_sz = safety_imgsz if safety_imgsz is not None else SAFETY_IMG_SIZE
    kwargs = {"classes": [0], "conf": conf_thresh, "imgsz": p_sz, "verbose": False}
    if half:
        kwargs["half"] = True
    person_results = person_model(frame, **kwargs)
    if person_results[0].boxes is None or len(person_results[0].boxes) == 0:
        return []
    person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
    out = []
    for box in person_boxes:
        x1, y1, x2, y2 = box
        pw, ph = x2 - x1, y2 - y1
        pad_x = max(10, int(pw * 0.12))
        pad_y = max(10, int(ph * 0.12))
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(frame.shape[1], x2 + pad_x)
        py2 = min(frame.shape[0], y2 + pad_y)
        crop = frame[py1:py2, px1:px2]
        status, conf, has_helmet, has_vest = classify_person(crop, safety_model, conf_thresh, imgsz=s_sz, half=half)
        out.append({
            "box": (x1, y1, x2, y2),
            "status": status,
            "conf": conf,
            "has_helmet": has_helmet,
            "has_vest": has_vest,
        })
    return out


def draw_results(frame, results: list[dict], fps: float) -> None:
    for r in results:
        x1, y1, x2, y2 = r["box"]
        status = r["status"]
        conf = r["conf"]
        if status == SAFE:
            color = COLOR_SAFE
            label = f"SAFE {conf:.2f}"
        elif status == PARTIAL:
            if r["has_helmet"]:
                color = COLOR_PARTIAL_HELMET
                label = f"PARTIAL (Helmet) {conf:.2f}"
            else:
                color = COLOR_PARTIAL_VEST
                label = f"PARTIAL (Vest) {conf:.2f}"
        else:
            color = COLOR_UNSAFE
            label = f"UNSAFE {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    safe_n = sum(1 for r in results if r["status"] == SAFE)
    part_n = sum(1 for r in results if r["status"] == PARTIAL)
    unsafe_n = sum(1 for r in results if r["status"] == UNSAFE)
    cv2.putText(frame, f"SAFE: {safe_n} | PARTIAL: {part_n} | UNSAFE: {unsafe_n}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def run_pipeline(
    video_path: Path,
    output_path: Path,
    root: Path | None = None,
    frame_callback=None,
) -> int:
    """
    Run detection on a single video and write annotated output.
    frame_callback(annotated_frame_bgr) is called after each annotated frame if set (e.g. for live stream).
    Returns 0 on success, 1 on error. Used by CLI and web backend.
    """
    root = root or get_root()
    safety_path = get_safety_model_path(root)
    if safety_path is None:
        print("Error: Safety model not found. Set MODEL_PATH or place weights/best.pt in project.", file=sys.stderr)
        return 1
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_manager = EvidenceManager(root)

    print(f"Loading person model: {PERSON_MODEL}")
    person_model = YOLO(PERSON_MODEL)
    print(f"Loading safety model: {safety_path}")
    safety_model = YOLO(str(safety_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}", file=sys.stderr)
        print("Place a video in videos/ (e.g. input.mp4).", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = OUTPUT_FPS_DEFAULT
    if input_fps and input_fps > 0 and input_fps >= OUTPUT_FPS_MIN:
        fps = min(OUTPUT_FPS_MAX, max(OUTPUT_FPS_MIN, float(input_fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"Error: Could not create output: {output_path}", file=sys.stderr)
        cap.release()
        return 1

    print(f"Processing: {video_path}")
    print(f"Output: {output_path} (FPS: {fps:.1f})")
    frame_idx = 0
    last_violation_time = 0.0
    fps_counter = [0.0]
    t_prev = time.perf_counter()
    last_results: list[dict] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_curr = time.perf_counter()
            fps_counter[0] = 1.0 / (t_curr - t_prev) if t_prev > 0 else 0.0
            t_prev = t_curr

            if frame_idx % PROCESS_EVERY_N_FRAMES == 0 or not last_results:
                last_results = process_frame(frame, person_model, safety_model, DETECTION_CONFIDENCE)
            results = last_results
            annotated = frame.copy()
            draw_results(annotated, results, fps_counter[0])

            unsafe_any = any(r["status"] == UNSAFE for r in results)
            partial_any = any(r["status"] == PARTIAL for r in results)
            if (unsafe_any or partial_any) and (time.perf_counter() - last_violation_time) >= HELMET_DEBOUNCE_SECONDS:
                last_violation_time = time.perf_counter()
                vio_type = "unsafe" if unsafe_any else "partial"
                best = max(results, key=lambda r: r["conf"])
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                evidence_manager.save_snapshot(frame, vio_type, best["conf"], frame_idx)
                evidence_manager.log_violation(ts, vio_type, best["conf"], frame_idx)
                print(f"  [{vio_type.upper()}] snapshot + log at frame {frame_idx}")

            out.write(annotated)
            if frame_callback is not None:
                frame_callback(annotated)
            if SHOW_PREVIEW:
                cv2.imshow("Safety Vest & Helmet Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"  Frame {frame_idx}/{total_frames}")
    finally:
        cap.release()
        out.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
    print(f"Done. Annotated video saved to: {output_path}")
    return 0


def run_live_capture(capture_source, person_model, safety_model, frame_callback, stop_event) -> None:
    """
    Run detection on a live source (camera index or RTSP URL).
    Calls frame_callback(annotated_frame_bgr) for each frame until stop_event is set.
    No evidence saving; models must be pre-loaded and passed in.
    """
    cap = cv2.VideoCapture(capture_source)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        return
    fps_counter = [0.0]
    t_prev = time.perf_counter()
    last_results: list[dict] = []
    frame_idx = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            t_curr = time.perf_counter()
            fps_counter[0] = 1.0 / (t_curr - t_prev) if t_prev > 0 else 0.0
            t_prev = t_curr
            if frame_idx % PROCESS_EVERY_N_FRAMES == 0 or not last_results:
                last_results = process_frame(frame, person_model, safety_model, DETECTION_CONFIDENCE)
            results = last_results
            annotated = frame.copy()
            draw_results(annotated, results, fps_counter[0])
            try:
                frame_callback(annotated)
            except Exception:
                pass
            frame_idx += 1
    finally:
        cap.release()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Safety Vest & Helmet Detection on video")
    parser.add_argument("--video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, help="Output annotated video path")
    args = parser.parse_args()
    root = get_root()
    safety_path = get_safety_model_path(root)
    if safety_path is None:
        print("Error: Safety model not found. Set MODEL_PATH or place weights/best.pt in project.", file=sys.stderr)
        return 1
    if args.video and args.output:
        return run_pipeline(args.video, args.output, root)
    video_path = get_video_path(root)
    output_path = root / "outputs" / "annotated.mp4"
    if not video_path or not video_path.exists():
        print("Error: No video found. Place a video in videos/ or use --video.", file=sys.stderr)
        return 1
    return run_pipeline(video_path, output_path, root)


if __name__ == "__main__":
    sys.exit(main())
