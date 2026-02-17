"""
CCTV-grade real-time PPE detection pipeline.
- Threaded frame capture with bounded queue; drop old frames when processing lags.
- FP16 + CUDA, warmup, YOLOv8n for speed, lower resolution for live.
- Inference time and FPS monitoring.
"""

import os
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from ultralytics import YOLO

from .main import DETECTION_CONFIDENCE, draw_results, get_safety_model_path, process_frame

# Live-specific config: speed over accuracy
LIVE_PERSON_MODEL = "yolov8n.pt"
LIVE_IMG_SIZE = 512
LIVE_PROCESS_EVERY_N_FRAMES = 2
LIVE_USE_FP16 = True
LIVE_WARMUP_ITERS = 50


def _use_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _open_capture(capture_source):
    """
    Open capture with backend fallbacks.
    On Windows webcams, CAP_DSHOW is often more stable than MSMF.
    """
    candidates = []
    if isinstance(capture_source, int):
        if os.name == "nt":
            if hasattr(cv2, "CAP_DSHOW"):
                candidates.append((capture_source, cv2.CAP_DSHOW))
            if hasattr(cv2, "CAP_MSMF"):
                candidates.append((capture_source, cv2.CAP_MSMF))
        candidates.append((capture_source, None))
    else:
        candidates.append((capture_source, None))

    for src, backend in candidates:
        try:
            cap = cv2.VideoCapture(src, backend) if backend is not None else cv2.VideoCapture(src)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            return cap
        try:
            cap.release()
        except Exception:
            pass
    return None


def capture_loop(capture_source, frame_queue: Queue, stop_event: threading.Event) -> None:
    """
    Read frames from camera/RTSP and put the latest into a size-1 queue.
    When queue is full, drop the old frame and put the new one (always process latest).
    """
    cap = _open_capture(capture_source)
    if cap is None:
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
            try:
                frame_queue.put_nowait(frame)
            except Exception:
                pass
    finally:
        cap.release()


def inference_loop(
    frame_queue: Queue,
    person_model,
    safety_model,
    frame_callback,
    stop_event: threading.Event,
    metrics: dict,
    metrics_lock: threading.Lock,
) -> None:
    """
    Pull latest frame from queue, run detection, draw, callback.
    This uses the original frame-level PPE path to preserve accuracy.
    """
    use_fp16 = LIVE_USE_FP16 and _use_cuda()
    fps_counter = [0.0]
    t_prev = time.perf_counter()
    last_results: list = []
    frame_idx = 0

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except Empty:
            continue

        t_start = time.perf_counter()
        if frame_idx % LIVE_PROCESS_EVERY_N_FRAMES == 0 or not last_results:
            last_results = process_frame(
                frame,
                person_model,
                safety_model,
                DETECTION_CONFIDENCE,
                person_imgsz=LIVE_IMG_SIZE,
                safety_imgsz=LIVE_IMG_SIZE,
                half=use_fp16,
            )
        results = last_results
        annotated = frame.copy()
        draw_results(annotated, results, fps_counter[0])
        inference_ms = (time.perf_counter() - t_start) * 1000

        t_curr = time.perf_counter()
        fps_counter[0] = 1.0 / (t_curr - t_prev) if t_prev > 0 else 0.0
        t_prev = t_curr

        with metrics_lock:
            metrics["fps"] = round(fps_counter[0], 1)
            metrics["inference_ms"] = round(inference_ms, 1)
            metrics["detect_every_n"] = LIVE_PROCESS_EVERY_N_FRAMES
            metrics["tracks"] = 0
            metrics["detected"] = len(results)

        try:
            frame_callback(annotated)
        except Exception:
            pass
        frame_idx += 1


def load_live_models(root: Path):
    """
    Load YOLOv8n (person) and safety model for live; optional FP16 and warmup.
    Returns (person_model, safety_model).
    """
    safety_path = get_safety_model_path(root)
    if safety_path is None or not safety_path.exists():
        raise FileNotFoundError("Safety model not found (weights/best.pt)")
    person_model = YOLO(LIVE_PERSON_MODEL)
    safety_model = YOLO(str(safety_path))
    use_fp16 = LIVE_USE_FP16 and _use_cuda()
    # Warmup with the same path used in live inference.
    try:
        import torch

        dummy = np.zeros((LIVE_IMG_SIZE, LIVE_IMG_SIZE, 3), dtype=np.uint8)
        for _ in range(LIVE_WARMUP_ITERS):
            process_frame(
                dummy,
                person_model,
                safety_model,
                DETECTION_CONFIDENCE,
                person_imgsz=LIVE_IMG_SIZE,
                safety_imgsz=LIVE_IMG_SIZE,
                half=use_fp16,
            )
        if _use_cuda():
            torch.cuda.synchronize()
    except Exception:
        pass
    return person_model, safety_model


def run_cctv_pipeline(
    capture_source,
    person_model,
    safety_model,
    frame_callback,
    stop_event: threading.Event,
    metrics: dict,
    metrics_lock: threading.Lock,
):
    """
    Start capture thread and inference thread; return (capture_thread, inference_thread).
    Caller must join both threads on stop.
    """
    frame_queue = Queue(maxsize=1)
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(capture_source, frame_queue, stop_event),
        daemon=True,
    )
    inference_thread = threading.Thread(
        target=inference_loop,
        args=(frame_queue, person_model, safety_model, frame_callback, stop_event, metrics, metrics_lock),
        daemon=True,
    )
    capture_thread.start()
    inference_thread.start()
    return capture_thread, inference_thread
