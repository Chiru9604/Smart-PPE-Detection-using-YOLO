"""
Web backend: upload video, process with Safety Vest & Helmet pipeline, serve annotated result.
Live stream during processing + optional real-time camera/RTSP feed.
Run: uvicorn app:app --reload
"""

import os
import sys
import time
import uuid
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project root (parent of app.py)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

UPLOAD_DIR = ROOT / "web_uploads"
OUTPUT_DIR = ROOT / "web_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store: job_id -> { status, output_path?, error?, latest_frame? }
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Live camera/RTSP state: CCTV-grade pipeline (capture + inference threads, metrics)
live_state = {
    "running": False,
    "latest_frame": None,
    "latest_raw_frame": None,
    "latest_raw_seq": 0,
    "stop_event": None,
    "capture_thread": None,
    "inference_thread": None,
    "encoder_thread": None,
    "lock": threading.Lock(),
    "models": None,  # (person_model, safety_model) from load_live_models
    "metrics": {"fps": 0.0, "inference_ms": 0.0, "detect_every_n": 0, "tracks": 0, "detected": 0},
    "metrics_lock": threading.Lock(),
}

# Stream encoding settings (transport-only, does not affect detection accuracy).
DEFAULT_STREAM_JPEG_QUALITY = 85
LIVE_STREAM_JPEG_QUALITY = 50
LIVE_STREAM_MAX_WIDTH = 640


def _encode_frame_jpeg(
    frame_bgr,
    quality: int = DEFAULT_STREAM_JPEG_QUALITY,
    max_width: int | None = None,
) -> bytes | None:
    try:
        frame = frame_bgr
        if max_width is not None and max_width > 0 and frame_bgr.shape[1] > max_width:
            scale = max_width / float(frame_bgr.shape[1])
            new_h = max(1, int(frame_bgr.shape[0] * scale))
            frame = cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)
        q = int(max(1, min(100, quality)))
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return buf.tobytes() if buf is not None else None
    except Exception:
        return None


def run_job(job_id: str, video_path: Path, output_path: Path) -> None:
    def on_frame(annotated_bgr):
        jpeg = _encode_frame_jpeg(annotated_bgr)
        if jpeg is not None:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["latest_frame"] = jpeg

    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"
        from src.main import run_pipeline
        code = run_pipeline(video_path, output_path, ROOT, frame_callback=on_frame)
        with jobs_lock:
            if code == 0 and output_path.exists():
                jobs[job_id]["status"] = "done"
                jobs[job_id]["output_path"] = str(output_path)
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "Pipeline failed or output missing."
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)


app = FastAPI(title="Safety Vest & Helmet Detection", version="1.0")


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | done | failed
    download_url: str | None = None
    error: str | None = None


@app.post("/api/upload", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Only video files (mp4, avi, mov, mkv) are allowed.")
    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}_input{ext}"
    output_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "output_path": None, "error": None, "latest_frame": None}
    thread = threading.Thread(target=run_job, args=(job_id, video_path, output_path))
    thread.daemon = True
    thread.start()
    return JobStatus(
        job_id=job_id,
        status="queued",
        download_url=None,
        error=None,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
        j = jobs[job_id]
    status = j["status"]
    download_url = f"/api/output/{job_id}" if status == "done" and j.get("output_path") else None
    return JobStatus(
        job_id=job_id,
        status=status,
        download_url=download_url,
        error=j.get("error"),
    )


@app.get("/api/output/{job_id}")
def download_output(job_id: str):
    with jobs_lock:
        if job_id not in jobs or jobs[job_id]["status"] != "done":
            raise HTTPException(404, "Output not ready or job not found")
        path = jobs[job_id].get("output_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Output file missing")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"annotated_{job_id[:8]}.mp4",
    )


def _stream_mjpeg_from_job(job_id: str):
    boundary = b"frame"
    no_frame_waits = 0
    max_no_frame_waits = 100
    while True:
        with jobs_lock:
            if job_id not in jobs:
                break
            j = jobs[job_id]
            frame = j.get("latest_frame")
            status = j["status"]
        if frame:
            no_frame_waits = 0
            yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            no_frame_waits += 1
            if no_frame_waits > max_no_frame_waits:
                break
        if status in ("done", "failed"):
            break
        time.sleep(0.05)


@app.get("/api/jobs/{job_id}/stream")
def job_stream(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(404, "Job not found")
    return StreamingResponse(
        _stream_mjpeg_from_job(job_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _stream_mjpeg_from_live():
    boundary = b"frame"
    empty_waits = 0
    max_empty_waits = 300
    while True:
        with live_state["lock"]:
            running = live_state["running"]
            frame = live_state["latest_frame"]
        if frame:
            empty_waits = 0
            yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            empty_waits += 1
            if not running and empty_waits > 10:
                break
            if empty_waits > max_empty_waits:
                break
        time.sleep(0.033)
    with live_state["lock"]:
        last = live_state.get("latest_frame")
    if last:
        yield b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"


@app.get("/api/live/stream")
def live_stream():
    return StreamingResponse(
        _stream_mjpeg_from_live(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/live/frame")
def live_frame():
    """
    Return only the latest encoded frame.
    Designed for low-latency polling (prevents MJPEG buffering lag in some browsers).
    """
    with live_state["lock"]:
        running = live_state["running"]
        frame = live_state.get("latest_frame")
    no_cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    if frame:
        return Response(content=frame, media_type="image/jpeg", headers=no_cache_headers)
    if running:
        return Response(status_code=204, headers=no_cache_headers)
    raise HTTPException(404, "Live stream not running")


def _live_encode_loop(stop_event: threading.Event):
    """
    Encode only the newest annotated frame in a dedicated thread.
    This keeps inference callback light and reduces end-to-end latency jitter.
    """
    last_seq = -1
    while not stop_event.is_set():
        with live_state["lock"]:
            seq = live_state.get("latest_raw_seq", 0)
            frame = live_state.get("latest_raw_frame")
        if frame is None or seq == last_seq:
            time.sleep(0.002)
            continue
        jpeg = _encode_frame_jpeg(
            frame,
            quality=LIVE_STREAM_JPEG_QUALITY,
            max_width=LIVE_STREAM_MAX_WIDTH,
        )
        if jpeg is not None:
            with live_state["lock"]:
                live_state["latest_frame"] = jpeg
        last_seq = seq


class LiveStartBody(BaseModel):
    source: str | None = None
    camera_index: int | None = None


@app.post("/api/live/start")
def live_start(body: LiveStartBody | None = None):
    with live_state["lock"]:
        if live_state["running"]:
            raise HTTPException(400, "Live stream already running")
    source = None
    if body:
        if body.source:
            source = body.source.strip()
        elif body.camera_index is not None:
            source = int(body.camera_index)
    if source is None:
        source = 0
    try:
        from src.live import load_live_models, run_cctv_pipeline
    except ImportError as e:
        raise HTTPException(503, f"CCTV pipeline unavailable: {e}") from e
    if live_state["models"] is None:
        try:
            live_state["models"] = load_live_models(ROOT)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from e
    person_model, safety_model = live_state["models"]
    stop_event = threading.Event()
    with live_state["lock"]:
        live_state["stop_event"] = stop_event
        live_state["latest_frame"] = None
        live_state["latest_raw_frame"] = None
        live_state["latest_raw_seq"] = 0
        live_state["running"] = True
        live_state["metrics"] = {"fps": 0.0, "inference_ms": 0.0, "detect_every_n": 0, "tracks": 0, "detected": 0}

    def on_frame(annotated_bgr):
        with live_state["lock"]:
            live_state["latest_raw_frame"] = annotated_bgr
            live_state["latest_raw_seq"] = int(live_state.get("latest_raw_seq", 0)) + 1

    encoder_thread = threading.Thread(target=_live_encode_loop, args=(stop_event,), daemon=True)
    encoder_thread.start()
    live_state["encoder_thread"] = encoder_thread

    capture_thread, inference_thread = run_cctv_pipeline(
        source,
        person_model,
        safety_model,
        on_frame,
        stop_event,
        live_state["metrics"],
        live_state["metrics_lock"],
    )
    live_state["capture_thread"] = capture_thread
    live_state["inference_thread"] = inference_thread

    # Fail fast if the capture thread cannot deliver frames.
    # This avoids showing "running" while the feed stays blank.
    startup_deadline = time.time() + 5.0
    while time.time() < startup_deadline:
        with live_state["lock"]:
            frame_ready = live_state.get("latest_frame") is not None
        if frame_ready:
            return {"status": "started", "source": str(source)}
        if not capture_thread.is_alive():
            break
        time.sleep(0.1)

    stop_event.set()
    if capture_thread.is_alive():
        capture_thread.join(timeout=1.0)
    if inference_thread.is_alive():
        inference_thread.join(timeout=1.0)
    if encoder_thread.is_alive():
        encoder_thread.join(timeout=1.0)
    with live_state["lock"]:
        live_state["running"] = False
        live_state["capture_thread"] = None
        live_state["inference_thread"] = None
        live_state["encoder_thread"] = None
        live_state["latest_frame"] = None
        live_state["latest_raw_frame"] = None
        live_state["latest_raw_seq"] = 0
    raise HTTPException(
        503,
        "Unable to read frames from the selected source. "
        "Try another camera index (1 or 2), close apps using the camera, or check camera permissions.",
    )


@app.post("/api/live/stop")
def live_stop():
    with live_state["lock"]:
        if not live_state["running"]:
            return {"status": "stopped"}
        ev = live_state.get("stop_event")
        live_state["running"] = False
        cap_t = live_state.get("capture_thread")
        inf_t = live_state.get("inference_thread")
        enc_t = live_state.get("encoder_thread")
    if ev:
        ev.set()
    if cap_t and cap_t.is_alive():
        cap_t.join(timeout=2.0)
    if inf_t and inf_t.is_alive():
        inf_t.join(timeout=2.0)
    if enc_t and enc_t.is_alive():
        enc_t.join(timeout=2.0)
    with live_state["lock"]:
        live_state["encoder_thread"] = None
        live_state["latest_raw_frame"] = None
        live_state["latest_raw_seq"] = 0
    return {"status": "stopped"}


@app.get("/api/live/status")
def live_status():
    with live_state["lock"]:
        running = live_state["running"]
    return {"running": running}


@app.get("/api/live/stats")
def live_stats():
    """Return current FPS and inference time (ms) for the live stream."""
    with live_state["lock"]:
        running = live_state["running"]
    with live_state["metrics_lock"]:
        fps = live_state["metrics"].get("fps", 0.0)
        inference_ms = live_state["metrics"].get("inference_ms", 0.0)
        detect_every_n = live_state["metrics"].get("detect_every_n", 0)
        tracks = live_state["metrics"].get("tracks", 0)
        detected = live_state["metrics"].get("detected", 0)
    return {
        "running": running,
        "fps": fps,
        "inference_ms": inference_ms,
        "detect_every_n": detect_every_n,
        "tracks": tracks,
        "detected": detected,
    }


# Serve frontend
STATIC_DIR = ROOT / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    from fastapi.responses import HTMLResponse
    path = ROOT / "web" / "index.html"
    if not path.exists():
        raise HTTPException(404, "Frontend not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))
