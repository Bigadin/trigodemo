import json
import time
import threading
from pathlib import Path
from queue import Queue
from collections import deque
import cv2
import numpy as np
import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import torch

# TurboJPEG for faster JPEG encoding (with OpenCV fallback)
try:
    from turbojpeg import TurboJPEG
    jpeg_encoder = TurboJPEG()
    USE_TURBOJPEG = True
    print("TurboJPEG encoder loaded!")
except (ImportError, OSError) as e:
    jpeg_encoder = None
    USE_TURBOJPEG = False
    print(f"TurboJPEG not available ({e}), using OpenCV fallback")

app = FastAPI(title="Zone Presence Tracker")

# Paths
BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

VIDEOS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("human.pt")
model_lock = threading.Lock()
print("YOLO model loaded!")

# Select device automatically (CPU if no CUDA)
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"YOLO inference device: {YOLO_DEVICE} (cuda_available={torch.cuda.is_available()})")

# Data state
zones_by_video = {}
zone_timers = {}
data_lock = threading.Lock()

# Active streams: {video_name: {"active": bool, "detections": [], "current_frame": int, "frame_lock": Lock}}
active_streams = {}
streams_lock = threading.Lock()
# Shared frames for streaming (processor writes, streamer reads)
shared_frames = {}  # {video_name: {"frame": ndarray, "frame_num": int}}

# Seuil minimum d'intersection bbox/zone (30%)
INTERSECTION_THRESHOLD = 0.30

# Floutage des visages (30% haut de la bbox)
blur_enabled = False
blur_lock = threading.Lock()

# Performance metrics storage
perf_metrics = {
    "inference_times": deque(maxlen=100),  # Last 100 inference times in ms
    "frame_times": deque(maxlen=100),      # Last 100 frame processing times
    "fps_history": deque(maxlen=60),       # FPS history for sparkline (60 samples)
    "inference_history": deque(maxlen=60), # Inference time history for sparkline
    "cpu_history": deque(maxlen=60),       # CPU usage history
    "gpu_history": deque(maxlen=60),       # GPU usage history
    "ram_history": deque(maxlen=60),       # RAM usage history
    "last_inference_time": 0,              # Last inference time in ms
    "last_frame_time": 0,                  # Last frame processing time in ms
    "current_fps": 0,                      # Current FPS
    "queue_size": 0,                       # Current detection queue size
    "total_detections": 0,                 # Total detections count
    "frames_processed": 0,                 # Total frames processed
    "batch_size_history": deque(maxlen=60),  # Batch size history
    "last_batch_size": 0,                    # Last batch size
    "batch_latency_ms": 0,                   # Batch collection latency
}
perf_lock = threading.Lock()

# Batch detection system
BATCH_MAX_SIZE = 8       # Maximum frames per batch
BATCH_TIMEOUT = 0.05     # 50ms max wait time for batch collection

batch_queue = Queue(maxsize=32)  # Centralized queue: (video_name, frame, timestamp)
detection_results = {}           # {video_name: {"detections": [], "timestamp": float}}
detection_results_lock = threading.Lock()
batch_worker_running = False

ZONES_FILE = DATA_DIR / "zones.json"
PRESENCE_FILE = DATA_DIR / "presence.json"

GRACE_PERIOD = 1.5
YOLO_CONFIDENCE = 0.45


def load_data():
    global zones_by_video, zone_timers
    if ZONES_FILE.exists():
        with open(ZONES_FILE, "r") as f:
            zones_by_video = json.load(f)
    if PRESENCE_FILE.exists():
        with open(PRESENCE_FILE, "r") as f:
            loaded = json.load(f)
            for zone_name, value in loaded.items():
                if isinstance(value, (int, float)):
                    zone_timers[zone_name] = {"total_time": value, "last_occupied": None}
                else:
                    zone_timers[zone_name] = value
                    if "last_occupied" not in zone_timers[zone_name]:
                        zone_timers[zone_name]["last_occupied"] = None


def save_zones():
    with open(ZONES_FILE, "w") as f:
        json.dump(zones_by_video, f, indent=2)


def save_presence():
    with data_lock:
        with open(PRESENCE_FILE, "w") as f:
            save_data = {k: {"total_time": v["total_time"]} for k, v in zone_timers.items()}
            json.dump(save_data, f, indent=2)


load_data()


def _list_video_files() -> list[str]:
    videos: list[str] = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]:
        videos.extend([f.name for f in VIDEOS_DIR.glob(ext)])
    return sorted(list(set(videos)))


class ZoneCreate(BaseModel):
    name: str
    polygons: list
    video: str


class ZoneUpdate(BaseModel):
    polygons: list


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/videos")
async def list_videos():
    return {"videos": _list_video_files()}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    video_path = VIDEOS_DIR / file.filename
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"message": "Video uploaded", "filename": file.filename}


@app.get("/api/videos/{video_name}/frame")
async def get_video_frame(video_name: str):
    video_path = VIDEOS_DIR / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read video frame")

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/videos/{video_name}/info")
async def get_video_info(video_name: str):
    video_path = VIDEOS_DIR / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0
    }


@app.get("/api/zones/{video_name}")
async def get_zones_for_video(video_name: str):
    video_zones = zones_by_video.get(video_name, {})
    result = {}
    for zone_name, zone_data in video_zones.items():
        timer = zone_timers.get(zone_name, {})
        result[zone_name] = {
            "polygons": zone_data["polygons"],
            "total_time": get_zone_display_time(zone_name),
            "is_occupied": timer.get("occupy_start") is not None
        }
    return {"zones": result}


@app.post("/api/zones")
async def create_zone(zone: ZoneCreate):
    video_name = zone.video
    zone_name = zone.name

    with data_lock:
        if video_name not in zones_by_video:
            zones_by_video[video_name] = {}

        if zone_name in zones_by_video[video_name]:
            zones_by_video[video_name][zone_name]["polygons"].extend(zone.polygons)
        else:
            zones_by_video[video_name][zone_name] = {"polygons": zone.polygons}

        if zone_name not in zone_timers:
            zone_timers[zone_name] = {"total_time": 0, "occupy_start": None, "last_seen": None}

    save_zones()
    return {"message": "Zone created", "name": zone_name}


@app.put("/api/zones/{video_name}/{zone_name}")
async def update_zone(video_name: str, zone_name: str, update: ZoneUpdate):
    """
    Replace polygons for an existing zone (edit) WITHOUT touching timers.
    This preserves zone_timers / accumulated presence.
    """
    with data_lock:
        if video_name not in zones_by_video:
            raise HTTPException(status_code=404, detail="Video not found")
        if zone_name not in zones_by_video[video_name]:
            raise HTTPException(status_code=404, detail="Zone not found")

        zones_by_video[video_name][zone_name]["polygons"] = update.polygons

    save_zones()
    return {"message": "Zone updated", "name": zone_name, "video": video_name}


@app.delete("/api/zones/{video_name}/{zone_name}")
async def delete_zone(video_name: str, zone_name: str):
    with data_lock:
        if video_name not in zones_by_video:
            raise HTTPException(status_code=404, detail="Video not found")
        if zone_name not in zones_by_video[video_name]:
            raise HTTPException(status_code=404, detail="Zone not found")

        del zones_by_video[video_name][zone_name]

        zone_exists_elsewhere = any(
            zone_name in zones
            for v, zones in zones_by_video.items()
            if v != video_name
        )

        if not zone_exists_elsewhere and zone_name in zone_timers:
            del zone_timers[zone_name]

    save_zones()
    save_presence()
    return {"message": "Zone deleted"}


@app.delete("/api/zones/{video_name}")
async def delete_all_zones_for_video(video_name: str):
    with data_lock:
        if video_name in zones_by_video:
            zones_to_check = list(zones_by_video[video_name].keys())
            del zones_by_video[video_name]

            for zone_name in zones_to_check:
                zone_exists_elsewhere = any(
                    zone_name in zones
                    for zones in zones_by_video.values()
                )
                if not zone_exists_elsewhere and zone_name in zone_timers:
                    del zone_timers[zone_name]

    save_zones()
    save_presence()
    return {"message": "All zones deleted for video"}


@app.post("/api/zones/reset")
async def reset_all_timers():
    with data_lock:
        for zone_name in zone_timers:
            zone_timers[zone_name]["total_time"] = 0
            zone_timers[zone_name]["occupy_start"] = None
            zone_timers[zone_name]["last_seen"] = None
    save_presence()
    return {"message": "All timers reset"}


@app.post("/api/zones/reset/{zone_name}")
async def reset_zone_timer(zone_name: str):
    with data_lock:
        if zone_name in zone_timers:
            zone_timers[zone_name]["total_time"] = 0
            zone_timers[zone_name]["occupy_start"] = None
            zone_timers[zone_name]["last_seen"] = None
    save_presence()
    return {"message": f"Timer reset for {zone_name}"}


@app.get("/api/presence")
async def get_presence():
    result = {}
    for name, data in zone_timers.items():
        display_time = get_zone_display_time(name)
        result[name] = {
            "total_time": display_time,
            "formatted_time": format_time(display_time),
            "is_occupied": data.get("occupy_start") is not None
        }
    return {"zones": result}


@app.get("/api/presence/{video_name}")
async def get_presence_for_video(video_name: str):
    video_zones = zones_by_video.get(video_name, {})
    result = {}
    for zone_name in video_zones:
        timer = zone_timers.get(zone_name, {})
        display_time = get_zone_display_time(zone_name)
        result[zone_name] = {
            "total_time": display_time,
            "formatted_time": format_time(display_time),
            "is_occupied": timer.get("occupy_start") is not None
        }
    return {"zones": result}


@app.get("/api/streams")
async def get_active_streams():
    """Get list of currently active streams"""
    with streams_lock:
        return {
            "streams": [
                {"video": name, "active": info["active"]}
                for name, info in active_streams.items()
                if info["active"]
            ]
        }


@app.post("/api/stream/{video_name}/stop")
async def stop_video_stream(video_name: str):
    """Stop a specific video stream"""
    with streams_lock:
        if video_name in active_streams:
            active_streams[video_name]["active"] = False
    return {"message": f"Stream stopped for {video_name}"}


@app.post("/api/streams/stop")
async def stop_all_streams():
    """Stop all active streams"""
    with streams_lock:
        for video_name in active_streams:
            active_streams[video_name]["active"] = False
    return {"message": "All streams stopped"}


@app.get("/api/blur")
async def get_blur_status():
    """Get blur status"""
    with blur_lock:
        return {"enabled": blur_enabled}


@app.post("/api/blur/toggle")
async def toggle_blur():
    """Toggle blur on/off"""
    global blur_enabled
    with blur_lock:
        blur_enabled = not blur_enabled
        return {"enabled": blur_enabled}


@app.post("/api/blur/{state}")
async def set_blur(state: str):
    """Set blur state (on/off)"""
    global blur_enabled
    with blur_lock:
        blur_enabled = state.lower() in ("on", "true", "1", "enabled")
        return {"enabled": blur_enabled}


def get_system_metrics():
    """Collect system metrics: CPU, RAM, GPU"""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu_available": torch.cuda.is_available(),
        "gpu_percent": 0,
        "gpu_memory_used_mb": 0,
        "gpu_memory_total_mb": 0,
        "gpu_temperature": None,
        "gpu_name": None,
    }

    if torch.cuda.is_available():
        try:
            metrics["gpu_name"] = torch.cuda.get_device_name(0)
            metrics["gpu_memory_used_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
            metrics["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            # GPU utilization requires pynvml or nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 2:
                        metrics["gpu_percent"] = float(parts[0].strip())
                        metrics["gpu_temperature"] = float(parts[1].strip())
            except:
                pass
        except:
            pass

    return metrics


@app.get("/api/stats")
async def get_performance_stats():
    """Get real-time performance statistics"""
    system = get_system_metrics()

    # Count active streams
    with streams_lock:
        active_count = sum(1 for s in active_streams.values() if s.get("active", False))

    with perf_lock:
        # Calculate averages
        inference_times = list(perf_metrics["inference_times"])
        frame_times = list(perf_metrics["frame_times"])
        batch_sizes = list(perf_metrics["batch_size_history"])

        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0

        # Inference per second (adjusted for batch - counts frames processed, not batches)
        inferences_per_sec = (1000 / avg_inference * avg_batch_size) if avg_inference > 0 else 0

        # Update history for sparklines
        perf_metrics["fps_history"].append(perf_metrics["current_fps"])
        perf_metrics["inference_history"].append(perf_metrics["last_inference_time"])
        perf_metrics["cpu_history"].append(system["cpu_percent"])
        perf_metrics["gpu_history"].append(system["gpu_percent"])
        perf_metrics["ram_history"].append(system["ram_percent"])

        stats = {
            # Inference metrics
            "inference_time_ms": round(perf_metrics["last_inference_time"], 2),
            "inference_avg_ms": round(avg_inference, 2),
            "inferences_per_sec": round(inferences_per_sec, 1),

            # FPS metrics
            "current_fps": round(perf_metrics["current_fps"], 1),
            "frame_time_ms": round(perf_metrics["last_frame_time"], 2),
            "frame_time_avg_ms": round(avg_frame_time, 2),

            # Queue & Batch
            "queue_size": perf_metrics["queue_size"],
            "batch_size": perf_metrics["last_batch_size"],
            "batch_size_avg": round(avg_batch_size, 1),
            "batch_latency_ms": round(perf_metrics["batch_latency_ms"], 1),
            "active_streams": active_count,
            "detection_skip": DETECTION_SKIP_FRAMES,  # Detect 1 frame every N

            # Totals
            "total_detections": perf_metrics["total_detections"],
            "frames_processed": perf_metrics["frames_processed"],

            # System
            "cpu_percent": round(system["cpu_percent"], 1),
            "ram_used_mb": round(system["ram_used_mb"], 1),
            "ram_total_mb": round(system["ram_total_mb"], 1),
            "ram_percent": round(system["ram_percent"], 1),

            # GPU
            "gpu_available": system["gpu_available"],
            "gpu_name": system["gpu_name"],
            "gpu_percent": round(system["gpu_percent"], 1),
            "gpu_memory_used_mb": round(system["gpu_memory_used_mb"], 1),
            "gpu_memory_total_mb": round(system["gpu_memory_total_mb"], 1),
            "gpu_temperature": system["gpu_temperature"],

            # Sparkline history (last 60 samples)
            "history": {
                "fps": list(perf_metrics["fps_history"]),
                "inference": list(perf_metrics["inference_history"]),
                "cpu": list(perf_metrics["cpu_history"]),
                "gpu": list(perf_metrics["gpu_history"]),
                "ram": list(perf_metrics["ram_history"]),
            }
        }

    return stats


def bbox_in_zone(x1: float, y1: float, x2: float, y2: float, zone_polygons: list) -> bool:
    """Check if bbox intersects zone with at least INTERSECTION_THRESHOLD (30%) overlap"""
    bbox = box(x1, y1, x2, y2)
    bbox_area = bbox.area

    if bbox_area == 0:
        return False

    for polygon_points in zone_polygons:
        if len(polygon_points) >= 3:
            polygon = Polygon(polygon_points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if bbox.intersects(polygon):
                intersection = bbox.intersection(polygon)
                intersection_ratio = intersection.area / bbox_area
                if intersection_ratio >= INTERSECTION_THRESHOLD:
                    return True
    return False


def check_zones(detections: list, video_name: str):
    """
    Check zone occupancy and update timers using real wall-clock time.
    Timer structure: {
        "total_time": float,        # accumulated time in seconds
        "occupy_start": float|None, # timestamp when current occupation started
        "last_seen": float|None     # last time zone was seen occupied (for grace period)
    }
    """
    video_zones = zones_by_video.get(video_name, {})
    zone_occupancy = {name: False for name in video_zones}
    current_time = time.time()

    for det in detections:
        for zone_name, zone_data in video_zones.items():
            if bbox_in_zone(det["x1"], det["y1"], det["x2"], det["y2"], zone_data["polygons"]):
                zone_occupancy[zone_name] = True

    with data_lock:
        for zone_name, is_occupied in zone_occupancy.items():
            if zone_name not in zone_timers:
                zone_timers[zone_name] = {"total_time": 0, "occupy_start": None, "last_seen": None}

            timer = zone_timers[zone_name]

            # Migrate old format if needed
            if "occupy_start" not in timer:
                timer["occupy_start"] = timer.get("last_occupied")
                timer["last_seen"] = timer.get("last_occupied")

            if is_occupied:
                timer["last_seen"] = current_time
                # Start occupation timer if not already running
                if timer["occupy_start"] is None:
                    timer["occupy_start"] = current_time
            else:
                # Check grace period
                if timer["last_seen"] is not None:
                    time_since_last = current_time - timer["last_seen"]
                    if time_since_last < GRACE_PERIOD:
                        # Still in grace period, consider occupied
                        zone_occupancy[zone_name] = True
                    else:
                        # Grace period expired, finalize the occupation
                        if timer["occupy_start"] is not None:
                            # Add the real elapsed time to total
                            occupation_duration = timer["last_seen"] - timer["occupy_start"] + GRACE_PERIOD
                            timer["total_time"] += occupation_duration
                            timer["occupy_start"] = None
                        timer["last_seen"] = None

    return zone_occupancy


def get_zone_display_time(zone_name: str) -> float:
    """Get current display time for a zone (including ongoing occupation)"""
    with data_lock:
        if zone_name not in zone_timers:
            return 0
        timer = zone_timers[zone_name]
        total = timer["total_time"]

        # Add current ongoing occupation time
        if timer.get("occupy_start") is not None:
            current_time = time.time()
            total += current_time - timer["occupy_start"]

        return total


def parse_single_result(result):
    """Parse detections from a single YOLO result"""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf
        })
    return detections


def batch_detection_worker():
    """
    Centralized batch detection worker.
    Collects frames from all videos and processes them in batches for better GPU utilization.
    """
    global batch_worker_running
    batch_worker_running = True
    print("Batch detection worker started")

    while batch_worker_running:
        frames_batch = []
        video_names = []
        batch_start = time.perf_counter()

        # Collect frames until batch is full or timeout
        deadline = time.time() + BATCH_TIMEOUT
        while len(frames_batch) < BATCH_MAX_SIZE and time.time() < deadline:
            try:
                # Non-blocking get with small timeout
                video_name, frame, ts = batch_queue.get(timeout=0.01)
                frames_batch.append(frame)
                video_names.append(video_name)
            except:
                # Queue empty or timeout, continue collecting
                continue

        if not frames_batch:
            # No frames to process, sleep briefly
            time.sleep(0.005)
            continue

        batch_size = len(frames_batch)
        batch_collect_time = (time.perf_counter() - batch_start) * 1000

        # Update queue size metric
        with perf_lock:
            perf_metrics["queue_size"] = batch_queue.qsize()
            perf_metrics["batch_latency_ms"] = batch_collect_time

        try:
            # Batch inference - single GPU call for all frames
            inference_start = time.perf_counter()

            with model_lock:
                results = model(frames_batch, verbose=False, classes=[0], conf=YOLO_CONFIDENCE, device=YOLO_DEVICE)

            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            # Update performance metrics
            with perf_lock:
                perf_metrics["inference_times"].append(inference_time_ms)
                perf_metrics["last_inference_time"] = inference_time_ms
                perf_metrics["total_detections"] += batch_size
                perf_metrics["last_batch_size"] = batch_size
                perf_metrics["batch_size_history"].append(batch_size)

            # Distribute results to each video
            with detection_results_lock:
                for i, video_name in enumerate(video_names):
                    detections = parse_single_result(results[i])
                    detection_results[video_name] = {
                        "detections": detections,
                        "timestamp": time.time()
                    }

            # Also update active_streams for backward compatibility
            with streams_lock:
                for i, video_name in enumerate(video_names):
                    if video_name in active_streams:
                        detections = parse_single_result(results[i])
                        active_streams[video_name]["detections"] = detections

        except Exception as e:
            print(f"Batch detection error: {e}")
            continue

    print("Batch detection worker stopped")


def start_batch_worker():
    """Start the centralized batch detection worker if not already running"""
    global batch_worker_running
    if not batch_worker_running:
        worker_thread = threading.Thread(target=batch_detection_worker, daemon=True)
        worker_thread.start()


def stop_batch_worker():
    """Stop the batch detection worker"""
    global batch_worker_running
    batch_worker_running = False


# Detection skip rate: detect 1 frame every N frames (keep bboxes between detections)
DETECTION_SKIP_FRAMES = 4  # Detect every 4th frame

# Streaming skip rate: decode/display 1 frame every N frames (uses cap.grab() for skipped frames)
# Set to 1 for full FPS streaming, 4 for 1/4 FPS (saves CPU on decoding)
STREAMING_SKIP_FRAMES = 4  # Display every 4th frame (saves ~75% decoding CPU)

# JPEG encoding quality for streaming (lower = smaller size, less CPU)
JPEG_QUALITY = 75  # Reduced from 85 for streaming


def video_processor(video_name: str):
    """Background thread that processes a video continuously"""
    video_path = VIDEOS_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps

    # Ensure batch worker is running
    start_batch_worker()

    last_save = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    frame_counter = 0  # Counter to track which frames to send for detection

    try:
        while True:
            with streams_lock:
                if video_name not in active_streams or not active_streams[video_name]["active"]:
                    break

            frame_start = time.time()
            frame_counter += 1

            # Decide if we should decode this frame for streaming
            should_decode = (frame_counter % STREAMING_SKIP_FRAMES == 0) or (STREAMING_SKIP_FRAMES == 1)
            should_detect = (frame_counter % DETECTION_SKIP_FRAMES == 0)

            if should_decode:
                # Decode frame for streaming display
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    continue

                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Store frame for streaming
                with streams_lock:
                    shared_frames[video_name] = {
                        "frame": frame.copy(),
                        "frame_num": frame_num
                    }

                # Send frame to batch queue for detection if needed
                if should_detect:
                    try:
                        batch_queue.put_nowait((video_name, frame.copy(), time.time()))
                    except:
                        pass  # Queue full, skip this frame
            else:
                # Skip frame without decoding (much faster)
                ret = cap.grab()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    continue
                # shared_frames keeps the last decoded frame (reused for streaming)

            # Get detections from batch results (same detections reused between updates)
            with streams_lock:
                detections = active_streams[video_name]["detections"].copy() if video_name in active_streams else []

            # Check zones and update timers (uses real wall-clock time internally)
            check_zones(detections, video_name)

            if time.time() - last_save > 2:
                save_presence()
                last_save = time.time()

            # Calculate FPS metrics
            fps_counter += 1
            elapsed_since_fps_start = time.time() - fps_start_time
            if elapsed_since_fps_start >= 1.0:
                current_fps = fps_counter / elapsed_since_fps_start
                with perf_lock:
                    perf_metrics["current_fps"] = current_fps
                    perf_metrics["frames_processed"] += fps_counter
                fps_counter = 0
                fps_start_time = time.time()

            # Track frame processing time
            frame_time_ms = (time.time() - frame_start) * 1000
            with perf_lock:
                perf_metrics["frame_times"].append(frame_time_ms)
                perf_metrics["last_frame_time"] = frame_time_ms

            elapsed = time.time() - frame_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        with streams_lock:
            if video_name in shared_frames:
                del shared_frames[video_name]
        save_presence()
        print(f"Video processor stopped for {video_name}")


def apply_smooth_blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Apply smooth elliptical blur to the top 30% of a bounding box"""
    h, w = frame.shape[:2]

    # Calculate top 30% region
    bbox_height = y2 - y1
    blur_height = int(bbox_height * 0.30)
    blur_y2 = y1 + blur_height

    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    blur_y2 = min(h, blur_y2)

    if x2 <= x1 or blur_y2 <= y1:
        return frame

    # Extract region to blur
    region = frame[y1:blur_y2, x1:x2].copy()
    if region.size == 0:
        return frame

    # Apply strong Gaussian blur
    blur_size = max(31, (blur_height // 2) * 2 + 1)  # Ensure odd number
    blurred_region = cv2.GaussianBlur(region, (blur_size, blur_size), 0)

    # Create elliptical gradient mask for smooth transition
    mask_h, mask_w = region.shape[:2]
    mask = np.zeros((mask_h, mask_w), dtype=np.float32)

    # Create ellipse parameters
    center_x = mask_w // 2
    center_y = mask_h // 2
    axis_x = mask_w // 2
    axis_y = mask_h // 2

    # Draw filled ellipse on mask
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    # Apply Gaussian blur to mask edges for smooth transition
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # Expand mask to 3 channels
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Blend original and blurred using mask
    blended = (blurred_region * mask_3ch + region * (1 - mask_3ch)).astype(np.uint8)

    # Apply back to frame
    frame[y1:blur_y2, x1:x2] = blended

    return frame


def generate_frames(video_name: str):
    """Generate frames for streaming - reads from shared_frames written by processor"""
    last_frame_num = -1

    while True:
        with streams_lock:
            if video_name not in active_streams or not active_streams[video_name]["active"]:
                break

            if video_name not in shared_frames:
                time.sleep(0.01)
                continue

            frame_data = shared_frames[video_name]
            frame_num = frame_data["frame_num"]

            # Skip if same frame
            if frame_num == last_frame_num:
                time.sleep(0.005)
                continue

            frame = frame_data["frame"].copy()
            last_frame_num = frame_num

            detections = active_streams[video_name]["detections"].copy() if video_name in active_streams else []

        # Apply blur if enabled (before drawing boxes)
        with blur_lock:
            should_blur = blur_enabled

        if should_blur:
            for det in detections:
                x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
                frame = apply_smooth_blur(frame, x1, y1, x2, y2)

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            conf = det["conf"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.0%}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw zones
        video_zones = zones_by_video.get(video_name, {})
        for zone_name, zone_data in video_zones.items():
            timer = zone_timers.get(zone_name, {})
            is_occupied = timer.get("occupy_start") is not None
            color = (0, 0, 255) if is_occupied else (255, 165, 0)

            for polygon_points in zone_data["polygons"]:
                if len(polygon_points) >= 3:
                    pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))

                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    cv2.polylines(frame, [pts], True, color, 3)

            if zone_data["polygons"] and len(zone_data["polygons"][0]) > 0:
                first_point = zone_data["polygons"][0][0]
                display_time = get_zone_display_time(zone_name)
                time_str = format_time(display_time)
                label = f"{zone_name}: {time_str}"

                (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame,
                              (int(first_point[0]) - 5, int(first_point[1]) - 25),
                              (int(first_point[0]) + w + 5, int(first_point[1]) + 5),
                              (0, 0, 0), -1)
                cv2.putText(frame, label, (int(first_point[0]), int(first_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode frame to JPEG using TurboJPEG (faster) or OpenCV fallback
        if USE_TURBOJPEG:
            jpeg_bytes = jpeg_encoder.encode(frame, quality=JPEG_QUALITY)
        else:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            jpeg_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')


@app.post("/api/stream/{video_name}/start")
async def start_video_processing(video_name: str):
    """Start processing a video (detection + zone tracking) in background"""
    video_path = VIDEOS_DIR / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    with streams_lock:
        if video_name in active_streams and active_streams[video_name]["active"]:
            return {"message": "Stream already active", "video": video_name}

        active_streams[video_name] = {
            "active": True,
            "detections": []
        }

    # Start background processing thread
    processor_thread = threading.Thread(
        target=video_processor,
        args=(video_name,),
        daemon=True
    )
    processor_thread.start()

    return {"message": "Stream started", "video": video_name}


@app.get("/api/stream/{video_name}")
async def video_stream(video_name: str):
    """Get video stream with detections overlay"""
    video_path = VIDEOS_DIR / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Auto-start processing if not already running
    with streams_lock:
        if video_name not in active_streams or not active_streams[video_name]["active"]:
            active_streams[video_name] = {
                "active": True,
                "detections": []
            }
            processor_thread = threading.Thread(
                target=video_processor,
                args=(video_name,),
                daemon=True
            )
            processor_thread.start()

    return StreamingResponse(
        generate_frames(video_name),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
