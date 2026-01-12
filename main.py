import json
import time
import threading
from pathlib import Path
from queue import Queue
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from shapely.geometry import Polygon, box
from ultralytics import YOLO

from conveyor_counter import ConveyorCounter

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

# Data state
zones_by_video = {}
zone_timers = {}
data_lock = threading.Lock()

# Active streams: {video_name: {"active": bool, "detections": [], "current_frame": int, "frame_lock": Lock}}
active_streams = {}
streams_lock = threading.Lock()
# Shared frames for streaming (processor writes, streamer reads)
shared_frames = {}  # {video_name: {"frame": ndarray, "frame_num": int}}

# Conveyor counters per video
conveyor_counters = {}  # {video_name: ConveyorCounter}
conveyor_lock = threading.Lock()

# Seuil minimum d'intersection bbox/zone (30%)
INTERSECTION_THRESHOLD = 0.30

# Floutage des visages (30% haut de la bbox)
blur_enabled = False
blur_lock = threading.Lock()

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


class ZoneCreate(BaseModel):
    name: str
    polygons: list
    video: str
    zone_type: str = "presence"  # "presence", "include", or "counting"


class ZoneUpdate(BaseModel):
    polygons: list


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/videos")
async def list_videos():
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV"]:
        videos.extend([f.name for f in VIDEOS_DIR.glob(ext)])
    return {"videos": list(set(videos))}


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
        zone_type = zone_data.get("type", "presence")
        timer = zone_timers.get(zone_name, {})
        result[zone_name] = {
            "polygons": zone_data["polygons"],
            "type": zone_type,
            "total_time": get_zone_display_time(zone_name) if zone_type == "presence" else 0,
            "is_occupied": timer.get("occupy_start") is not None if zone_type == "presence" else False
        }
    return {"zones": result}


@app.post("/api/zones")
async def create_zone(zone: ZoneCreate):
    video_name = zone.video
    zone_name = zone.name
    zone_type = zone.zone_type

    with data_lock:
        if video_name not in zones_by_video:
            zones_by_video[video_name] = {}

        if zone_name in zones_by_video[video_name]:
            zones_by_video[video_name][zone_name]["polygons"].extend(zone.polygons)
        else:
            zones_by_video[video_name][zone_name] = {
                "polygons": zone.polygons,
                "type": zone_type
            }

        # Only create timer for presence zones
        if zone_type == "presence":
            if zone_name not in zone_timers:
                zone_timers[zone_name] = {"total_time": 0, "occupy_start": None, "last_seen": None}

    save_zones()

    # Update conveyor counter masks if needed
    if zone_type in ("include", "counting"):
        update_conveyor_masks(video_name)

    return {"message": "Zone created", "name": zone_name, "type": zone_type}


@app.delete("/api/zones/{video_name}/{zone_name}")
async def delete_zone(video_name: str, zone_name: str):
    zone_type = None
    with data_lock:
        if video_name not in zones_by_video:
            raise HTTPException(status_code=404, detail="Video not found")
        if zone_name not in zones_by_video[video_name]:
            raise HTTPException(status_code=404, detail="Zone not found")

        zone_type = zones_by_video[video_name][zone_name].get("type", "presence")
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

    # Update conveyor masks if needed
    if zone_type in ("include", "counting"):
        update_conveyor_masks(video_name)

    return {"message": "Zone deleted"}


@app.delete("/api/zones/{video_name}")
async def delete_all_zones_for_video(video_name: str):
    had_conveyor_zones = False
    with data_lock:
        if video_name in zones_by_video:
            # Check if any conveyor zones exist
            for zone_data in zones_by_video[video_name].values():
                if zone_data.get("type", "presence") in ("include", "counting"):
                    had_conveyor_zones = True
                    break

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

    # Update conveyor masks if needed
    if had_conveyor_zones:
        update_conveyor_masks(video_name)

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


# =============================================================================
# CONVEYOR COUNTING ENDPOINTS
# =============================================================================

@app.get("/api/conveyor/{video_name}")
async def get_conveyor_count(video_name: str):
    """Get current conveyor counter stats for a video"""
    with conveyor_lock:
        if video_name not in conveyor_counters:
            return {
                "count": 0,
                "state": "inactive",
                "warning": "Counter not initialized. Start the video stream first."
            }
        return conveyor_counters[video_name].get_stats()


@app.post("/api/conveyor/{video_name}/reset")
async def reset_conveyor_count(video_name: str):
    """Reset conveyor counter to zero"""
    with conveyor_lock:
        if video_name in conveyor_counters:
            conveyor_counters[video_name].reset_count()
    return {"message": f"Counter reset for {video_name}"}


@app.post("/api/conveyor/{video_name}/reset-background")
async def reset_conveyor_background(video_name: str):
    """Reset background model (useful after lighting changes)"""
    with conveyor_lock:
        if video_name in conveyor_counters:
            conveyor_counters[video_name].reset_background()
    return {"message": f"Background model reset for {video_name}"}


@app.post("/api/conveyor/{video_name}/calibrate")
async def calibrate_conveyor(video_name: str):
    """Manually trigger threshold calibration (belt should be empty)"""
    with conveyor_lock:
        if video_name in conveyor_counters:
            counter = conveyor_counters[video_name]
            counter.warmup_samples.clear()
            counter.warmup_complete = False
            counter.frames_processed = 0
    return {"message": f"Calibration started for {video_name}"}


@app.post("/api/conveyor/{video_name}/flip")
async def flip_conveyor_direction(video_name: str):
    """Rotate conveyor direction by 90 degrees - reorients bands perpendicular to new axis"""
    with conveyor_lock:
        if video_name in conveyor_counters:
            counter = conveyor_counters[video_name]
            # Rotate axis by 90 degrees (perpendicular)
            ux, uy = counter.axis_u
            # Rotate 90° counterclockwise: (ux, uy) -> (-uy, ux)
            counter.axis_u = (-uy, ux)
            # Update theta by +90°
            counter.axis_theta_deg = (counter.axis_theta_deg + 90) % 360
            if counter.axis_theta_deg > 180:
                counter.axis_theta_deg -= 360

            # Recompute band boundaries with new axis direction
            if counter.counting_polygon_pts is not None:
                new_ux, new_uy = counter.axis_u
                projections = counter.counting_polygon_pts[:, 0] * new_ux + counter.counting_polygon_pts[:, 1] * new_uy
                counter.p_min = float(np.min(projections))
                counter.p_max = float(np.max(projections))
                import conveyor_config as cfg
                if counter.p_max > counter.p_min:
                    counter.band_width = (counter.p_max - counter.p_min) / cfg.NUM_BANDS
    return {"message": f"Direction rotated 90° for {video_name}"}


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


def get_zones_by_type(video_name: str, zone_type: str) -> list:
    """Get all polygons of a specific type for a video."""
    video_zones = zones_by_video.get(video_name, {})
    polygons = []
    for zone_name, zone_data in video_zones.items():
        if zone_data.get("type", "presence") == zone_type:
            polygons.extend(zone_data["polygons"])
    return polygons


def update_conveyor_masks(video_name: str):
    """Update conveyor counter masks for a video based on current zones."""
    with conveyor_lock:
        if video_name not in conveyor_counters:
            return

        counter = conveyor_counters[video_name]
        include_polygons = get_zones_by_type(video_name, "include")
        counting_polygons = get_zones_by_type(video_name, "counting")
        counter.set_masks(include_polygons, counting_polygons)


def init_conveyor_counter(video_name: str, frame_h: int, frame_w: int):
    """Initialize or get conveyor counter for a video."""
    with conveyor_lock:
        if video_name not in conveyor_counters:
            conveyor_counters[video_name] = ConveyorCounter((frame_h, frame_w))

        counter = conveyor_counters[video_name]
        include_polygons = get_zones_by_type(video_name, "include")
        counting_polygons = get_zones_by_type(video_name, "counting")
        counter.set_masks(include_polygons, counting_polygons)
        return counter


def detection_worker(video_name: str, frame_queue: Queue):
    """Background thread that runs YOLO detection for a specific video"""
    while True:
        with streams_lock:
            if video_name not in active_streams or not active_streams[video_name]["active"]:
                break

        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = None
            while not frame_queue.empty():
                frame = frame_queue.get_nowait()

            if frame is None:
                continue

            with model_lock:
                results = model(frame, verbose=False, classes=[0], conf=YOLO_CONFIDENCE, device=0)

            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf
                    })

            with streams_lock:
                if video_name in active_streams:
                    active_streams[video_name]["detections"] = detections

        except Exception as e:
            print(f"Detection error for {video_name}: {e}")
            continue


def video_processor(video_name: str):
    """Background thread that processes a video continuously"""
    video_path = VIDEOS_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_queue = Queue(maxsize=2)

    detection_thread = threading.Thread(
        target=detection_worker,
        args=(video_name, frame_queue),
        daemon=True
    )
    detection_thread.start()

    # Initialize conveyor counter
    conveyor_counter = init_conveyor_counter(video_name, frame_h, frame_w)

    last_save = time.time()

    try:
        while True:
            with streams_lock:
                if video_name not in active_streams or not active_streams[video_name]["active"]:
                    break

            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset background model on video loop
                with conveyor_lock:
                    if video_name in conveyor_counters:
                        conveyor_counters[video_name].reset_background()
                continue

            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Update conveyor counter
            with conveyor_lock:
                if video_name in conveyor_counters:
                    conveyor_result = conveyor_counters[video_name].update(frame)
                else:
                    conveyor_result = None

            # Store frame and conveyor stats for streaming
            with streams_lock:
                shared_frames[video_name] = {
                    "frame": frame.copy(),
                    "frame_num": frame_num,
                    "conveyor": conveyor_result
                }

            if not frame_queue.full():
                try:
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass

            with streams_lock:
                detections = active_streams[video_name]["detections"].copy() if video_name in active_streams else []

            # Check zones and update timers (uses real wall-clock time internally)
            check_zones(detections, video_name)

            if time.time() - last_save > 2:
                save_presence()
                last_save = time.time()

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
            conveyor_result = frame_data.get("conveyor")

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

        # Draw zones by type
        video_zones = zones_by_video.get(video_name, {})
        for zone_name, zone_data in video_zones.items():
            zone_type = zone_data.get("type", "presence")

            if zone_type == "presence":
                # Original presence zone drawing (orange/red based on occupancy)
                timer = zone_timers.get(zone_name, {})
                is_occupied = timer.get("occupy_start") is not None
                color = (0, 0, 255) if is_occupied else (255, 165, 0)  # Red if occupied, orange otherwise

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

            elif zone_type == "include":
                # Include zone: cyan/teal color
                color = (255, 255, 0)  # Cyan (BGR)
                for polygon_points in zone_data["polygons"]:
                    if len(polygon_points) >= 3:
                        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                        cv2.polylines(frame, [pts], True, color, 2)

                if zone_data["polygons"] and len(zone_data["polygons"][0]) > 0:
                    first_point = zone_data["polygons"][0][0]
                    label = f"{zone_name} [ROI]"
                    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame,
                                  (int(first_point[0]) - 3, int(first_point[1]) - 18),
                                  (int(first_point[0]) + w + 3, int(first_point[1]) + 3),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, label, (int(first_point[0]), int(first_point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif zone_type == "counting":
                # Counting zone: magenta/purple, with counter display
                is_occupied = conveyor_result and conveyor_result.get("is_present", False)
                color = (255, 0, 255) if is_occupied else (180, 0, 180)  # Bright magenta if occupied

                for polygon_points in zone_data["polygons"]:
                    if len(polygon_points) >= 3:
                        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                        overlay = frame.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                        cv2.polylines(frame, [pts], True, color, 3)

                # Method 6 debug overlay
                conveyor_counter = conveyor_counters.get(video_name)
                if conveyor_result and conveyor_counter:
                    import conveyor_config as cfg
                    if cfg.USE_METHOD_6 and cfg.DEBUG_OVERLAY:
                        debug_data = conveyor_counter.get_debug_overlay_data()
                        min_area_rect = debug_data.get("min_area_rect")
                        axis_u = debug_data.get("axis_u", (0, 1))
                        axis_theta = debug_data.get("axis_theta", 0)
                        p_min = debug_data.get("p_min", 0)
                        p_max = debug_data.get("p_max", 1)
                        band_width = debug_data.get("band_width", 1)
                        band_counts = debug_data.get("band_counts", [])

                        # Draw minAreaRect box
                        if min_area_rect is not None:
                            box = cv2.boxPoints(min_area_rect)
                            box = np.int32(box)
                            cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)

                            # Draw arrow showing conveyor direction from center
                            center = min_area_rect[0]
                            cx, cy = int(center[0]), int(center[1])
                            ux, uy = axis_u
                            arrow_len = 60
                            ax, ay = int(cx + ux * arrow_len), int(cy + uy * arrow_len)
                            cv2.arrowedLine(frame, (cx, cy), (ax, ay), (0, 255, 0), 3, tipLength=0.3)

                            # Draw band boundaries
                            if band_width > 0 and len(band_counts) > 0:
                                # Perpendicular direction for band lines
                                perp_x, perp_y = -uy, ux
                                perp_len = 50

                                for i in range(cfg.NUM_BANDS + 1):
                                    p = p_min + i * band_width
                                    # Point on axis at projection p
                                    bx = cx + ux * (p - (p_min + p_max) / 2)
                                    by = cy + uy * (p - (p_min + p_max) / 2)
                                    # Draw perpendicular line
                                    x1 = int(bx - perp_x * perp_len)
                                    y1 = int(by - perp_y * perp_len)
                                    x2 = int(bx + perp_x * perp_len)
                                    y2 = int(by + perp_y * perp_len)
                                    # Upstream bands (first ones) in orange
                                    is_upstream = i < cfg.UPSTREAM_BANDS
                                    band_color = (255, 128, 0) if is_upstream else (128, 128, 128)
                                    cv2.line(frame, (x1, y1), (x2, y2), band_color, 1)

                            # Draw axis angle text
                            angle_txt = f"θ:{axis_theta:.1f}°"
                            cv2.putText(frame, angle_txt, (cx + 10, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Display counter near the counting zone
                if zone_data["polygons"] and len(zone_data["polygons"][0]) > 0 and conveyor_result:
                    first_point = zone_data["polygons"][0][0]
                    count = conveyor_result.get("count", 0)
                    warmup = conveyor_result.get("warmup_progress", 1.0)
                    s_norm = conveyor_result.get("s_norm", 0)
                    threshold = conveyor_result.get("threshold", 0)
                    state = conveyor_result.get("state", "?")

                    if warmup < 1.0:
                        label = f"{zone_name}: Calibrating {int(warmup * 100)}%"
                    else:
                        label = f"{zone_name}: {count}"

                    # Method 6 debug info
                    import conveyor_config as cfg
                    if cfg.USE_METHOD_6:
                        e_smooth = conveyor_result.get("e_smooth", 0)
                        armed = conveyor_result.get("armed", False)
                        cooldown = conveyor_result.get("cooldown", 0)
                        ds = conveyor_result.get("ds", 0)
                        dc = conveyor_result.get("dc", 0)
                        ds1 = conveyor_result.get("ds1", 0)

                        arm_txt = "ARM" if armed else f"CD:{cooldown}"
                        debug_label = f"E:{e_smooth:.1f} dS1:{ds1:.0f} dC:{dc:.1f} [{arm_txt}]"
                        debug_label2 = f"S:{s_norm:.3f} dS:{ds:.0f} T_E:{cfg.EVENT_THRESHOLD}"
                    else:
                        debug_label = f"S:{s_norm:.3f} T:{threshold:.3f} [{state}]"
                        debug_label2 = None

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame,
                                  (int(first_point[0]) - 5, int(first_point[1]) - 28),
                                  (int(first_point[0]) + w + 5, int(first_point[1]) + 5),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, label, (int(first_point[0]), int(first_point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Debug text below the main label
                    (w2, _), _ = cv2.getTextSize(debug_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame,
                                  (int(first_point[0]) - 5, int(first_point[1]) + 8),
                                  (int(first_point[0]) + w2 + 5, int(first_point[1]) + 28),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, debug_label, (int(first_point[0]), int(first_point[1]) + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Second debug line for Method 6
                    if debug_label2:
                        (w3, _), _ = cv2.getTextSize(debug_label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame,
                                      (int(first_point[0]) - 5, int(first_point[1]) + 30),
                                      (int(first_point[0]) + w3 + 5, int(first_point[1]) + 50),
                                      (0, 0, 0), -1)
                        cv2.putText(frame, debug_label2, (int(first_point[0]), int(first_point[1]) + 44),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


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
