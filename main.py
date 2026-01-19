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
import torch

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

# Active streams: {video_name: {"active": bool, "detections": [], "frame_event": Event, ...}}
active_streams = {}
streams_lock = threading.Lock()
# Shared frames for streaming (processor writes, streamer reads)
# Using separate lock for frames to reduce contention
shared_frames = {}  # {video_name: {"frame": ndarray, "frame_num": int}}
frames_lock = threading.Lock()

# Seuil minimum d'intersection bbox/zone (30%)
INTERSECTION_THRESHOLD = 0.30

# Floutage des visages (30% haut de la bbox)
blur_enabled = False
blur_lock = threading.Lock()

ZONES_FILE = DATA_DIR / "zones.json"
PRESENCE_FILE = DATA_DIR / "presence.json"
CAMERAS_FILE = DATA_DIR / "cameras.json"

GRACE_PERIOD = 0.5
YOLO_CONFIDENCE = 0.45
TRACKING_REINFERENCE_INTERVAL = 30  # Réinférence complète toutes les 30 frames

# Camera sources storage: {camera_id: {"type": "webcam"|"rtsp", "name": str, ...}}
cameras = {}


def load_data():
    global zones_by_video, zone_timers, cameras
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
    if CAMERAS_FILE.exists():
        with open(CAMERAS_FILE, "r") as f:
            cameras = json.load(f)


def save_zones():
    with open(ZONES_FILE, "w") as f:
        json.dump(zones_by_video, f, indent=2)


def save_presence():
    with data_lock:
        with open(PRESENCE_FILE, "w") as f:
            save_data = {k: {"total_time": v["total_time"]} for k, v in zone_timers.items()}
            json.dump(save_data, f, indent=2)


def save_cameras():
    with open(CAMERAS_FILE, "w") as f:
        json.dump(cameras, f, indent=2)


def get_camera_source(camera_id: str):
    """Get the OpenCV capture source for a camera (device ID for webcam, URL for RTSP)"""
    if camera_id not in cameras:
        return None
    cam = cameras[camera_id]
    if cam["type"] == "webcam":
        return cam["device_id"]
    elif cam["type"] == "rtsp":
        return cam["url"]
    return None


def is_camera_source(source_name: str) -> bool:
    """Check if source_name is a camera (vs a video file)"""
    return source_name.startswith("camera:")


def get_source_identifier(source_name: str):
    """
    Get the OpenCV capture source from a source name.
    - For videos: returns file path string
    - For cameras: returns device ID (int) or RTSP URL (string)
    """
    if is_camera_source(source_name):
        camera_id = source_name.replace("camera:", "")
        return get_camera_source(camera_id)
    else:
        return str(VIDEOS_DIR / source_name)


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


@app.get("/api/videos/{video_name:path}/frame")
async def get_video_frame(video_name: str):
    # Support both video files and camera sources (camera:xxx)
    is_camera = is_camera_source(video_name)

    # For cameras, first try to get frame from active stream (faster, no reconnect)
    if is_camera:
        with frames_lock:
            if video_name in shared_frames and shared_frames[video_name]["frame"] is not None:
                frame = shared_frames[video_name]["frame"].copy()
                _, buffer = cv2.imencode('.jpg', frame)
                return StreamingResponse(
                    iter([buffer.tobytes()]),
                    media_type="image/jpeg"
                )

    # Fallback: open capture directly
    if is_camera:
        camera_id = video_name.replace("camera:", "")
        source = get_camera_source(camera_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        cap = cv2.VideoCapture(source)
        # For cameras, wait a bit for connection to establish
        if not cap.isOpened():
            cap.release()
            raise HTTPException(status_code=500, detail="Could not open camera")
    else:
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        cap = cv2.VideoCapture(str(video_path))

    # For cameras/RTSP, try multiple reads to get a valid frame
    frame = None
    max_attempts = 10 if is_camera else 1
    for _ in range(max_attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame")

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.get("/api/videos/{video_name:path}/info")
async def get_video_info(video_name: str):
    # Support both video files and camera sources (camera:xxx)
    if is_camera_source(video_name):
        camera_id = video_name.replace("camera:", "")
        source = get_camera_source(camera_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        cap = cv2.VideoCapture(source)
    else:
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        cap = cv2.VideoCapture(str(video_path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # For cameras, frame_count is 0 or invalid
    is_live = is_camera_source(video_name)

    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30,
        "frame_count": frame_count if not is_live else 0,
        "duration": frame_count / fps if fps > 0 and not is_live else 0,
        "is_live": is_live
    }


@app.get("/api/zones/{video_name:path}")
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


@app.put("/api/zones/{video_name:path}/{zone_name}")
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


@app.delete("/api/zones/{video_name:path}/{zone_name}")
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


@app.delete("/api/zones/{video_name:path}")
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


@app.get("/api/presence/{video_name:path}")
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


@app.post("/api/stream/{video_name:path}/stop")
async def stop_video_stream(video_name: str):
    """Stop a specific video or camera stream"""
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


# ==================== Camera API Endpoints ====================

class CameraCreate(BaseModel):
    camera_id: str
    name: str
    type: str  # "webcam" or "rtsp"
    device_id: int | None = None  # For webcam
    url: str | None = None  # For RTSP


@app.get("/api/cameras")
async def list_cameras():
    """List all configured cameras"""
    return {"cameras": cameras}


@app.post("/api/cameras")
async def add_camera(camera: CameraCreate):
    """Add a new camera source"""
    if camera.camera_id in cameras:
        raise HTTPException(status_code=400, detail="Camera ID already exists")

    if camera.type == "webcam":
        if camera.device_id is None:
            raise HTTPException(status_code=400, detail="device_id required for webcam")
        cameras[camera.camera_id] = {
            "type": "webcam",
            "name": camera.name,
            "device_id": camera.device_id
        }
    elif camera.type == "rtsp":
        if not camera.url:
            raise HTTPException(status_code=400, detail="url required for RTSP")
        cameras[camera.camera_id] = {
            "type": "rtsp",
            "name": camera.name,
            "url": camera.url
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid camera type")

    save_cameras()
    return {"message": "Camera added", "camera_id": camera.camera_id}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop stream if active
    source_name = f"camera:{camera_id}"
    with streams_lock:
        if source_name in active_streams:
            active_streams[source_name]["active"] = False

    del cameras[camera_id]
    save_cameras()
    return {"message": "Camera deleted"}


@app.get("/api/cameras/detect/webcams")
async def detect_webcams():
    """Detect available webcams on the system"""
    available = []
    # Test indices 0-4 for common webcam setups
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({
                "device_id": i,
                "name": f"Webcam {i}",
                "resolution": f"{width}x{height}"
            })
            cap.release()
    return {"webcams": available}


@app.get("/api/cameras/detect/onvif")
async def detect_onvif_cameras():
    """Discover ONVIF cameras on the local network using WS-Discovery"""
    try:
        from wsdiscovery import WSDiscovery
        from wsdiscovery import QName, Scope

        discovered = []

        wsd = WSDiscovery()
        wsd.start()

        # Search for ONVIF devices (NetworkVideoTransmitter type)
        services = wsd.searchServices(
            types=[QName("http://www.onvif.org/ver10/network/wsdl", "NetworkVideoTransmitter")]
        )

        for service in services:
            xaddrs = service.getXAddrs()
            scopes = service.getScopes()

            # Extract name from scopes if available
            name = "ONVIF Camera"
            for scope in scopes:
                scope_str = str(scope)
                if "onvif://www.onvif.org/name/" in scope_str:
                    name = scope_str.split("/name/")[-1]
                    break

            for xaddr in xaddrs:
                discovered.append({
                    "name": name,
                    "xaddr": xaddr,
                    "scopes": [str(s) for s in scopes]
                })

        wsd.stop()
        return {"cameras": discovered}

    except ImportError:
        return {"cameras": [], "error": "WS-Discovery not available. Install with: pip install wsdiscovery"}
    except Exception as e:
        return {"cameras": [], "error": str(e)}


@app.post("/api/cameras/test-rtsp")
async def test_rtsp_url(url: str):
    """Test if an RTSP URL is accessible"""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return {"success": False, "error": "Could not connect to RTSP stream"}

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"success": False, "error": "Connected but could not read frame"}

    height, width = frame.shape[:2]
    return {
        "success": True,
        "resolution": f"{width}x{height}"
    }


@app.post("/api/cameras/{camera_id}/test")
async def test_camera(camera_id: str):
    """Test if a camera connection works"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    source = get_camera_source(camera_id)
    if source is None:
        raise HTTPException(status_code=400, detail="Invalid camera configuration")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open camera"}

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"success": False, "error": "Could not read frame"}

    height, width = frame.shape[:2]
    return {
        "success": True,
        "resolution": f"{width}x{height}"
    }


@app.get("/api/cameras/{camera_id}/frame")
async def get_camera_frame(camera_id: str):
    """Get a single frame from a camera (for preview)"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    source = get_camera_source(camera_id)
    if source is None:
        raise HTTPException(status_code=400, detail="Invalid camera configuration")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open camera")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Could not read frame")

    _, buffer = cv2.imencode('.jpg', frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.post("/api/stream/camera/{camera_id}/start")
async def start_camera_stream(camera_id: str):
    """Start streaming a camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    source_name = f"camera:{camera_id}"

    with streams_lock:
        if source_name in active_streams and active_streams[source_name]["active"]:
            return {"message": "Stream already active", "source": source_name}

        active_streams[source_name] = {
            "active": True,
            "detections": [],
            "frame_event": threading.Event()
        }

    processor_thread = threading.Thread(
        target=video_processor,
        args=(source_name,),
        daemon=True
    )
    processor_thread.start()

    return {"message": "Camera stream started", "source": source_name}


@app.get("/api/stream/camera/{camera_id}")
async def camera_stream(camera_id: str):
    """Get camera stream with detections overlay"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    source_name = f"camera:{camera_id}"

    # Auto-start processing if not already running
    with streams_lock:
        if source_name not in active_streams or not active_streams[source_name]["active"]:
            active_streams[source_name] = {
                "active": True,
                "detections": [],
                "frame_event": threading.Event()
            }
            processor_thread = threading.Thread(
                target=video_processor,
                args=(source_name,),
                daemon=True
            )
            processor_thread.start()

    return StreamingResponse(
        generate_frames(source_name),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


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


def detection_worker(video_name: str, frame_queue: Queue):
    """
    Background thread that runs YOLO tracking for a specific video.
    Uses tracking instead of pure detection - only runs full inference every N frames
    or when a tracked ID is lost.
    """
    frame_count = 0
    last_track_ids = set()

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

            frame_count += 1

            # Decide whether to run full inference or just tracking
            # Run full inference if:
            # 1. It's the first frame
            # 2. Every TRACKING_REINFERENCE_INTERVAL frames
            # 3. We lost a tracked ID (someone left the frame)
            run_full_inference = (
                frame_count == 1 or
                frame_count % TRACKING_REINFERENCE_INTERVAL == 0
            )

            with model_lock:
                # Use model.track() instead of model() for tracking
                # persist=True keeps track IDs across frames
                results = model.track(
                    frame,
                    verbose=False,
                    classes=[0],
                    conf=YOLO_CONFIDENCE,
                    device=YOLO_DEVICE,
                    persist=True,
                    tracker="bytetrack.yaml"  # ByteTrack is fast and efficient
                )

            detections = []
            current_track_ids = set()

            for r in results:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    # Get track ID if available
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                        current_track_ids.add(track_id)

                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf,
                        "track_id": track_id
                    })

            # Check if any tracked IDs were lost (for next iteration)
            lost_ids = last_track_ids - current_track_ids
            if lost_ids:
                # IDs were lost, next frame should run full inference
                pass  # The tracker handles this automatically

            last_track_ids = current_track_ids

            with streams_lock:
                if video_name in active_streams:
                    active_streams[video_name]["detections"] = detections

        except Exception as e:
            print(f"Detection error for {video_name}: {e}")
            continue


def video_processor(source_name: str):
    """Background thread that processes a video or camera stream continuously"""
    source = get_source_identifier(source_name)
    is_camera = is_camera_source(source_name)

    if source is None:
        print(f"Error: Could not find source for {source_name}")
        return

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # For cameras, use a default FPS since they may not report it correctly
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:  # Invalid or unrealistic FPS
        fps = 30
    frame_delay = 1.0 / fps

    # Larger queue to handle processing spikes in Docker
    frame_queue = Queue(maxsize=5)

    detection_thread = threading.Thread(
        target=detection_worker,
        args=(source_name, frame_queue),
        daemon=True
    )
    detection_thread.start()

    last_save = time.time()
    frame_num = 0

    # Get frame_event for signaling new frames
    with streams_lock:
        frame_event = active_streams.get(source_name, {}).get("frame_event")

    try:
        while True:
            with streams_lock:
                if source_name not in active_streams or not active_streams[source_name]["active"]:
                    break

            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    # For cameras, try to reconnect
                    print(f"Lost connection to camera {source_name}, attempting reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        print(f"Reconnect failed for {source_name}")
                        break
                    continue
                else:
                    # For video files, loop back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            frame_num += 1

            # Store frame for streaming using dedicated frames_lock (reduces contention)
            with frames_lock:
                shared_frames[source_name] = {
                    "frame": frame,  # No copy needed - we're the only writer
                    "frame_num": frame_num
                }

            # Signal that a new frame is available
            if frame_event:
                frame_event.set()

            # Feed detection queue (non-blocking)
            if not frame_queue.full():
                try:
                    frame_queue.put_nowait(frame.copy())
                except:
                    pass

            with streams_lock:
                detections = active_streams[source_name]["detections"].copy() if source_name in active_streams else []

            # Check zones and update timers (uses real wall-clock time internally)
            check_zones(detections, source_name)

            if time.time() - last_save > 2:
                save_presence()
                last_save = time.time()

            elapsed = time.time() - frame_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        with frames_lock:
            if source_name in shared_frames:
                del shared_frames[source_name]
        save_presence()
        print(f"Video processor stopped for {source_name}")


def apply_smooth_blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Apply smooth feathered blur to the entire bounding box (full body)"""
    h, w = frame.shape[:2]

    # Add padding for feather effect
    feather_size = 20
    pad_x1 = max(0, x1 - feather_size)
    pad_y1 = max(0, y1 - feather_size)
    pad_x2 = min(w, x2 + feather_size)
    pad_y2 = min(h, y2 + feather_size)

    if pad_x2 <= pad_x1 or pad_y2 <= pad_y1:
        return frame

    # Extract padded region
    region = frame[pad_y1:pad_y2, pad_x1:pad_x2].copy()
    if region.size == 0:
        return frame

    region_h, region_w = region.shape[:2]

    # Calculate blur kernel size based on bbox size
    bbox_size = max(x2 - x1, y2 - y1)
    blur_size = max(51, (bbox_size // 4) * 2 + 1)  # Ensure odd number

    # Apply strong Gaussian blur to entire region
    blurred_region = cv2.GaussianBlur(region, (blur_size, blur_size), 0)

    # Create gradient mask with smooth feathered edges
    mask = np.zeros((region_h, region_w), dtype=np.float32)

    # Calculate inner bbox position relative to padded region
    inner_x1 = x1 - pad_x1
    inner_y1 = y1 - pad_y1
    inner_x2 = x2 - pad_x1
    inner_y2 = y2 - pad_y1

    # Fill inner rectangle with full opacity
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0

    # Apply Gaussian blur to mask for smooth feathered edges
    mask = cv2.GaussianBlur(mask, (feather_size * 2 + 1, feather_size * 2 + 1), 0)

    # Normalize mask to ensure smooth transition
    mask = np.clip(mask, 0, 1)

    # Expand mask to 3 channels
    mask_3ch = np.stack([mask] * 3, axis=-1)

    # Blend original and blurred using mask
    blended = (blurred_region * mask_3ch + region * (1 - mask_3ch)).astype(np.uint8)

    # Apply back to frame
    frame[pad_y1:pad_y2, pad_x1:pad_x2] = blended

    return frame


def generate_frames(video_name: str):
    """Generate frames for streaming - reads from shared_frames written by processor"""
    last_frame_num = -1

    # Get the frame event for this stream
    with streams_lock:
        frame_event = active_streams.get(video_name, {}).get("frame_event")

    while True:
        # Check if stream is still active
        with streams_lock:
            if video_name not in active_streams or not active_streams[video_name]["active"]:
                break

        # Wait for new frame signal instead of polling (with timeout for cleanup)
        if frame_event:
            frame_event.wait(timeout=0.1)
            frame_event.clear()
        else:
            time.sleep(0.01)

        # Get frame data with minimal lock time
        with frames_lock:
            if video_name not in shared_frames:
                continue

            frame_data = shared_frames[video_name]
            frame_num = frame_data["frame_num"]

            # Skip if same frame
            if frame_num == last_frame_num:
                continue

            frame = frame_data["frame"].copy()
            last_frame_num = frame_num

        # Get detections separately
        with streams_lock:
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
            track_id = det.get("track_id")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display track ID and confidence
            label = f"#{track_id} {conf:.0%}" if track_id is not None else f"{conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 5),
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

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.post("/api/stream/{video_name:path}/start")
async def start_video_processing(video_name: str):
    """Start processing a video or camera (detection + zone tracking) in background"""
    # Support both video files and camera sources (camera:xxx)
    if is_camera_source(video_name):
        camera_id = video_name.replace("camera:", "")
        if camera_id not in cameras:
            raise HTTPException(status_code=404, detail="Camera not found")
    else:
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

    with streams_lock:
        if video_name in active_streams and active_streams[video_name]["active"]:
            return {"message": "Stream already active", "video": video_name}

        active_streams[video_name] = {
            "active": True,
            "detections": [],
            "frame_event": threading.Event()  # Event for frame synchronization
        }

    # Start background processing thread
    processor_thread = threading.Thread(
        target=video_processor,
        args=(video_name,),
        daemon=True
    )
    processor_thread.start()

    return {"message": "Stream started", "video": video_name}


@app.get("/api/stream/{video_name:path}")
async def video_stream(video_name: str):
    """Get video stream with detections overlay"""
    # Support both video files and camera sources (camera:xxx)
    if is_camera_source(video_name):
        camera_id = video_name.replace("camera:", "")
        if camera_id not in cameras:
            raise HTTPException(status_code=404, detail="Camera not found")
    else:
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

    # Auto-start processing if not already running
    with streams_lock:
        if video_name not in active_streams or not active_streams[video_name]["active"]:
            active_streams[video_name] = {
                "active": True,
                "detections": [],
                "frame_event": threading.Event()  # Event for frame synchronization
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
    import logging

    # Filter out static file requests from logs
    class StaticFileFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            # Filter out static assets, SVG, CSS, JS file requests
            if '/static/' in msg or '.svg' in msg or '.css' in msg or '.js' in msg:
                return False
            return True

    # Apply filter to uvicorn access logger
    logging.getLogger("uvicorn.access").addFilter(StaticFileFilter())

    uvicorn.run(app, host="0.0.0.0", port=8000)
