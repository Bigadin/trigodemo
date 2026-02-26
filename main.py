import json
import subprocess
import tempfile
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

from datetime import datetime, timezone
from typing import Optional

from store import persist, audit_event, append_event, load_audit_log, get_audit_entries, clear_audit_log, init as store_init

app = FastAPI(title="Zone Presence Tracker")

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class Utf8JsonMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        if ct.startswith("application/json") and "charset" not in ct:
            response.headers["content-type"] = "application/json; charset=utf-8"
        return response

app.add_middleware(Utf8JsonMiddleware)

# Paths
BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

VIDEOS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
store_init(DATA_DIR)

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
LIEUX_FILE = DATA_DIR / "lieux.json"
SITES_FILE = DATA_DIR / "sites.json"
BENEFITS_FILE = DATA_DIR / "benefits.json"

GRACE_PERIOD = 0.5
YOLO_CONFIDENCE = 0.45
TRACKING_REINFERENCE_INTERVAL = 30

ALLOWED_SKILL = "detection"
ALLOWED_SKILL_ITEM = "detection_presence"
ALLOWED_CATEGORY = "human::silhouette"

# Camera sources storage: {camera_id: {"type": "webcam"|"rtsp", "name": str, ...}}
cameras = {}

# Hierarchy storage
lieux = {}      # {lieu_id: {"name", "address", "description", "icon", "created_at"}}
sites = {}      # {site_id: {"name", "lieu_id", "address", "description", "icon", "created_at"}}
benefits = {}   # {benefit_id: {"name", "skill", "skill_item", "categories", "camera_id", "zone_polygons", "active", "canvas", "created_at"}}

# ==================== Counting Module State ====================
# Config per video — each zone has its own mode + flip_count:
# {video_name: {"zone_name": str, "zone_settings": {zone_name: {"mode": "simple"|"complex", "flip_count": int}}}}
counting_config = {}
# Runtime state for complex (blob/MOG2) counting per video:
counting_state = {}
counting_lock = threading.Lock()
COUNTING_FILE = DATA_DIR / "counting.json"

# Blob-based counting: MOG2 background models per (video, zone) — persistent once learned
counting_bg_models = {}       # {video_name: {zone_name: {"mog2": MOG2, "frame_count": int}}}
counting_debug_masks = {}     # {video_name: ndarray} - foreground mask for overlay

# Simple gradient-based counting: runtime state per video
simple_counting_state = {}

# Counting parameters (loaded from data/counting_params.json)
COUNTING_PARAMS_FILE = DATA_DIR / "counting_params.json"
COUNTING_PARAMS_DEFAULTS = {
    "min_blob_area": 5000,
    "blob_proximity_threshold": 80,
    "mog2_history": 500,
    "mog2_var_threshold": 50,
    "mog2_detect_shadows": True,
    "mog2_learning_frames": 30,
    "shadow_threshold": 200,
    "gaussian_blur_kernel": 5,
    "morphology_kernel_open": 7,
    "morphology_iterations_open": 1,
    "morphology_kernel_close": 7,
    "morphology_iterations_close": 1,
    "counting_line_position": 0.75,
    "grace_frames": 5,
    "counted_grace_frames": 1,
    "crossing_display_time": 2.0,
    "simple_gradient_threshold": 30,
    "simple_cooldown_frames": 10,
}
counting_params = {}


def load_counting_params():
    global counting_params
    counting_params = COUNTING_PARAMS_DEFAULTS.copy()
    if COUNTING_PARAMS_FILE.exists():
        with open(COUNTING_PARAMS_FILE, "r", encoding="utf-8") as f:
            user = json.load(f)
        counting_params.update(user)
    print(f"[COUNTING] Params loaded: {counting_params}")


load_counting_params()


def load_data():
    global zones_by_video, zone_timers, cameras, lieux, sites, benefits
    if ZONES_FILE.exists():
        with open(ZONES_FILE, "r", encoding="utf-8") as f:
            zones_by_video = json.load(f)
    if PRESENCE_FILE.exists():
        with open(PRESENCE_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for zone_name, value in loaded.items():
                if isinstance(value, (int, float)):
                    zone_timers[zone_name] = {"total_time": value, "last_occupied": None}
                else:
                    zone_timers[zone_name] = value
                    if "last_occupied" not in zone_timers[zone_name]:
                        zone_timers[zone_name]["last_occupied"] = None
    if CAMERAS_FILE.exists():
        with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
            cameras = json.load(f)
    if LIEUX_FILE.exists():
        with open(LIEUX_FILE, "r", encoding="utf-8") as f:
            lieux = json.load(f)
    if SITES_FILE.exists():
        with open(SITES_FILE, "r", encoding="utf-8") as f:
            sites = json.load(f)
    if BENEFITS_FILE.exists():
        with open(BENEFITS_FILE, "r", encoding="utf-8") as f:
            benefits = json.load(f)


def save_zones():
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones_by_video, f, indent=2, ensure_ascii=False)


def save_presence():
    with data_lock:
        with open(PRESENCE_FILE, "w", encoding="utf-8") as f:
            save_data = {k: {"total_time": v["total_time"]} for k, v in zone_timers.items()}
            json.dump(save_data, f, indent=2, ensure_ascii=False)


def save_cameras():
    with open(CAMERAS_FILE, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2, ensure_ascii=False)


def save_lieux():
    with open(LIEUX_FILE, "w", encoding="utf-8") as f:
        json.dump(lieux, f, indent=2, ensure_ascii=False)


def save_sites():
    with open(SITES_FILE, "w", encoding="utf-8") as f:
        json.dump(sites, f, indent=2, ensure_ascii=False)


def save_benefits():
    with open(BENEFITS_FILE, "w", encoding="utf-8") as f:
        json.dump(benefits, f, indent=2, ensure_ascii=False)


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


def _is_valid_zone_data(zone_data) -> bool:
    """Vérifie qu'une entrée zone a la structure attendue: { polygons: [[...]] }."""
    return isinstance(zone_data, dict) and "polygons" in zone_data and isinstance(zone_data.get("polygons"), list)


def _is_valid_video_zones(video_zones) -> bool:
    """Vérifie qu'une entrée video est un dict de zones valides (zone_name -> { polygons })."""
    if not isinstance(video_zones, dict):
        return False
    for zone_name, zone_data in video_zones.items():
        if not isinstance(zone_name, str) or not _is_valid_zone_data(zone_data):
            return False
    return True


def cleanup_zones_data():
    """
    Purge les reliquats : zones malformées, zone_timers orphelins,
    counting_config et counting_bg_models pour zones inexistantes.
    """
    global zones_by_video, zone_timers, counting_config, counting_bg_models
    with data_lock:
        # 1. Supprimer les entrées video malformées (ex: T1, T2 avec structure zone directe)
        to_remove = []
        for video_name, video_zones in zones_by_video.items():
            if not _is_valid_video_zones(video_zones):
                to_remove.append(video_name)
        for v in to_remove:
            del zones_by_video[v]

        # 2. Construire l'ensemble des zones valides (video -> zone_name)
        valid_zones = set()
        for video_zones in zones_by_video.values():
            for zone_name in video_zones:
                valid_zones.add(zone_name)

        # 3. Purger zone_timers des zones orphelines
        orphan_timers = [zn for zn in zone_timers if zn not in valid_zones]
        for zn in orphan_timers:
            del zone_timers[zn]

        # 4. Purger counting_config : zone_settings et zone_name invalides
        with counting_lock:
            for video_name in list(counting_config.keys()):
                cfg = counting_config[video_name]
                video_zones = zones_by_video.get(video_name, {})
                zone_settings = cfg.get("zone_settings", {})
                # Retirer les zone_settings pour zones inexistantes
                for zn in list(zone_settings.keys()):
                    if zn not in video_zones:
                        del zone_settings[zn]
                # Si zone_name active n'existe plus, la vider
                active_zn = cfg.get("zone_name")
                if active_zn and active_zn not in video_zones:
                    cfg["zone_name"] = ""
                    if zone_settings:
                        cfg["zone_name"] = next(iter(zone_settings.keys()), "")
                # Supprimer config video si plus de zones
                if not video_zones and video_name in counting_config:
                    del counting_config[video_name]

            # 5. Purger counting_bg_models pour (video, zone) inexistants
            for video_name in list(counting_bg_models.keys()):
                video_zones = zones_by_video.get(video_name, {})
                for zn in list(counting_bg_models.get(video_name, {}).keys()):
                    if zn not in video_zones:
                        del counting_bg_models[video_name][zn]
                if not counting_bg_models.get(video_name):
                    del counting_bg_models[video_name]

    if to_remove or orphan_timers:
        save_zones()
        save_presence()
        save_counting_config()
        print(f"[CLEANUP] Zones purgées: {len(to_remove)} vidéos malformées, {len(orphan_timers)} timers orphelins")


def load_counting_config():
    global counting_config
    if COUNTING_FILE.exists():
        with open(COUNTING_FILE, "r", encoding="utf-8") as f:
            counting_config = json.load(f)
    # Migrate old format: {video: {"zone_name", "flip_count"}} → new format with zone_settings
    for video_name, cfg in counting_config.items():
        if "zone_settings" not in cfg:
            zn = cfg.get("zone_name", "")
            fc = cfg.get("flip_count", 0)
            cfg["zone_settings"] = {zn: {"mode": "complex", "flip_count": fc}} if zn else {}
            # Keep zone_name as active zone
    save_counting_config()


def save_counting_config():
    with open(COUNTING_FILE, "w", encoding="utf-8") as f:
        json.dump(counting_config, f, indent=2, ensure_ascii=False)


def get_zone_settings(video_name: str, zone_name: str = None) -> dict | None:
    """Get counting settings for a specific zone. Returns {"mode", "flip_count"} or None."""
    config = counting_config.get(video_name, {})
    if zone_name is None:
        zone_name = config.get("zone_name")
    if not zone_name:
        return None
    return config.get("zone_settings", {}).get(zone_name)


def get_active_zone_mode(video_name: str) -> str:
    """Get mode of the currently active counting zone. Returns 'simple' or 'complex'."""
    settings = get_zone_settings(video_name)
    if settings:
        return settings.get("mode", "complex")
    return "complex"


load_counting_config()
cleanup_zones_data()


def _get_video_path_for_camera(cam_id: str) -> str | None:
    """Retourne le video_path utilisé par le frontend (path fichier ou camera:xxx)."""
    if cam_id not in cameras:
        return None
    c = cameras[cam_id]
    if c.get("type") in ("webcam", "rtsp"):
        return f"camera:{cam_id}"
    return c.get("path") or cam_id


def _sync_benefit_zones_on_startup():
    """Sync benefit zones to zones_by_video at startup: detection_presence + counting."""
    synced_det = 0
    synced_count = 0
    for bid, b in benefits.items():
        polys = b.get("zone_polygons", [])
        if not polys:
            continue
        cam_id = b.get("camera_id", "")
        if not cam_id:
            continue
        video_name = _get_video_path_for_camera(cam_id) or cam_id
        if video_name not in zones_by_video:
            zones_by_video[video_name] = {}

        # Detection présence (actifs uniquement)
        if b.get("skill") == ALLOWED_SKILL and b.get("skill_item") == ALLOWED_SKILL_ITEM:
            if not b.get("active", True):
                continue
            zones_by_video[video_name][bid] = {"polygons": polys}
            synced_det += 1
        # Comptage: tous (actifs ou non) pour permettre la sélection de zone
        elif b.get("skill") == "counting":
            zones_by_video[video_name][bid] = {"polygons": polys}
            synced_count += 1
            types = b.get("zone_polygon_types") or ["include"] * len(polys)
            for idx, (poly, pt) in enumerate(zip(polys, types)):
                if pt == "include" and len(poly) >= 3:
                    zone_key = f"{bid}:{idx}"
                    zones_by_video[video_name][zone_key] = {"polygons": [poly]}
                    synced_count += 1
    if synced_det or synced_count:
        save_zones()
        print(f"[STARTUP] Synced {synced_det} detection + {synced_count} counting zone(s) to zones_by_video")


_sync_benefit_zones_on_startup()


def compute_polygon_direction(all_points):
    """Compute the principal direction angle of a polygon using minimum bounding rectangle.
    Returns angle in degrees of the long axis (0-360)."""
    if len(all_points) < 3:
        return 0.0
    poly = Polygon(all_points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    # Find long axis among the 4 edges
    edge1 = np.array(coords[1]) - np.array(coords[0])
    edge2 = np.array(coords[2]) - np.array(coords[1])
    if np.linalg.norm(edge1) >= np.linalg.norm(edge2):
        long_axis = edge1
    else:
        long_axis = edge2
    angle = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
    return angle % 360


def get_counting_line(video_name: str):
    """Calculate counting line from ROI polygon's principal direction.
    Returns dict with line endpoints, direction vector, threshold, angle, roi_bounds or None.
    """
    config = counting_config.get(video_name)
    with counting_lock:
        complex_on = counting_state.get(video_name, {}).get("enabled", False)
        simple_on = simple_counting_state.get(video_name, {}).get("enabled", False)
        if not complex_on and not simple_on:
            return None

    if not config:
        return None

    zone_name = config.get("zone_name")
    if not zone_name:
        return None
    # Get flip_count from per-zone settings
    zs = get_zone_settings(video_name, zone_name)
    flip_count = zs.get("flip_count", 0) if zs else 0

    video_zones = zones_by_video.get(video_name, {})
    zone_data = video_zones.get(zone_name)
    if not zone_data or not zone_data.get("polygons"):
        return None

    all_points = [p for poly in zone_data["polygons"] for p in poly]
    if len(all_points) < 3:
        return None

    # Auto-compute direction from polygon shape + apply flip rotation
    base_angle = compute_polygon_direction(all_points)
    effective_angle = (base_angle + flip_count * 90) % 360

    # Direction unit vector
    angle_rad = np.radians(effective_angle)
    dir_x = float(np.cos(angle_rad))
    dir_y = float(np.sin(angle_rad))

    # Perpendicular unit vector (for the counting line)
    perp_x = -dir_y
    perp_y = dir_x

    # Project all points onto direction and perpendicular axes
    pts = np.array(all_points, dtype=np.float64)
    proj_dir = pts[:, 0] * dir_x + pts[:, 1] * dir_y
    proj_perp = pts[:, 0] * perp_x + pts[:, 1] * perp_y

    min_dir, max_dir = float(proj_dir.min()), float(proj_dir.max())
    min_perp, max_perp = float(proj_perp.min()), float(proj_perp.max())

    # Counting line position along the direction axis
    threshold = min_dir + counting_params["counting_line_position"] * (max_dir - min_dir)

    # Line endpoints: threshold along direction + full perpendicular extent
    line_start = (threshold * dir_x + min_perp * perp_x, threshold * dir_y + min_perp * perp_y)
    line_end = (threshold * dir_x + max_perp * perp_x, threshold * dir_y + max_perp * perp_y)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    return {
        "line_start": line_start,
        "line_end": line_end,
        "dir_x": dir_x,
        "dir_y": dir_y,
        "threshold": threshold,
        "angle": effective_angle,
        "roi_bounds": (min(xs), min(ys), max(xs), max(ys)),
    }


_counting_log_counter = 0  # throttle console logs
_counting_next_blob_id = 1  # auto-increment blob track IDs


def update_counting_blob(video_name: str, frame: np.ndarray):
    """Blob-based counting: background subtraction (MOG2) on ROI → contour detection → proximity tracking → line crossing.
    Independent from YOLO — works on raw pixels only."""
    global _counting_log_counter, _counting_next_blob_id

    line_info = get_counting_line(video_name)
    if not line_info:
        return

    # Check if MOG2 instance exists for this video+zone
    config_check = counting_config.get(video_name)
    zone_name_check = config_check.get("zone_name") if config_check else None
    if not zone_name_check:
        return
    video_models = counting_bg_models.get(video_name, {})
    zone_model = video_models.get(zone_name_check)
    if not zone_model or "mog2" not in zone_model:
        return

    dir_x = line_info["dir_x"]
    dir_y = line_info["dir_y"]
    threshold = line_info["threshold"]
    roi_left, roi_top, roi_right, roi_bottom = line_info["roi_bounds"]
    now = time.time()

    with counting_lock:
        state = counting_state.get(video_name)
        if not state or not state.get("enabled"):
            return

    # --- Step 1: Crop frame to ROI bounding box ---
    r_l, r_t = int(roi_left), int(roi_top)
    r_r, r_b = int(roi_right), int(roi_bottom)
    h, w = frame.shape[:2]
    r_l, r_t = max(0, r_l), max(0, r_t)
    r_r, r_b = min(w, r_r), min(h, r_b)
    if r_r <= r_l or r_b <= r_t:
        return

    crop = frame[r_t:r_b, r_l:r_r].copy()

    # --- Step 2: Apply polygon mask within the crop ---
    config = counting_config.get(video_name)
    if not config:
        return
    zone_name = config["zone_name"]
    video_zones = zones_by_video.get(video_name, {})
    zone_data = video_zones.get(zone_name)
    if not zone_data or not zone_data.get("polygons"):
        return

    # Build mask from polygons (shifted to crop coordinates)
    crop_h, crop_w = crop.shape[:2]
    poly_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for poly_pts in zone_data["polygons"]:
        if len(poly_pts) >= 3:
            shifted = np.array([[p[0] - r_l, p[1] - r_t] for p in poly_pts], dtype=np.int32)
            cv2.fillPoly(poly_mask, [shifted], 255)

    # Apply polygon mask to crop (black outside ROI polygon)
    masked_crop = cv2.bitwise_and(crop, crop, mask=poly_mask)

    # --- Step 3: Gaussian blur + MOG2 background subtraction ---
    k = counting_params["gaussian_blur_kernel"]
    blurred = cv2.GaussianBlur(masked_crop, (k, k), 0)
    mog2 = zone_model["mog2"]

    # Track frame count per zone: learn background for N frames, then freeze permanently
    frame_count = zone_model.get("frame_count", 0)
    zone_model["frame_count"] = frame_count + 1

    learning_frames = counting_params["mog2_learning_frames"]
    if frame_count < learning_frames:
        fg_mask = mog2.apply(blurred, learningRate=-1)  # Auto learning
        if frame_count == learning_frames - 1:
            print(f"[COUNTING-BLOB] Background model frozen after {learning_frames} frames for {video_name}")
    else:
        fg_mask = mog2.apply(blurred, learningRate=0)  # Frozen background

    # --- Step 4: Threshold to remove shadows (MOG2 marks shadows as 127) + morphology ---
    _, fg_mask = cv2.threshold(fg_mask, counting_params["shadow_threshold"], 255, cv2.THRESH_BINARY)
    # Also apply the polygon mask to remove any edge artifacts outside the ROI
    fg_mask = cv2.bitwise_and(fg_mask, poly_mask)

    # Morphological operations: open (separate touching blobs) then close (fill gaps)
    mk_open = counting_params["morphology_kernel_open"]
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk_open, mk_open))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k_open, iterations=counting_params["morphology_iterations_open"])

    mk_close = counting_params["morphology_kernel_close"]
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk_close, mk_close))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close, iterations=counting_params["morphology_iterations_close"])

    # --- Step 5: Find contours and filter by area ---
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_centers = []  # [(cx_global, cy_global, area), ...]
    blob_contours_global = []  # contours shifted back to global frame coords

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < counting_params["min_blob_area"]:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        # Centroid in crop coordinates → global coordinates
        cx_crop = M["m10"] / M["m00"]
        cy_crop = M["m01"] / M["m00"]
        cx_global = cx_crop + r_l
        cy_global = cy_crop + r_t
        blob_centers.append((cx_global, cy_global, area))

        # Shift contour back to global coords for debug drawing
        cnt_global = cnt.copy()
        cnt_global[:, :, 0] += r_l
        cnt_global[:, :, 1] += r_t
        blob_contours_global.append(cnt_global)

    # Store debug mask (in global frame coords: place into full-size mask)
    debug_mask = np.zeros((h, w), dtype=np.uint8)
    debug_mask[r_t:r_b, r_l:r_r] = fg_mask
    counting_debug_masks[video_name] = debug_mask

    # --- Step 6: Proximity-based blob tracking + line crossing ---
    with counting_lock:
        state = counting_state.get(video_name)
        if not state or not state.get("enabled"):
            return

        tracked = state["tracked_objects"]
        current_ids = set()
        debug_objects = {}

        # Clean up old crossing events (keep last 2 seconds)
        recent = state.get("recent_crossings", [])
        recent = [ev for ev in recent if now - ev["time"] < counting_params["crossing_display_time"]]

        # Match blob centers to existing tracked objects by proximity
        used_track_ids = set()
        matched_blobs = []  # [(blob_idx, track_id_str)]

        for i, (cx, cy, area) in enumerate(blob_centers):
            best_tid = None
            best_dist = counting_params["blob_proximity_threshold"]
            for tid, obj in tracked.items():
                if tid in used_track_ids:
                    continue
                dx = cx - obj["prev_cx"]
                dy = cy - obj["prev_cy"]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            if best_tid:
                matched_blobs.append((i, best_tid))
                used_track_ids.add(best_tid)
            else:
                # New blob: assign new ID
                new_id = f"b{_counting_next_blob_id}"
                _counting_next_blob_id += 1
                matched_blobs.append((i, new_id))

        for blob_idx, track_id_str in matched_blobs:
            cx, cy, area = blob_centers[blob_idx]
            current_ids.add(track_id_str)

            # Projection onto direction axis
            curr_proj = cx * dir_x + cy * dir_y

            if track_id_str not in tracked:
                tracked[track_id_str] = {"prev_cx": cx, "prev_cy": cy, "prev_proj": curr_proj, "counted": False}
                debug_objects[track_id_str] = {"cx": cx, "cy": cy, "proj": curr_proj, "counted": False, "area": area}
                print(f"[COUNTING-BLOB] NEW #{track_id_str} at ({cx:.0f},{cy:.0f}) area={area:.0f} proj={curr_proj:.0f} thr={threshold:.0f}")
                continue

            obj = tracked[track_id_str]
            prev_proj = obj["prev_proj"]

            if not obj["counted"]:
                crossed = (prev_proj < threshold and curr_proj >= threshold) or \
                          (prev_proj > threshold and curr_proj <= threshold)
                if crossed:
                    state["count"] += 1
                    obj["counted"] = True
                    recent.append({"track_id": track_id_str, "cx": cx, "cy": cy, "time": now})
                    print(f"[COUNTING-BLOB] CROSSED #{track_id_str} prev={prev_proj:.0f} -> curr={curr_proj:.0f} thr={threshold:.0f} COUNT={state['count']}")

            debug_objects[track_id_str] = {
                "cx": cx, "cy": cy,
                "prev_cx": obj["prev_cx"], "prev_cy": obj["prev_cy"],
                "proj": curr_proj, "counted": obj["counted"], "area": area,
            }

            obj["prev_cx"] = cx
            obj["prev_cy"] = cy
            obj["prev_proj"] = curr_proj

        # Clean up disappeared blobs (shorter grace for already-counted blobs)
        disappeared = set(tracked.keys()) - current_ids
        to_delete = []
        for tid in disappeared:
            obj = tracked[tid]
            miss_count = obj.get("miss_count", 0) + 1
            grace = counting_params["counted_grace_frames"] if obj.get("counted") else counting_params["grace_frames"]
            if miss_count > grace:
                to_delete.append(tid)
            else:
                obj["miss_count"] = miss_count
        for tid in to_delete:
            del tracked[tid]

        # Reset miss_count for seen objects
        for tid in current_ids:
            if tid in tracked:
                tracked[tid]["miss_count"] = 0

        state["debug_objects"] = debug_objects
        state["debug_contours"] = blob_contours_global
        state["recent_crossings"] = recent

        # Periodic diagnostic log
        _counting_log_counter += 1
        if _counting_log_counter % 60 == 0:
            print(f"[COUNTING-BLOB] blobs={len(blob_centers)} tracked={len(tracked)} count={state['count']} thr={threshold:.0f} roi=({roi_left:.0f},{roi_top:.0f})-({roi_right:.0f},{roi_bottom:.0f}) angle={line_info['angle']:.0f}°")


def update_counting_simple(video_name: str, frame: np.ndarray):
    """Simple gradient counting: sample pixels along counting line, detect inter-frame spikes."""
    line_info = get_counting_line(video_name)
    if not line_info:
        return

    with counting_lock:
        state = simple_counting_state.get(video_name)
        if not state or not state.get("enabled"):
            return

    ls = line_info["line_start"]
    le = line_info["line_end"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sample ~100 evenly spaced pixels along the counting line
    num_samples = 100
    xs = np.linspace(ls[0], le[0], num_samples).astype(int)
    ys = np.linspace(ls[1], le[1], num_samples).astype(int)
    h, w = gray.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    current_pixels = gray[ys, xs].astype(np.float32)

    with counting_lock:
        prev_pixels = state.get("prev_line_pixels")
        cooldown = state.get("cooldown_remaining", 0)

        if prev_pixels is not None and len(prev_pixels) == len(current_pixels):
            diff = float(np.mean(np.abs(current_pixels - prev_pixels)))
            state["current_diff"] = diff

            if cooldown > 0:
                state["cooldown_remaining"] = cooldown - 1
            elif diff > counting_params.get("simple_gradient_threshold", 30):
                state["count"] += 1
                state["cooldown_remaining"] = counting_params.get("simple_cooldown_frames", 10)
                print(f"[COUNTING-SIMPLE] {video_name} SPIKE diff={diff:.1f} COUNT={state['count']}")
        else:
            state["current_diff"] = 0.0

        state["prev_line_pixels"] = current_pixels


# ==================== Audit / Event Log System ====================
# Journal centralisé dans store.py — load_audit_log au démarrage


def _seed_demo_events():
    """Seed historical demo events if the log is empty or very small (< 5 entries)."""
    entries = get_audit_entries()
    if len(entries) > 5:
        return  # Already has data, skip

    from datetime import timedelta
    now = datetime.now(timezone.utc)

    demo = [
        # Day -7: system setup
        (-7, 0, 0, "system", "startup", "Application démarrée — déploiement initial", "success"),
        (-7, 0, 2, "system", "config", "Modèle YOLO human.pt chargé (CPU)", "info"),
        # Day -6: cameras configured
        (-6, 9, 0, "camera", "added", "Caméra « popo » ajoutée (RTSP — rtsp://…/movie)", "success",
         {"camera_id": "rtsp_popo", "type": "rtsp"}),
        (-6, 9, 5, "camera", "added", "Caméra « sqd » ajoutée (Webcam — device 0)", "success",
         {"camera_id": "webcam_sqd", "type": "webcam"}),
        # Day -6: videos uploaded
        (-6, 10, 0, "video", "upload", "Vidéo « entr1.mp4 » uploadée", "success", {"filename": "entr1.mp4"}),
        (-6, 10, 12, "video", "upload", "Vidéo « videoplayback.mp4 » uploadée", "success", {"filename": "videoplayback.mp4"}),
        (-6, 10, 30, "video", "upload", "Vidéo « video_01.mp4 » uploadée", "success", {"filename": "video_01.mp4"}),
        (-6, 10, 45, "video", "upload", "Vidéo « video_04.mp4 » uploadée", "success", {"filename": "video_04.mp4"}),
        # Day -5: zones created on entr1.mp4
        (-5, 8, 30, "zone", "created", "Zone « Zone A » créée sur entr1.mp4 (2 forme(s))", "success",
         {"zone": "Zone A", "video": "entr1.mp4", "polygons": 2}),
        (-5, 8, 35, "zone", "created", "Zone « Zone B » créée sur entr1.mp4 (1 forme(s))", "success",
         {"zone": "Zone B", "video": "entr1.mp4", "polygons": 1}),
        # Day -5: zones on videoplayback
        (-5, 9, 0, "zone", "created", "Zone « Zone A » créée sur videoplayback.mp4 (1 forme(s))", "success",
         {"zone": "Zone A", "video": "videoplayback.mp4", "polygons": 1}),
        (-5, 9, 10, "zone", "created", "Zone « Contrôle Pièces » créée sur videoplayback.mp4 (1 forme(s))", "success",
         {"zone": "Contrôle Pièces", "video": "videoplayback.mp4", "polygons": 1}),
        # Day -5: first detection session
        (-5, 9, 30, "stream", "started", "Stream démarré : entr1.mp4", "success", {"source": "entr1.mp4"}),
        (-5, 9, 31, "detection", "occupancy", "Zone A (entr1.mp4) — première détection de présence", "info",
         {"zone": "Zone A", "video": "entr1.mp4"}),
        (-5, 10, 15, "stream", "stopped", "Stream arrêté : entr1.mp4", "info", {"source": "entr1.mp4"}),
        # Day -4: zone edits
        (-4, 14, 0, "zone", "edited", "Zone « Zone A » éditée sur entr1.mp4 (2 forme(s))", "info",
         {"zone": "Zone A", "video": "entr1.mp4", "polygons": 2}),
        (-4, 14, 5, "zone", "created", "Zone « T2 » créée sur video_01.mp4 (1 forme(s))", "success",
         {"zone": "T2", "video": "video_01.mp4", "polygons": 1}),
        # Day -4: detection session
        (-4, 14, 20, "stream", "started", "Stream démarré : videoplayback.mp4", "success", {"source": "videoplayback.mp4"}),
        (-4, 14, 22, "detection", "occupancy", "Zone A (videoplayback.mp4) — présence détectée", "info",
         {"zone": "Zone A", "video": "videoplayback.mp4"}),
        (-4, 15, 0, "blur", "toggled", "Floutage activé", "info", {"enabled": True}),
        (-4, 15, 45, "stream", "stopped", "Stream arrêté : videoplayback.mp4", "info", {"source": "videoplayback.mp4"}),
        # Day -3: camera zone
        (-3, 11, 0, "zone", "created", "Zone « Z2 » créée sur camera:webcam_sqd (1 forme(s))", "success",
         {"zone": "Z2", "video": "camera:webcam_sqd", "polygons": 1}),
        (-3, 11, 15, "zone", "created", "Zone « T2 » créée sur camera:rtsp_popo (1 forme(s))", "success",
         {"zone": "T2", "video": "camera:rtsp_popo", "polygons": 1}),
        (-3, 11, 30, "stream", "started", "Stream démarré : camera:webcam_sqd", "success", {"source": "camera:webcam_sqd"}),
        (-3, 11, 32, "detection", "occupancy", "Z2 (webcam sqd) — présence détectée", "info",
         {"zone": "Z2", "video": "camera:webcam_sqd"}),
        (-3, 12, 0, "stream", "stopped", "Stream arrêté : camera:webcam_sqd", "info", {"source": "camera:webcam_sqd"}),
        # Day -2: more work
        (-2, 9, 0, "zone", "created", "Zone « de » créée sur video_04.mp4 (1 forme(s))", "success",
         {"zone": "de", "video": "video_04.mp4", "polygons": 1}),
        (-2, 10, 0, "stream", "started", "Stream démarré : video_04.mp4", "success", {"source": "video_04.mp4"}),
        (-2, 10, 5, "detection", "occupancy", "de (video_04.mp4) — présence détectée", "info",
         {"zone": "de", "video": "video_04.mp4"}),
        (-2, 11, 30, "stream", "stopped", "Stream arrêté : video_04.mp4", "info", {"source": "video_04.mp4"}),
        (-2, 14, 0, "stream", "started", "Stream démarré : entr1.mp4", "success", {"source": "entr1.mp4"}),
        (-2, 14, 2, "detection", "occupancy", "Zone A (entr1.mp4) — présence continue détectée (cumul 1200s+)", "info",
         {"zone": "Zone A", "video": "entr1.mp4"}),
        (-2, 14, 3, "detection", "occupancy", "Zone B (entr1.mp4) — présence détectée (cumul 997s)", "info",
         {"zone": "Zone B", "video": "entr1.mp4"}),
        (-2, 15, 0, "blur", "toggled", "Floutage désactivé", "info", {"enabled": False}),
        (-2, 16, 0, "stream", "stopped_all", "Tous les streams arrêtés (1)", "warn", {"count": 1}),
        # Day -1: edits & resets
        (-1, 8, 0, "zone", "edited", "Zone « Zone B » éditée sur entr1.mp4 (1 forme(s))", "info",
         {"zone": "Zone B", "video": "entr1.mp4", "polygons": 1}),
        (-1, 8, 30, "zone", "reset", "Timer de « Contrôle Pièces » réinitialisé", "info", {"zone": "Contrôle Pièces"}),
        (-1, 9, 0, "stream", "started", "Stream démarré : camera:rtsp_popo", "success", {"source": "camera:rtsp_popo"}),
        (-1, 9, 5, "detection", "occupancy", "T2 (RTSP popo) — présence détectée", "info",
         {"zone": "T2", "video": "camera:rtsp_popo"}),
        (-1, 10, 30, "stream", "stopped", "Stream arrêté : camera:rtsp_popo", "info", {"source": "camera:rtsp_popo"}),
        # Today: startup
        (0, 0, 0, "system", "startup", "Application démarrée", "success"),
    ]

    for entry in demo:
        days_offset = entry[0]
        hour = entry[1]
        minute = entry[2]
        category = entry[3]
        action = entry[4]
        detail = entry[5]
        level = entry[6]
        meta = entry[7] if len(entry) > 7 else None

        ts = (now + timedelta(days=days_offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)

        ev = {
            "ts": ts.isoformat(),
            "category": category,
            "action": action,
            "detail": detail,
            "level": level,
        }
        if meta:
            ev["meta"] = meta
        append_event(ev)


load_audit_log()
_seed_demo_events()


# ==================== Performance Metrics (demo) ====================

import math
import random as _rnd

_metrics_seed = 42  # fixed seed for consistent demo


def _generate_demo_metrics(points: int = 200) -> dict:
    """
    Generate stepped / staircase style demo metrics (like Chrome DevTools perf monitor).
    Phases: idle → cam start → heavy load → cam stop → idle → cam restart → medium load
    Values jump in steps (not smooth curves) and hold for a few ticks before changing.
    """
    now = datetime.now(timezone.utc)
    rng = _rnd.Random(_metrics_seed)
    interval_s = 15  # one point every 15s

    series = {
        "GPU Usage":         {"unit": "%",  "color": "#22c55e", "min": 0, "max": 100, "data": []},
        "Latency Cam":       {"unit": "ms", "color": "#f59e0b", "min": 0, "max": 250, "data": []},
        "YOLO Inference":    {"unit": "ms", "color": "#3b82f6", "min": 0, "max": 180, "data": []},
        "FPS":               {"unit": "fps","color": "#a855f7", "min": 0, "max": 60,  "data": []},
        "Memory":            {"unit": "Mo", "color": "#ef4444", "min": 0, "max": 2048,"data": []},
        "Active Detections": {"unit": "",   "color": "#06b6d4", "min": 0, "max": 15,  "data": []},
    }

    # ---- Phase definitions (ratio of total points) ----
    # Each phase: (start_pct, end_pct, label)
    #  idle_boot | cam1_start | heavy_2cams | cam_stop | idle_mid | cam_restart | medium_tail
    phases = [
        (0.00, 0.10, "idle"),       # boot / no stream
        (0.10, 0.12, "ramp_up"),    # starting first camera
        (0.12, 0.30, "one_cam"),    # 1 camera active
        (0.30, 0.32, "ramp_up2"),   # starting second camera
        (0.32, 0.52, "two_cams"),   # 2 cameras = heavy
        (0.52, 0.55, "ramp_down"),  # stopping cams
        (0.55, 0.68, "idle2"),      # all cameras off
        (0.68, 0.70, "ramp_up3"),   # restarting 1 cam
        (0.70, 0.88, "one_cam2"),   # 1 cam medium load
        (0.88, 0.90, "ramp_down2"), # stopping
        (0.90, 1.01, "idle3"),      # idle tail
    ]

    def get_phase(pct):
        for (s, e, label) in phases:
            if s <= pct < e:
                return label
        return "idle3"

    # Target values per phase: (gpu, latency, yolo_ms, fps, mem_Mo, detections)
    targets = {
        "idle":        (0,   0,   0,   0,  310, 0),
        "ramp_up":     (25,  35,  45,  12, 480, 0),
        "one_cam":     (42,  55,  38,  24, 620, 3),
        "ramp_up2":    (58,  70,  52,  20, 780, 4),
        "two_cams":    (72,  95,  68,  18, 950, 7),
        "ramp_down":   (30,  25,  20,  10, 700, 1),
        "idle2":       (0,   0,   0,   0,  340, 0),
        "ramp_up3":    (20,  30,  40,  10, 500, 0),
        "one_cam2":    (38,  48,  35,  25, 580, 4),
        "ramp_down2":  (15,  12,  10,  5,  420, 0),
        "idle3":       (0,   0,   0,   0,  320, 0),
    }

    # State: current values (start idle)
    gpu = 0.0; lat = 0.0; yolo = 0.0; fps = 0.0; mem = 310.0; det = 0.0
    # Step hold: values hold for N ticks then jump (staircase effect)
    hold_counter = 0
    hold_ticks = rng.randint(2, 5)
    step_gpu = gpu; step_lat = lat; step_yolo = yolo
    step_fps = fps; step_mem = mem; step_det = det

    for i in range(points):
        t = now - __import__('datetime').timedelta(seconds=(points - i) * interval_s)
        ts = t.isoformat()

        pct = i / points
        phase = get_phase(pct)
        tgt = targets[phase]

        # Pull toward target with noise (but compute new target each tick)
        pull = 0.25  # how fast we snap to target
        noise_scale = 0.15

        gpu += (tgt[0] - gpu) * pull + rng.gauss(0, max(1, tgt[0] * noise_scale))
        lat += (tgt[1] - lat) * pull + rng.gauss(0, max(1, tgt[1] * noise_scale))
        yolo += (tgt[2] - yolo) * pull + rng.gauss(0, max(1, tgt[2] * noise_scale))
        fps += (tgt[3] - fps) * pull + rng.gauss(0, max(0.5, tgt[3] * noise_scale))
        mem += (tgt[4] - mem) * pull * 0.6 + rng.gauss(0, 12)
        det += (tgt[5] - det) * pull + rng.gauss(0, max(0.3, tgt[5] * 0.2))

        # Clamp
        gpu = max(0, min(98, gpu))
        lat = max(0, min(240, lat))
        yolo = max(0, min(170, yolo))
        fps = max(0, min(55, fps))
        mem = max(180, min(1800, mem))
        det = max(0, min(14, det))

        # Force idle phases to truly 0 for GPU/lat/yolo/fps/det
        if phase in ("idle", "idle2", "idle3"):
            gpu = max(0, gpu * 0.6)
            lat = max(0, lat * 0.5)
            yolo = max(0, yolo * 0.5)
            fps = max(0, fps * 0.5)
            det = max(0, det * 0.5)

        # Staircase: hold values for N ticks then snap to new computed value
        hold_counter += 1
        if hold_counter >= hold_ticks:
            hold_counter = 0
            hold_ticks = rng.randint(2, 5)
            step_gpu = round(gpu, 1)
            step_lat = round(lat, 1)
            step_yolo = round(yolo, 1)
            step_fps = round(fps, 1)
            step_mem = round(mem, 0)
            step_det = round(max(0, det), 0)

        series["GPU Usage"]["data"].append({"t": ts, "v": step_gpu})
        series["Latency Cam"]["data"].append({"t": ts, "v": step_lat})
        series["YOLO Inference"]["data"].append({"t": ts, "v": step_yolo})
        series["FPS"]["data"].append({"t": ts, "v": step_fps})
        series["Memory"]["data"].append({"t": ts, "v": step_mem})
        series["Active Detections"]["data"].append({"t": ts, "v": step_det})

    return series


# Arbre skills/catégories (modal + spec). Backend actif: detection_presence + human::silhouette.
SKILLS_CONFIG = {
    "skills": [
        {
            "key": "detection",
            "label": "Detection",
            "icon": "/static/assets_youn/SvIcons/SVGnew/Yclassify.svg",
            "items": [
                {"id": "detection_presence", "label": "Presence / Absence", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"},
                {"id": "detection_linecross", "label": "Franchissement ligne", "icon": "/static/assets_youn/SvIcons/SVGnew/Ylinecross.svg"},
                {"id": "detection_zone", "label": "Detection zone", "icon": "/static/assets_youn/SvIcons/SVGnew/Yzonedetect.svg"},
            ],
        },
        {
            "key": "counting",
            "label": "Comptage",
            "icon": "/static/assets_youn/SvIcons/SVGnew/Ycounting.svg",
            "items": [
                {"id": "counting_people", "label": "Comptage personnes", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg"},
                {"id": "counting_objects", "label": "Comptage objets", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycounting.svg"},
                {"id": "counting_zone", "label": "Comptage zone", "icon": "/static/assets_youn/SvIcons/SVGnew/square-area-svgrepo-com.svg"},
            ],
        },
        {
            "key": "heatmap",
            "label": "Heatmap",
            "icon": "/static/assets_youn/SvIcons/SVGnew/Yheatmap.svg",
            "items": [
                {"id": "heatmap_density", "label": "Densite de flux", "icon": "/static/assets_youn/SvIcons/SVGnew/grid-svgrepo-com.svg"},
                {"id": "heatmap_presence", "label": "Heatmap presence", "icon": "/static/assets_youn/SvIcons/SVGnew/Yheatmapdense.svg"},
                {"id": "heatmap_trajectory", "label": "Heatmap trajectoires", "icon": "/static/assets_youn/SvIcons/SVGnew/Ytraj.svg"},
            ],
        },
        {
            "key": "quality",
            "label": "Qualite",
            "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg",
            "items": [
                {"id": "quality_fissure", "label": "Fissure", "icon": "/static/assets_youn/SvIcons/SVGnew/Yfissure.svg"},
                {"id": "quality_humidity", "label": "Humidite", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg"},
                {"id": "quality_check", "label": "Qualite generale", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg"},
            ],
        },
    ],
    "categories_by_skill": {
        "detection": [
            {
                "key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg",
                "items": [
                    {"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"},
                    {"id": "visage", "label": "Visage", "icon": "/static/assets_youn/SvIcons/SVGnew/Yface.svg"},
                    {"id": "foule", "label": "Foule", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"},
                ],
            },
            {
                "key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg",
                "items": [
                    {"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"},
                    {"id": "velo", "label": "Velo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"},
                    {"id": "public_transport", "label": "Transport public", "icon": "/static/assets_youn/SvIcons/SVGnew/Ypublic transport.svg"},
                    {"id": "avion", "label": "Avion", "icon": "/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg"},
                    {"id": "moto", "label": "Moto", "icon": "/static/assets_youn/SvIcons/SVGnew/motorcycle.svg"},
                ],
            },
        ],
        "counting": [
            {
                "key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg",
                "items": [
                    {"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"},
                    {"id": "visage", "label": "Visage", "icon": "/static/assets_youn/SvIcons/SVGnew/Yface.svg"},
                    {"id": "foule", "label": "Foule", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"},
                ],
            },
            {
                "key": "transport", "label": "Transport", "icon": "/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg",
                "items": [
                    {"id": "voiture", "label": "Voiture", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycar.svg"},
                    {"id": "velo", "label": "Velo", "icon": "/static/assets_youn/SvIcons/SVGnew/Ybike.svg"},
                    {"id": "public_transport", "label": "Transport public", "icon": "/static/assets_youn/SvIcons/SVGnew/Ypublic transport.svg"},
                    {"id": "avion", "label": "Avion", "icon": "/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg"},
                    {"id": "moto", "label": "Moto", "icon": "/static/assets_youn/SvIcons/SVGnew/motorcycle.svg"},
                ],
            },
        ],
        "heatmap": [
            {
                "key": "human", "label": "Humain", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg",
                "items": [
                    {"id": "silhouette", "label": "Silhouette", "icon": "/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg"},
                    {"id": "visage", "label": "Visage", "icon": "/static/assets_youn/SvIcons/SVGnew/Yface.svg"},
                    {"id": "foule", "label": "Foule", "icon": "/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg"},
                ],
            },
            {
                "key": "object", "label": "Objet", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg",
                "items": [
                    {"id": "encombrement", "label": "Encombrement", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"},
                    {"id": "zone_encombre", "label": "Zone encombre", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"},
                    {"id": "fissure", "label": "Fissure", "icon": "/static/assets_youn/SvIcons/SVGnew/Yfissure.svg"},
                    {"id": "humidity", "label": "Humidite", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg"},
                    {"id": "qualitycheck", "label": "Qualite generale", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg"},
                ],
            },
        ],
        "quality": [
            {
                "key": "object", "label": "Objet", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg",
                "items": [
                    {"id": "encombrement", "label": "Encombrement", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"},
                    {"id": "zone_encombre", "label": "Zone encombre", "icon": "/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg"},
                    {"id": "fissure", "label": "Fissure", "icon": "/static/assets_youn/SvIcons/SVGnew/Yfissure.svg"},
                    {"id": "humidity", "label": "Humidite", "icon": "/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg"},
                    {"id": "qualitycheck", "label": "Qualite generale", "icon": "/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg"},
                ],
            },
        ],
    },
}




@app.get("/api/skills")
async def get_skills():
    """Retourne les skills et catégories supportés par le backend (détection présence humain)."""
    return SKILLS_CONFIG

@app.get("/api/solution-spec")
async def get_solution_spec():
    """Retourne la spec centrale de la solution (skills, categories, mapping)."""
    spec_path = Path(__file__).parent / "static" / "config" / "solution-spec.json"
    if spec_path.exists():
        with open(spec_path, encoding="utf-8") as f:
            return json.load(f)
    return {"error": "solution-spec.json not found"}

@app.get("/api/metrics")
async def get_metrics(points: int = 120):
    """Return demo performance metrics for the monitoring chart."""
    return _generate_demo_metrics(min(points, 500))


@app.get("/api/logs")
async def get_audit_logs(limit: int = 200, offset: int = 0, category: str = ""):
    """Return recent audit log entries (newest first)."""
    entries = get_audit_entries()
    # Filter by category/entity if provided
    if category:
        cats = set(c.strip() for c in category.split(","))
        entries = [e for e in entries if e.get("category") in cats or e.get("entity") in cats]
    # Newest first
    entries.reverse()
    total = len(entries)
    page = entries[offset: offset + limit]
    return {"logs": page, "total": total}


@app.delete("/api/logs")
async def clear_audit_logs():
    """Clear the audit log."""
    clear_audit_log()
    audit_event("system", "logs_cleared", "Journal d'audit effacé", "warn")
    return {"message": "Logs cleared"}


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


# Optimisation vidéo à l'upload (réduction taille)
VIDEO_OPTIMIZE_MAX_WIDTH = 1280
VIDEO_OPTIMIZE_MAX_HEIGHT = 720
VIDEO_OPTIMIZE_CRF = 28  # Qualité H.264 (18-28 = bon compromis)


def _optimize_video(src: Path, dst: Path) -> bool:
    """Réencode la vidéo avec ffmpeg pour réduire la taille (résolution + bitrate)."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-vf", f"scale='min({VIDEO_OPTIMIZE_MAX_WIDTH},iw)':'min({VIDEO_OPTIMIZE_MAX_HEIGHT},ih)':force_original_aspect_ratio=decrease",
                "-c:v", "libx264", "-crf", str(VIDEO_OPTIMIZE_CRF),
                "-preset", "fast", "-c:a", "aac", "-b:a", "128k", "-f", "mp4",
                str(dst)
            ],
            capture_output=True,
            timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@app.get("/api/videos")
async def list_videos():
    return {"videos": _list_video_files()}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    ext = Path(file.filename).suffix or ".mp4"
    # Optimisation: sortie en .mp4 (H.264) si ffmpeg disponible
    out_optimized = VIDEOS_DIR / (Path(file.filename).stem + ".mp4")
    out_fallback = VIDEOS_DIR / file.filename

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if _optimize_video(tmp_path, out_optimized):
            final_filename = out_optimized.name
        else:
            with open(out_fallback, "wb") as f:
                f.write(content)
            final_filename = file.filename
    finally:
        tmp_path.unlink(missing_ok=True)

    audit_event("video", "upload", f"Vidéo « {final_filename} » uploadée", "success",
                {"filename": final_filename})
    return {"message": "Video uploaded", "filename": final_filename}


@app.post("/api/videos/optimize/{video_name:path}")
async def optimize_existing_video(video_name: str):
    """Optimise une vidéo déjà présente dans le dossier videos/."""
    if is_camera_source(video_name):
        raise HTTPException(status_code=400, detail="Cannot optimize camera streams")
    src = VIDEOS_DIR / video_name
    if not src.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    stem = Path(video_name).stem
    out_name = stem + ".mp4"
    tmp_path = VIDEOS_DIR / (stem + "_tmp_opt.mp4")
    if not _optimize_video(src, tmp_path):
        raise HTTPException(
            status_code=500,
            detail="Optimization failed. Ensure ffmpeg is installed and in PATH."
        )
    src.unlink()
    tmp_path.rename(VIDEOS_DIR / out_name)

    # Migration: si le nom a changé (ex. .webm -> .mp4), copier zones et counting
    if video_name != out_name:
        with data_lock:
            if video_name in zones_by_video:
                zones_by_video[out_name] = zones_by_video.pop(video_name)
                save_zones()
        with counting_lock:
            if video_name in counting_config:
                counting_config[out_name] = counting_config.pop(video_name)
                save_counting_config()

    audit_event("video", "edited", f"Vidéo « {video_name} » optimisée", "success", {"filename": out_name})
    return {"message": "Video optimized", "filename": out_name}


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

    persist(save_zones, "zone", "created", f"Zone « {zone_name} » créée sur {video_name} ({poly_count} forme(s))", "success",
            meta={"zone": zone_name, "video": video_name, "polygons": len(zone.polygons)})
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

    persist(save_zones, "zone", "updated", f"Zone « {zone_name} » éditée sur {video_name} ({len(update.polygons)} forme(s))", "info",
            meta={"zone": zone_name, "video": video_name, "polygons": len(update.polygons)})
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

        # Nettoyer counting_config et counting_bg_models pour cette vidéo/zone
        with counting_lock:
            if video_name in counting_config:
                cfg = counting_config[video_name]
                zs = cfg.get("zone_settings", {})
                if zone_name in zs:
                    del zs[zone_name]
                if cfg.get("zone_name") == zone_name:
                    cfg["zone_name"] = next((zn for zn in zs if zn), "")
                save_counting_config()
            if video_name in counting_bg_models and zone_name in counting_bg_models[video_name]:
                del counting_bg_models[video_name][zone_name]
                if not counting_bg_models[video_name]:
                    del counting_bg_models[video_name]

    def _save():
        save_zones()
        save_presence()
    persist(_save, "zone", "deleted", f"Zone « {zone_name} » supprimée de {video_name}", "warn",
            meta={"zone": zone_name, "video": video_name})
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

            # Nettoyer counting_config et counting_bg_models pour cette vidéo
            with counting_lock:
                if video_name in counting_config:
                    del counting_config[video_name]
                    save_counting_config()
                if video_name in counting_bg_models:
                    del counting_bg_models[video_name]

    def _save():
        save_zones()
        save_presence()
    persist(_save, "zone", "deleted", f"Toutes les zones supprimées pour {video_name}", "warn",
            meta={"video": video_name})
    return {"message": "All zones deleted for video"}


@app.post("/api/zones/cleanup")
async def cleanup_zones_api():
    """Purge manuelle des reliquats (zones malformées, timers orphelins, counting)."""
    cleanup_zones_data()
    return {"message": "Zones cleanup completed"}


@app.post("/api/zones/reset")
async def reset_all_timers():
    with data_lock:
        for zone_name in zone_timers:
            zone_timers[zone_name]["total_time"] = 0
            zone_timers[zone_name]["occupy_start"] = None
            zone_timers[zone_name]["last_seen"] = None
    persist(save_presence, "presence", "reset", "Tous les timers de zones réinitialisés", "warn")
    return {"message": "All timers reset"}


@app.post("/api/zones/reset/{zone_name}")
async def reset_zone_timer(zone_name: str):
    with data_lock:
        if zone_name in zone_timers:
            zone_timers[zone_name]["total_time"] = 0
            zone_timers[zone_name]["occupy_start"] = None
            zone_timers[zone_name]["last_seen"] = None
    persist(save_presence, "presence", "reset", f"Timer de « {zone_name} » réinitialisé", "info",
            meta={"zone": zone_name})
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


@app.get("/api/detections/{video_name:path}")
async def get_detections(video_name: str):
    """Return current YOLO detection bounding boxes for a running stream"""
    with streams_lock:
        info = active_streams.get(video_name)
        if not info or not info["active"]:
            return {"detections": [], "active": False}
        dets = info.get("detections", [])
        return {
            "detections": [
                {
                    "x1": d["x1"], "y1": d["y1"],
                    "x2": d["x2"], "y2": d["y2"],
                    "conf": round(d["conf"], 3),
                    "track_id": d.get("track_id"),
                }
                for d in dets
            ],
            "active": True,
        }


@app.post("/api/stream/{video_name:path}/stop")
async def stop_video_stream(video_name: str):
    """Stop a specific video or camera stream"""
    with streams_lock:
        if video_name in active_streams:
            active_streams[video_name]["active"] = False
    audit_event("stream", "stopped", f"Stream arrêté : {video_name}", "info", {"source": video_name})
    return {"message": f"Stream stopped for {video_name}"}


@app.post("/api/streams/stop")
async def stop_all_streams():
    """Stop all active streams"""
    with streams_lock:
        stopped = list(active_streams.keys())
        for video_name in active_streams:
            active_streams[video_name]["active"] = False
    audit_event("stream", "stopped_all", f"Tous les streams arrêtés ({len(stopped)})", "warn",
                {"count": len(stopped)})
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
        state = blur_enabled
    audit_event("blur", "toggled", f"Floutage {'activé' if state else 'désactivé'}", "info",
                {"enabled": state})
    return {"enabled": state}


@app.post("/api/blur/{state}")
async def set_blur(state: str):
    """Set blur state (on/off)"""
    global blur_enabled
    with blur_lock:
        blur_enabled = state.lower() in ("on", "true", "1", "enabled")
        return {"enabled": blur_enabled}


# ==================== Counting API Endpoints ====================

class CountingConfig(BaseModel):
    zone_name: str
    mode: str = "simple"  # "simple" or "complex"


@app.get("/api/counting/{video_name:path}")
async def get_counting(video_name: str):
    """Get counting config and state for a video"""
    config = counting_config.get(video_name)
    with counting_lock:
        complex_state = counting_state.get(video_name, {})
        simple_state = simple_counting_state.get(video_name, {})

    zone_name = config.get("zone_name") if config else None
    zs = get_zone_settings(video_name) if config else None
    mode = zs.get("mode", "complex") if zs else "simple"
    flip_count = zs.get("flip_count", 0) if zs else 0

    # Compute effective angle for display
    effective_angle = None
    if zone_name:
        video_zones = zones_by_video.get(video_name, {})
        zone_data = video_zones.get(zone_name)
        if zone_data and zone_data.get("polygons"):
            all_points = [p for poly in zone_data["polygons"] for p in poly]
            if len(all_points) >= 3:
                base_angle = compute_polygon_direction(all_points)
                effective_angle = (base_angle + flip_count * 90) % 360

    # Return all zone_settings so frontend knows each zone's mode/flip
    all_zone_settings = config.get("zone_settings", {}) if config else {}

    return {
        "configured": config is not None and zone_name is not None,
        "zone_name": zone_name,
        "flip_count": flip_count,
        "mode": mode,
        "angle": effective_angle,
        "zone_settings": all_zone_settings,
        # Active counting state (based on mode)
        "enabled": complex_state.get("enabled", False) if mode == "complex" else simple_state.get("enabled", False),
        "count": complex_state.get("count", 0) if mode == "complex" else simple_state.get("count", 0),
    }


@app.post("/api/counting/{video_name:path}/config")
async def set_counting_config(video_name: str, cfg: CountingConfig):
    """Configure counting for a video: set active zone + mode"""
    video_zones = zones_by_video.get(video_name, {})
    if cfg.zone_name not in video_zones:
        raise HTTPException(status_code=404, detail=f"Zone '{cfg.zone_name}' not found for this video")

    if video_name not in counting_config:
        counting_config[video_name] = {"zone_name": cfg.zone_name, "zone_settings": {}}

    config = counting_config[video_name]
    config["zone_name"] = cfg.zone_name

    # Create zone_settings entry if it doesn't exist
    if cfg.zone_name not in config.get("zone_settings", {}):
        config.setdefault("zone_settings", {})[cfg.zone_name] = {"mode": cfg.mode, "flip_count": 0}
    else:
        # Update mode if provided
        config["zone_settings"][cfg.zone_name]["mode"] = cfg.mode

    save_counting_config()
    zs = config["zone_settings"][cfg.zone_name]
    return {"message": "Counting configured", "zone_name": cfg.zone_name, "mode": zs["mode"], "flip_count": zs["flip_count"]}


@app.post("/api/counting/{video_name:path}/toggle")
async def toggle_counting(video_name: str):
    """Toggle counting on/off based on active zone's mode."""
    config = counting_config.get(video_name)
    if not config or not config.get("zone_name"):
        raise HTTPException(status_code=400, detail="Counting not configured for this video.")

    mode = get_active_zone_mode(video_name)

    zone_name = config["zone_name"]

    if mode == "complex":
        with counting_lock:
            state = counting_state.get(video_name)
            if state and state.get("enabled"):
                # Disable: keep MOG2 model (it stays frozen for reuse)
                state["enabled"] = False
                state["tracked_objects"] = {}
                state["debug_objects"] = {}
                state["debug_contours"] = []
                state["recent_crossings"] = []
                if video_name in counting_debug_masks:
                    del counting_debug_masks[video_name]
            else:
                # Enable: reuse existing MOG2 for this zone if already learned
                counting_state[video_name] = {
                    "enabled": True, "count": 0,
                    "tracked_objects": {}, "debug_objects": {},
                    "debug_contours": [], "recent_crossings": [],
                }
                video_models = counting_bg_models.setdefault(video_name, {})
                if zone_name not in video_models:
                    # First time: create new MOG2 (will learn for N frames then freeze)
                    mog2 = cv2.createBackgroundSubtractorMOG2(
                        history=counting_params["mog2_history"],
                        varThreshold=counting_params["mog2_var_threshold"],
                        detectShadows=counting_params["mog2_detect_shadows"],
                    )
                    video_models[zone_name] = {"mog2": mog2, "frame_count": 0}
                    print(f"[COUNTING-BLOB] Created new MOG2 for {video_name} zone={zone_name}")
                else:
                    fc = video_models[zone_name].get("frame_count", 0)
                    print(f"[COUNTING-BLOB] Reusing frozen MOG2 for {video_name} zone={zone_name} (frame_count={fc})")
        with counting_lock:
            enabled = counting_state.get(video_name, {}).get("enabled", False)
            count = counting_state.get(video_name, {}).get("count", 0)
    else:
        with counting_lock:
            state = simple_counting_state.get(video_name)
            if state and state.get("enabled"):
                state["enabled"] = False
                state["prev_line_pixels"] = None
            else:
                simple_counting_state[video_name] = {
                    "enabled": True, "count": 0,
                    "prev_line_pixels": None, "cooldown_remaining": 0, "current_diff": 0.0,
                }
        with counting_lock:
            s = simple_counting_state.get(video_name, {})
            enabled = s.get("enabled", False)
            count = s.get("count", 0)

    return {"enabled": enabled, "count": count, "mode": mode}


@app.post("/api/counting/{video_name:path}/flip")
async def flip_counting_direction(video_name: str):
    """Rotate counting direction by 90 degrees for the active zone"""
    config = counting_config.get(video_name)
    if not config or not config.get("zone_name"):
        raise HTTPException(status_code=400, detail="Counting not configured")

    zone_name = config["zone_name"]
    zs = config.get("zone_settings", {}).get(zone_name, {"mode": "simple", "flip_count": 0})
    zs["flip_count"] = (zs.get("flip_count", 0) + 1) % 4
    config.setdefault("zone_settings", {})[zone_name] = zs
    save_counting_config()

    # Reset tracked objects / baseline since the line moved
    with counting_lock:
        state = counting_state.get(video_name)
        if state:
            state["tracked_objects"] = {}
            state["debug_objects"] = {}
            state["debug_contours"] = []
        simple_st = simple_counting_state.get(video_name)
        if simple_st:
            simple_st["prev_line_pixels"] = None

    return {"flip_count": zs["flip_count"]}


@app.post("/api/counting/{video_name:path}/reset")
async def reset_counting(video_name: str):
    """Reset the counter based on active zone's mode"""
    config = counting_config.get(video_name)
    zone_name = config.get("zone_name") if config else None
    mode = get_active_zone_mode(video_name)

    if mode == "complex":
        with counting_lock:
            state = counting_state.get(video_name)
            if state:
                state["count"] = 0
                state["tracked_objects"] = {}
                state["debug_objects"] = {}
                state["debug_contours"] = []
        # Destroy MOG2 for this zone so it re-learns from scratch on next enable
        if zone_name and video_name in counting_bg_models:
            if zone_name in counting_bg_models[video_name]:
                del counting_bg_models[video_name][zone_name]
                print(f"[COUNTING-BLOB] Destroyed MOG2 for {video_name} zone={zone_name} (will re-learn on next enable)")
            # Re-create immediately if counting is still enabled
            cs = counting_state.get(video_name, {})
            if cs.get("enabled"):
                mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=counting_params["mog2_history"],
                    varThreshold=counting_params["mog2_var_threshold"],
                    detectShadows=counting_params["mog2_detect_shadows"],
                )
                counting_bg_models.setdefault(video_name, {})[zone_name] = {"mog2": mog2, "frame_count": 0}
                print(f"[COUNTING-BLOB] Re-created MOG2 for {video_name} zone={zone_name}")
    else:
        with counting_lock:
            state = simple_counting_state.get(video_name)
            if state:
                state["count"] = 0
                state["prev_line_pixels"] = None
                state["cooldown_remaining"] = 0

    return {"count": 0}


@app.get("/api/counting/params")
async def get_counting_params():
    """Get current counting algorithm parameters"""
    return counting_params


@app.put("/api/counting/params")
async def update_counting_params(params: dict):
    """Update counting algorithm parameters and save to file"""
    for key, value in params.items():
        if key in COUNTING_PARAMS_DEFAULTS:
            counting_params[key] = value
    with open(COUNTING_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(counting_params, f, indent=2, ensure_ascii=False)
    return counting_params


# ==================== Lieu API Endpoints ====================

class LieuCreate(BaseModel):
    lieu_id: str
    name: str
    address: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


class LieuUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


@app.get("/api/lieux")
async def list_lieux():
    return {"lieux": lieux}


@app.get("/api/lieux/{lieu_id}")
async def get_lieu(lieu_id: str):
    if lieu_id not in lieux:
        raise HTTPException(status_code=404, detail="Lieu not found")
    lieu = lieux[lieu_id]
    lieu_sites = {sid: s for sid, s in sites.items() if s.get("lieu_id") == lieu_id}
    return {"lieu": lieu, "sites": lieu_sites}


@app.post("/api/lieux")
async def create_lieu(lieu: LieuCreate):
    if lieu.lieu_id in lieux:
        raise HTTPException(status_code=400, detail="Lieu ID already exists")
    lieux[lieu.lieu_id] = {
        "name": lieu.name,
        "address": lieu.address or "",
        "description": lieu.description or "",
        "icon": lieu.icon or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    persist(save_lieux, "lieu", "created", f"Lieu « {lieu.name} » créé", "success",
                {"lieu_id": lieu.lieu_id, "name": lieu.name})
    return {"message": "Lieu created", "lieu_id": lieu.lieu_id}


@app.put("/api/lieux/{lieu_id}")
async def update_lieu(lieu_id: str, payload: LieuUpdate):
    if lieu_id not in lieux:
        raise HTTPException(status_code=404, detail="Lieu not found")
    for field in ("name", "address", "description", "icon"):
        val = getattr(payload, field, None)
        if val is not None:
            lieux[lieu_id][field] = val
    persist(save_lieux, "lieu", "updated", f"Lieu « {lieux[lieu_id]['name']} » modifié", "info",
            meta={"lieu_id": lieu_id})
    return {"message": "Lieu updated"}


@app.delete("/api/lieux/{lieu_id}")
async def delete_lieu(lieu_id: str):
    if lieu_id not in lieux:
        raise HTTPException(status_code=404, detail="Lieu not found")
    child_sites = [sid for sid, s in sites.items() if s.get("lieu_id") == lieu_id]
    if child_sites:
        raise HTTPException(status_code=400,
                            detail=f"Cannot delete: {len(child_sites)} site(s) still attached")
    lieu_name = lieux[lieu_id].get("name", lieu_id)
    del lieux[lieu_id]
    persist(save_lieux, "lieu", "deleted", f"Lieu « {lieu_name} » supprimé", "warn",
            meta={"lieu_id": lieu_id})
    return {"message": "Lieu deleted"}


# ==================== Site API Endpoints ====================

class SiteCreate(BaseModel):
    site_id: str
    name: str
    lieu_id: str
    address: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


class SiteUpdate(BaseModel):
    name: Optional[str] = None
    lieu_id: Optional[str] = None
    address: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


@app.get("/api/sites")
async def list_sites():
    return {"sites": sites}


@app.get("/api/sites/{site_id}")
async def get_site(site_id: str):
    if site_id not in sites:
        raise HTTPException(status_code=404, detail="Site not found")
    site = sites[site_id]
    site_cameras = {cid: c for cid, c in cameras.items() if c.get("site_id") == site_id}
    return {"site": site, "cameras": site_cameras}


@app.post("/api/sites")
async def create_site(site: SiteCreate):
    if site.site_id in sites:
        raise HTTPException(status_code=400, detail="Site ID already exists")
    if site.lieu_id not in lieux:
        raise HTTPException(status_code=400, detail=f"Lieu '{site.lieu_id}' not found")
    sites[site.site_id] = {
        "name": site.name,
        "lieu_id": site.lieu_id,
        "address": site.address or "",
        "description": site.description or "",
        "icon": site.icon or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    persist(save_sites, "site", "created", f"Site « {site.name} » créé dans lieu « {site.lieu_id} »", "success",
            meta={"site_id": site.site_id, "name": site.name, "lieu_id": site.lieu_id})
    return {"message": "Site created", "site_id": site.site_id}


@app.put("/api/sites/{site_id}")
async def update_site(site_id: str, payload: SiteUpdate):
    if site_id not in sites:
        raise HTTPException(status_code=404, detail="Site not found")
    if payload.lieu_id is not None and payload.lieu_id not in lieux:
        raise HTTPException(status_code=400, detail=f"Lieu '{payload.lieu_id}' not found")
    for field in ("name", "lieu_id", "address", "description", "icon"):
        val = getattr(payload, field, None)
        if val is not None:
            sites[site_id][field] = val
    persist(save_sites, "site", "updated", f"Site « {sites[site_id]['name']} » modifié", "info",
            meta={"site_id": site_id})
    return {"message": "Site updated"}


@app.delete("/api/sites/{site_id}")
async def delete_site(site_id: str):
    if site_id not in sites:
        raise HTTPException(status_code=404, detail="Site not found")
    child_cams = [cid for cid, c in cameras.items() if c.get("site_id") == site_id]
    if child_cams:
        raise HTTPException(status_code=400,
                            detail=f"Cannot delete: {len(child_cams)} camera(s) still attached")
    site_name = sites[site_id].get("name", site_id)
    del sites[site_id]
    persist(save_sites, "site", "deleted", f"Site « {site_name} » supprimé", "warn",
            meta={"site_id": site_id})
    return {"message": "Site deleted"}


# ==================== Benefit API Endpoints ====================

def _derive_zone_polygon_types(polygons: list) -> list[str]:
    """Fallback: tout en include (on ne peut pas deviner)"""
    return ["include"] * len(polygons)


class BenefitCreate(BaseModel):
    benefit_id: str
    name: str
    skill: str
    skill_item: Optional[str] = None
    categories: list[str] = []
    camera_id: str
    zone_polygons: Optional[list] = None
    zone_polygon_types: Optional[list[str]] = None  # 'include'|'exclude' par polygone
    zone_ref_width: Optional[int] = None
    zone_ref_height: Optional[int] = None
    active: bool = True
    canvas: Optional[dict] = None


class BenefitUpdate(BaseModel):
    name: Optional[str] = None
    skill: Optional[str] = None
    skill_item: Optional[str] = None
    categories: Optional[list[str]] = None
    zone_polygons: Optional[list] = None
    zone_polygon_types: Optional[list[str]] = None
    zone_ref_width: Optional[int] = None
    zone_ref_height: Optional[int] = None
    active: Optional[bool] = None
    canvas: Optional[dict] = None


@app.get("/api/benefits")
async def list_benefits():
    return {"benefits": benefits}


@app.get("/api/benefits/{benefit_id}")
async def get_benefit(benefit_id: str):
    if benefit_id not in benefits:
        raise HTTPException(status_code=404, detail="Benefit not found")
    return {"benefit": benefits[benefit_id]}


@app.get("/api/cameras/{camera_id}/benefits")
async def list_camera_benefits(camera_id: str):
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam_benefits = {bid: b for bid, b in benefits.items() if b.get("camera_id") == camera_id}
    return {"benefits": cam_benefits}


@app.post("/api/benefits")
async def create_benefit(benefit: BenefitCreate):
    if benefit.benefit_id in benefits:
        raise HTTPException(status_code=400, detail="Benefit ID already exists")
    if benefit.camera_id not in cameras:
        raise HTTPException(status_code=400, detail=f"Camera '{benefit.camera_id}' not found")
    valid_skills = {"detection", "counting", "heatmap", "quality"}
    if benefit.skill not in valid_skills:
        raise HTTPException(status_code=400, detail=f"Skill invalide. Valeurs acceptées: {', '.join(sorted(valid_skills))}")
    if not benefit.categories:
        benefit.categories = ["human::silhouette"]
    benefits[benefit.benefit_id] = {
        "name": benefit.name,
        "skill": benefit.skill,
        "skill_item": benefit.skill_item or "",
        "categories": benefit.categories,
        "camera_id": benefit.camera_id,
        "zone_polygons": benefit.zone_polygons or [],
        "zone_polygon_types": benefit.zone_polygon_types or _derive_zone_polygon_types(benefit.zone_polygons or []),
        "zone_ref_height": benefit.zone_ref_height,
        "active": benefit.active,
        "canvas": benefit.canvas or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    persist(save_benefits, "benefit", "created",
           f"Bénéfice « {benefit.name} » créé (skill={benefit.skill}, camera={benefit.camera_id})",
           "success", meta={"benefit_id": benefit.benefit_id, "skill": benefit.skill, "camera_id": benefit.camera_id})
    await sync_benefit_zones()
    return {"message": "Benefit created", "benefit_id": benefit.benefit_id}


@app.put("/api/benefits/{benefit_id}")
async def update_benefit(benefit_id: str, payload: BenefitUpdate):
    if benefit_id not in benefits:
        raise HTTPException(status_code=404, detail="Benefit not found")
    for field in ("name", "skill", "skill_item", "categories", "zone_polygons", "zone_polygon_types", "zone_ref_width", "zone_ref_height", "active", "canvas"):
        val = getattr(payload, field, None)
        if val is not None:
            benefits[benefit_id][field] = val
    persist(save_benefits, "benefit", "updated", f"Bénéfice « {benefits[benefit_id]['name']} » modifié", "info",
            meta={"benefit_id": benefit_id})
    await sync_benefit_zones()
    return {"message": "Benefit updated"}


@app.delete("/api/benefits/{benefit_id}")
async def delete_benefit(benefit_id: str):
    if benefit_id not in benefits:
        raise HTTPException(status_code=404, detail="Benefit not found")
    b_name = benefits[benefit_id].get("name", benefit_id)
    b_camera = benefits[benefit_id].get("camera_id", "")
    del benefits[benefit_id]
    persist(save_benefits, "benefit", "deleted", f"Bénéfice « {b_name} » supprimé", "warn",
            meta={"benefit_id": benefit_id, "camera_id": b_camera})
    return {"message": "Benefit deleted"}


# Hierarchical convenience endpoint
@app.get("/api/hierarchy")
async def get_hierarchy():
    """Full hierarchy: LIEU -> SITE -> CAMERA -> BENEFIT"""
    tree = {}
    for lid, l in lieux.items():
        node = {**l, "lieu_id": lid, "sites": {}}
        for sid, s in sites.items():
            if s.get("lieu_id") == lid:
                s_node = {**s, "site_id": sid, "cameras": {}}
                for cid, c in cameras.items():
                    if c.get("site_id") == sid:
                        c_node = {**c, "camera_id": cid, "benefits": {}}
                        for bid, b in benefits.items():
                            if b.get("camera_id") == cid:
                                c_node["benefits"][bid] = {**b, "benefit_id": bid}
                        s_node["cameras"][cid] = c_node
                node["sites"][sid] = s_node
        tree[lid] = node
    return {"hierarchy": tree}


@app.post("/api/benefits/sync-zones")
async def sync_benefit_zones():
    """Push detection_presence (actifs) + counting (tous) zone_polygons into zones_by_video."""
    synced = 0
    for bid, b in benefits.items():
        polys = b.get("zone_polygons", [])
        if not polys:
            continue
        cam_id = b.get("camera_id", "")
        if not cam_id:
            continue
        video_name = _get_video_path_for_camera(cam_id) or cam_id
        if video_name not in zones_by_video:
            zones_by_video[video_name] = {}

        # Detection présence (actifs uniquement)
        if b.get("skill") == ALLOWED_SKILL and b.get("skill_item") == ALLOWED_SKILL_ITEM:
            if not b.get("active", True):
                continue
            zones_by_video[video_name][bid] = {"polygons": polys}
            synced += 1
        # Comptage: tous (actifs ou non)
        elif b.get("skill") == "counting":
            zones_by_video[video_name][bid] = {"polygons": polys}
            synced += 1
            types = b.get("zone_polygon_types") or ["include"] * len(polys)
            for idx, (poly, pt) in enumerate(zip(polys, types)):
                if pt == "include" and len(poly) >= 3:
                    zone_key = f"{bid}:{idx}"
                    zones_by_video[video_name][zone_key] = {"polygons": [poly]}
                    synced += 1
    if synced:
        save_zones()
    return {"synced": synced}


# ==================== Camera API Endpoints ====================

class CameraCreate(BaseModel):
    camera_id: str
    name: str
    type: str  # "webcam" or "rtsp"
    device_id: int | None = None  # For webcam
    url: str | None = None  # For RTSP
    site_id: str | None = None


@app.get("/api/cameras")
async def list_cameras():
    """List all configured cameras"""
    return {"cameras": cameras}


@app.post("/api/cameras")
async def add_camera(camera: CameraCreate):
    """Add a new camera source"""
    if camera.camera_id in cameras:
        raise HTTPException(status_code=400, detail="Camera ID already exists")

    if camera.site_id and camera.site_id not in sites:
        raise HTTPException(status_code=400, detail=f"Site '{camera.site_id}' not found")

    if camera.type == "webcam":
        if camera.device_id is None:
            raise HTTPException(status_code=400, detail="device_id required for webcam")
        cameras[camera.camera_id] = {
            "type": "webcam",
            "name": camera.name,
            "device_id": camera.device_id,
            "site_id": camera.site_id or "",
        }
    elif camera.type == "rtsp":
        if not camera.url:
            raise HTTPException(status_code=400, detail="url required for RTSP")
        cameras[camera.camera_id] = {
            "type": "rtsp",
            "name": camera.name,
            "url": camera.url,
            "site_id": camera.site_id or "",
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid camera type")

    persist(save_cameras, "camera", "created", f"Caméra « {camera.name} » ajoutée ({camera.type})", "success",
            meta={"camera_id": camera.camera_id, "name": camera.name, "type": camera.type})
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

    cam_name = cameras.get(camera_id, {}).get("name", camera_id)
    orphan_benefits = [bid for bid, b in benefits.items() if b.get("camera_id") == camera_id]
    for bid in orphan_benefits:
        del benefits[bid]
    if orphan_benefits:
        save_benefits()
    del cameras[camera_id]
    persist(save_cameras, "camera", "deleted",
            f"Caméra « {cam_name} » supprimée ({len(orphan_benefits)} bénéfice(s) cascade)",
            "warn", meta={"camera_id": camera_id, "benefits_removed": orphan_benefits})
    return {"message": "Camera deleted", "benefits_removed": len(orphan_benefits)}


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
    Uses a SEPARATE model instance per stream to avoid tracker state pollution.
    """
    frame_count = 0

    # Create a separate model instance for this stream's tracking
    # This prevents tracker state from leaking between different video streams
    local_model = YOLO("human.pt")
    print(f"[{video_name}] Created dedicated YOLO tracker instance")

    try:
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

                # Use model.track() with persist=True for this stream only
                # Each stream has its own model instance, so no state pollution
                results = local_model.track(
                    frame,
                    verbose=False,
                    classes=[0],
                    conf=YOLO_CONFIDENCE,
                    device=YOLO_DEVICE,
                    persist=True,
                    tracker="bytetrack.yaml"
                )

                detections = []

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])

                        # Get track ID if available
                        track_id = None
                        if box.id is not None:
                            track_id = int(box.id[0])

                        detections.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": conf,
                            "track_id": track_id
                        })

                with streams_lock:
                    if video_name in active_streams:
                        active_streams[video_name]["detections"] = detections

            except Exception as e:
                print(f"Detection error for {video_name}: {e}")
                continue
    finally:
        # Clean up the local model when stream ends
        del local_model
        print(f"[{video_name}] Released YOLO tracker instance")


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

            # Update counting modules (independent from YOLO)
            update_counting_blob(source_name, frame)
            update_counting_simple(source_name, frame)

            if time.time() - last_save > 2:
                save_presence()
                last_save = time.time()

            elapsed = time.time() - frame_start
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        # Clean up shared frames
        with frames_lock:
            if source_name in shared_frames:
                del shared_frames[source_name]
        # Mark stream as inactive and clean up
        with streams_lock:
            if source_name in active_streams:
                active_streams[source_name]["active"] = False
                active_streams[source_name]["detections"] = []
        # Clean up counting state (complex + simple) when stream stops
        with counting_lock:
            if source_name in counting_state:
                counting_state[source_name]["enabled"] = False
                counting_state[source_name]["tracked_objects"] = {}
                counting_state[source_name]["debug_objects"] = {}
                counting_state[source_name]["debug_contours"] = []
                counting_state[source_name]["recent_crossings"] = []
            if source_name in simple_counting_state:
                simple_counting_state[source_name]["enabled"] = False
                simple_counting_state[source_name]["prev_line_pixels"] = None
        # Keep counting_bg_models[source_name] — MOG2 models persist across stream restarts
        if source_name in counting_debug_masks:
            del counting_debug_masks[source_name]
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


def generate_frames(video_name: str, draw_overlay: bool = True):
    """Generate frames for streaming - reads from shared_frames written by processor.
    draw_overlay=False: raw video only (no detections/zones drawing)."""
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

        if draw_overlay:
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

                        zone_overlay = frame.copy()
                        cv2.fillPoly(zone_overlay, [pts], color)
                        cv2.addWeighted(zone_overlay, 0.3, frame, 0.7, 0, frame)
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

            # Draw counting line, overlay, and counter
            active_mode = get_active_zone_mode(video_name)
            line_info = get_counting_line(video_name)
            if line_info:
                ls = line_info["line_start"]
                le = line_info["line_end"]
                dx = line_info["dir_x"]
                dy = line_info["dir_y"]
                angle_deg = line_info["angle"]
                r_left, r_top, r_right, r_bottom = line_info["roi_bounds"]
                line_color = (0, 200, 255) if active_mode == "complex" else (0, 150, 255)

                # --- Complex mode: foreground mask overlay ---
                if active_mode == "complex":
                    debug_mask = counting_debug_masks.get(video_name)
                    if debug_mask is not None:
                        mask_colored = np.zeros_like(frame)
                        mask_colored[:, :, 1] = debug_mask
                        mask_bool = debug_mask > 0
                        frame[mask_bool] = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)[mask_bool]

                # --- Get counting data based on mode ---
                with counting_lock:
                    if active_mode == "complex":
                        cs = counting_state.get(video_name, {})
                        count_val = cs.get("count", 0)
                        debug_objs = cs.get("debug_objects", {}).copy()
                        debug_contours = list(cs.get("debug_contours", []))
                        recent_cross = list(cs.get("recent_crossings", []))
                    else:
                        ss = simple_counting_state.get(video_name, {})
                        count_val = ss.get("count", 0)
                        debug_objs = {}
                        debug_contours = []
                        recent_cross = []

                # Draw contours in bright cyan
                if debug_contours:
                    cv2.drawContours(frame, debug_contours, -1, (255, 255, 0), 2, cv2.LINE_AA)

                # Draw the counting line
                cv2.line(frame, (int(ls[0]), int(ls[1])), (int(le[0]), int(le[1])), line_color, 2, cv2.LINE_AA)

                # Draw direction arrow at center of the line
                mid_x = (ls[0] + le[0]) / 2
                mid_y = (ls[1] + le[1]) / 2
                arrow_len = 30
                arr_start = (int(mid_x - dx * arrow_len), int(mid_y - dy * arrow_len))
                arr_end = (int(mid_x + dx * arrow_len), int(mid_y + dy * arrow_len))
                cv2.arrowedLine(frame, arr_start, arr_end, line_color, 2, cv2.LINE_AA, tipLength=0.4)

                # === Visual Debug Logs ===
                now = time.time()

                # Draw tracked blob centers + IDs + movement arrows
                for tid, dobj in debug_objs.items():
                    cx_d, cy_d = int(dobj["cx"]), int(dobj["cy"])
                    counted = dobj["counted"]

                    # Color: green if counted, orange if not yet
                    dot_color = (0, 200, 0) if counted else (0, 165, 255)
                    cv2.circle(frame, (cx_d, cy_d), 6, dot_color, -1, cv2.LINE_AA)

                    # Blob ID label + projection value + area for debugging
                    proj_val = dobj.get("proj", 0)
                    thr_val = line_info["threshold"]
                    area_val = dobj.get("area", 0)
                    id_label = f"#{tid}" + (" OK" if counted else f" p={proj_val:.0f}/t={thr_val:.0f}")
                    cv2.putText(frame, id_label, (cx_d + 8, cy_d - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1, cv2.LINE_AA)
                    # Show blob area below
                    cv2.putText(frame, f"a={area_val:.0f}", (cx_d + 8, cy_d + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)

                    # Movement arrow (prev -> current)
                    if "prev_cx" in dobj:
                        pcx, pcy = int(dobj["prev_cx"]), int(dobj["prev_cy"])
                        if abs(pcx - cx_d) > 1 or abs(pcy - cy_d) > 1:
                            cv2.arrowedLine(frame, (pcx, pcy), (cx_d, cy_d),
                                            dot_color, 1, cv2.LINE_AA, tipLength=0.3)

            # Flash recent crossing events (bright circle + label)
            for ev in recent_cross:
                age = now - ev["time"]
                if age < counting_params["crossing_display_time"]:
                    radius = int(20 + 15 * max(0, 1 - age / 0.5))
                    ecx, ecy = int(ev["cx"]), int(ev["cy"])
                    flash_color = (0, 255, 255)
                    cv2.circle(frame, (ecx, ecy), radius, flash_color, 2, cv2.LINE_AA)
                    if age < 1.0:
                        cv2.putText(frame, "COUNTED", (ecx + radius + 4, ecy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, flash_color, 1, cv2.LINE_AA)

            # Log panel: recent crossings list (bottom-left of ROI)
            if recent_cross:
                log_x = int(r_left) + 5
                log_y = int(r_bottom) - 10
                for i, ev in enumerate(reversed(recent_cross[-5:])):
                    age = now - ev["time"]
                    log_label = f"#{ev['track_id']} crossed ({age:.1f}s ago)"
                    log_col = (0, 255, 255) if age < 0.5 else (180, 180, 180)
                    cv2.putText(frame, log_label, (log_x, log_y - i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, log_col, 1, cv2.LINE_AA)

            # Draw count label (top-right of ROI)
            count_label = f"Count: {count_val}"
            (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_x = int(r_right) - tw - 10
            label_y = int(r_top) - 10
            if label_y < th + 5:
                label_y = int(r_top) + th + 10
            cv2.rectangle(frame, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, count_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

            # Angle indicator + mode label (top-left of ROI)
            mode_str = "BLOB/MOG2" if active_mode == "complex" else "GRADIENT"
            with counting_lock:
                is_enabled = False
                if active_mode == "complex":
                    is_enabled = counting_state.get(video_name, {}).get("enabled", False)
                else:
                    is_enabled = simple_counting_state.get(video_name, {}).get("enabled", False)
            status_str = mode_str if is_enabled else "OFF"
            angle_label = f"{angle_deg:.0f} deg | {status_str}"
            cv2.putText(frame, angle_label, (int(r_left) + 5, int(r_top) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Simple mode: show diff debug indicator below angle label
            if active_mode == "simple":
                with counting_lock:
                    simple_diff = simple_counting_state.get(video_name, {}).get("current_diff", 0.0)
                diff_label = f"diff={simple_diff:.1f}"
                cv2.putText(frame, diff_label, (int(r_left) + 5, int(r_top) + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        # Resize frame for streaming to reduce bandwidth (keep aspect ratio)
        h, w = frame.shape[:2]
        max_width = 960  # Reduced from original for bandwidth savings
        if w > max_width:
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Lower JPEG quality for streaming (50 instead of 85) - saves ~50% bandwidth
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
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

    audit_event("stream", "started", f"Stream démarré : {video_name}", "success", {"source": video_name})
    return {"message": "Stream started", "video": video_name}


@app.get("/api/stream/{video_name:path}")
async def video_stream(video_name: str, overlay: bool = True):
    """Get video stream. overlay=True: with detections/zones overlay; overlay=False: raw video only"""
    draw_overlay = overlay
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
        generate_frames(video_name, draw_overlay=draw_overlay),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
    )


def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
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


