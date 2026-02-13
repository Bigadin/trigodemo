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

# ==================== Counting Module State ====================
# Config per video: {video_name: {"active_zone": str, "zones": {zone_name: {"mode": str, "flip_count": int}}}}
counting_config = {}
# Runtime state per video per zone: {video_name: {zone_name: {"enabled": bool, "count": int, "tracked_objects": {...}, ...}}}
counting_state = {}
counting_lock = threading.Lock()
COUNTING_FILE = DATA_DIR / "counting.json"

# Blob-based counting: MOG2 background subtractors per (video, zone)
counting_bg_subtractors = {}  # {(video_name, zone_name): cv2.BackgroundSubtractorMOG2}
counting_debug_masks = {}     # {(video_name, zone_name): ndarray} - foreground mask for overlay

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
    "gradient_baseline_frames": 30,
    "gradient_threshold_high": 15,
    "gradient_threshold_low": 5,
}
counting_params = {}


def load_counting_params():
    global counting_params
    counting_params = COUNTING_PARAMS_DEFAULTS.copy()
    if COUNTING_PARAMS_FILE.exists():
        with open(COUNTING_PARAMS_FILE, "r") as f:
            user = json.load(f)
        counting_params.update(user)
    print(f"[COUNTING] Params loaded: {counting_params}")


load_counting_params()


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


def load_counting_config():
    global counting_config
    if COUNTING_FILE.exists():
        with open(COUNTING_FILE, "r") as f:
            counting_config = json.load(f)
    # Migrate old format (single zone per video) to new per-zone format
    migrated = False
    for video_name, cfg in list(counting_config.items()):
        if "zones" not in cfg:
            zone_name = cfg.get("zone_name", "")
            old_counts = cfg.get("zone_counts", {})
            zones = {}
            if zone_name:
                zones[zone_name] = {"mode": cfg.get("mode", "blob"), "flip_count": cfg.get("flip_count", 0), "count": old_counts.get(zone_name, 0)}
            for zn, fc in cfg.get("zone_flips", {}).items():
                if zn not in zones:
                    zones[zn] = {"mode": cfg.get("mode", "blob"), "flip_count": fc, "count": old_counts.get(zn, 0)}
            counting_config[video_name] = {"active_zone": zone_name, "zones": zones}
            migrated = True
    if migrated:
        print("[COUNTING] Migrated config to per-zone format")


def save_counting_config():
    with open(COUNTING_FILE, "w") as f:
        json.dump(counting_config, f, indent=2)


load_counting_config()


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


def get_counting_line(video_name: str, zone_name: str):
    """Calculate counting line from ROI polygon's principal direction.
    Returns dict with line endpoints, direction vector, threshold, angle, roi_bounds or None.
    """
    config = counting_config.get(video_name)
    if not config or zone_name not in config.get("zones", {}):
        return None

    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        zs = zone_states.get(zone_name)
        if not zs or not zs.get("enabled"):
            return None

    zone_cfg = config["zones"][zone_name]
    flip_count = zone_cfg.get("flip_count", 0)

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


def update_counting_blob(video_name: str, zone_name: str, frame: np.ndarray):
    """Blob-based counting: background subtraction (MOG2) on ROI → contour detection → proximity tracking → line crossing.
    Independent from YOLO — works on raw pixels only."""
    global _counting_log_counter, _counting_next_blob_id

    line_info = get_counting_line(video_name, zone_name)
    if not line_info:
        return

    # Check if MOG2 instance exists for this (video, zone)
    mog2_key = (video_name, zone_name)
    if mog2_key not in counting_bg_subtractors:
        return

    dir_x = line_info["dir_x"]
    dir_y = line_info["dir_y"]
    threshold = line_info["threshold"]
    roi_left, roi_top, roi_right, roi_bottom = line_info["roi_bounds"]
    now = time.time()

    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        state = zone_states.get(zone_name)
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
    mog2 = counting_bg_subtractors[mog2_key]

    # Track frame count: learn background for N frames, then freeze (learningRate=0)
    with counting_lock:
        frame_count = state.get("bg_frame_count", 0)
        state["bg_frame_count"] = frame_count + 1

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
    counting_debug_masks[mog2_key] = debug_mask

    # --- Step 6: Proximity-based blob tracking + line crossing ---
    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        state = zone_states.get(zone_name)
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


# ==================== Gradient 1D Counting (Simple mode) ====================
# State: counting_state[video]["gradient_state"] = "idle" | "triggered"
# Debug: counting_state[video]["gradient_profile"] = 1D array for overlay

def update_counting_gradient(video_name: str, zone_name: str, frame: np.ndarray):
    """Simple counting: project ROI to 1D intensity profile along direction axis.
    Uses hysteresis (front descendant): baseline → deviation triggers → return to baseline = +1."""
    global _counting_log_counter

    line_info = get_counting_line(video_name, zone_name)
    if not line_info:
        return

    dir_x = line_info["dir_x"]
    dir_y = line_info["dir_y"]
    roi_left, roi_top, roi_right, roi_bottom = line_info["roi_bounds"]

    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        state = zone_states.get(zone_name)
        if not state or not state.get("enabled"):
            return

    # --- Step 1: Crop frame to ROI bounding box ---
    r_l, r_t = max(0, int(roi_left)), max(0, int(roi_top))
    r_r, r_b = min(frame.shape[1], int(roi_right)), min(frame.shape[0], int(roi_bottom))
    if r_r <= r_l or r_b <= r_t:
        return

    crop = frame[r_t:r_b, r_l:r_r]

    # --- Step 2: Apply polygon mask ---
    video_zones = zones_by_video.get(video_name, {})
    zone_data = video_zones.get(zone_name)
    if not zone_data or not zone_data.get("polygons"):
        return

    crop_h, crop_w = crop.shape[:2]
    poly_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for poly_pts in zone_data["polygons"]:
        if len(poly_pts) >= 3:
            shifted = np.array([[p[0] - r_l, p[1] - r_t] for p in poly_pts], dtype=np.int32)
            cv2.fillPoly(poly_mask, [shifted], 255)

    # --- Step 3: Convert to grayscale, mask, compute 1D profile ---
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=poly_mask)

    # Project each pixel onto direction axis → bin by position → average intensity per bin
    # Create coordinate grids (in crop space)
    ys, xs = np.mgrid[0:crop_h, 0:crop_w]
    # Global coordinates
    xs_g = xs + r_l
    ys_g = ys + r_t

    # Projection onto direction axis
    proj = xs_g.astype(np.float64) * dir_x + ys_g.astype(np.float64) * dir_y

    # Only consider pixels inside polygon mask
    mask_bool = poly_mask > 0
    proj_vals = proj[mask_bool]
    intensity_vals = gray_masked[mask_bool].astype(np.float64)

    if len(proj_vals) == 0:
        return

    # Bin into N bins along the direction axis
    n_bins = 50
    proj_min, proj_max = proj_vals.min(), proj_vals.max()
    if proj_max - proj_min < 1:
        return

    bin_indices = np.clip(
        ((proj_vals - proj_min) / (proj_max - proj_min) * (n_bins - 1)).astype(int),
        0, n_bins - 1
    )
    profile = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(profile, bin_indices, intensity_vals)
    np.add.at(counts, bin_indices, 1)
    counts[counts == 0] = 1
    profile /= counts  # Average intensity per bin

    # --- Step 4: Compute mean intensity (scalar signal) ---
    mean_intensity = float(profile.mean())

    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        state = zone_states.get(zone_name)
        if not state or not state.get("enabled"):
            return

        frame_count = state.get("gradient_frame_count", 0)
        state["gradient_frame_count"] = frame_count + 1
        learning_frames = counting_params["gradient_baseline_frames"]

        # --- Step 5: Baseline learning phase ---
        if frame_count < learning_frames:
            # Accumulate baseline
            baseline_acc = state.get("gradient_baseline_acc", 0.0)
            baseline_profile_acc = state.get("gradient_baseline_profile_acc", np.zeros(n_bins))
            state["gradient_baseline_acc"] = baseline_acc + mean_intensity
            state["gradient_baseline_profile_acc"] = baseline_profile_acc + profile
            if frame_count == learning_frames - 1:
                state["gradient_baseline"] = state["gradient_baseline_acc"] / learning_frames
                state["gradient_baseline_profile"] = state["gradient_baseline_profile_acc"] / learning_frames
                print(f"[COUNTING-GRADIENT] Baseline frozen: {state['gradient_baseline']:.1f} for {video_name}")
            # Store profile for debug even during learning
            state["gradient_profile"] = profile.tolist()
            state["gradient_mean"] = mean_intensity
            return

        baseline = state.get("gradient_baseline")
        if baseline is None:
            return

        # --- Step 6: Hysteresis detection (front descendant) ---
        deviation = abs(mean_intensity - baseline)
        thr_high = counting_params["gradient_threshold_high"]
        thr_low = counting_params["gradient_threshold_low"]
        grad_state = state.get("gradient_state", "idle")

        if grad_state == "idle":
            if deviation > thr_high:
                state["gradient_state"] = "triggered"
        elif grad_state == "triggered":
            if deviation < thr_low:
                # Object has fully passed → count +1
                state["count"] += 1
                state["gradient_state"] = "idle"
                now = time.time()
                recent = state.get("recent_crossings", [])
                recent.append({"track_id": f"g{state['count']}", "cx": (r_l + r_r) / 2, "cy": (r_t + r_b) / 2, "time": now})
                state["recent_crossings"] = recent
                print(f"[COUNTING-GRADIENT] COUNT={state['count']} deviation returned to {deviation:.1f} < {thr_low}")

        # Store debug data
        state["gradient_profile"] = profile.tolist()
        state["gradient_baseline_profile_data"] = state.get("gradient_baseline_profile", np.zeros(n_bins)).tolist() if isinstance(state.get("gradient_baseline_profile"), np.ndarray) else state.get("gradient_baseline_profile", [])
        state["gradient_mean"] = mean_intensity
        state["gradient_deviation"] = deviation
        state["gradient_state_label"] = grad_state

        # Periodic diagnostic log
        _counting_log_counter += 1
        if _counting_log_counter % 60 == 0:
            print(f"[COUNTING-GRADIENT] mean={mean_intensity:.1f} baseline={baseline:.1f} dev={deviation:.1f} state={grad_state} count={state['count']}")


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


# ==================== Counting API Endpoints ====================

class CountingConfig(BaseModel):
    zone_name: str
    mode: str = "blob"  # "blob" (MOG2 advanced) or "gradient" (simple 1D)


@app.get("/api/counting/{video_name:path}")
async def get_counting(video_name: str):
    """Get counting config and state for a video (all zones)"""
    config = counting_config.get(video_name)
    if not config:
        return {"configured": False, "active_zone": None, "zones": {}}

    active_zone = config.get("active_zone", "")
    zones_cfg = config.get("zones", {})
    video_zones = zones_by_video.get(video_name, {})

    # Build per-zone response
    zones_resp = {}
    with counting_lock:
        zone_states = counting_state.get(video_name, {})
        for zn, zcfg in zones_cfg.items():
            zs = zone_states.get(zn, {})
            flip_count = zcfg.get("flip_count", 0)
            # Compute angle
            angle = None
            zone_data = video_zones.get(zn)
            if zone_data and zone_data.get("polygons"):
                all_points = [p for poly in zone_data["polygons"] for p in poly]
                if len(all_points) >= 3:
                    base_angle = compute_polygon_direction(all_points)
                    angle = (base_angle + flip_count * 90) % 360
            # Use runtime count if available, otherwise fall back to persisted count
            count = zs.get("count", zcfg.get("count", 0))
            zones_resp[zn] = {
                "mode": zcfg.get("mode", "blob"),
                "flip_count": flip_count,
                "angle": angle,
                "enabled": zs.get("enabled", False),
                "count": count,
            }

    # Also return active zone's info at top level for backward compat
    active_info = zones_resp.get(active_zone, {})
    return {
        "configured": len(zones_cfg) > 0,
        "active_zone": active_zone,
        "zone_name": active_zone,  # backward compat
        "flip_count": active_info.get("flip_count", 0),
        "mode": active_info.get("mode", "blob"),
        "angle": active_info.get("angle"),
        "enabled": active_info.get("enabled", False),
        "count": active_info.get("count", 0),
        "zones": zones_resp,
    }


@app.post("/api/counting/{video_name:path}/config")
async def set_counting_config(video_name: str, cfg: CountingConfig):
    """Configure counting for a video: add/update a zone and set as active"""
    video_zones = zones_by_video.get(video_name, {})
    if cfg.zone_name not in video_zones:
        raise HTTPException(status_code=404, detail=f"Zone '{cfg.zone_name}' not found for this video")

    existing = counting_config.get(video_name, {"active_zone": "", "zones": {}})
    zones = existing.get("zones", {})

    # Add or update zone config (preserve flip_count if already exists)
    if cfg.zone_name not in zones:
        zones[cfg.zone_name] = {"mode": cfg.mode, "flip_count": 0}
    else:
        zones[cfg.zone_name]["mode"] = cfg.mode

    counting_config[video_name] = {
        "active_zone": cfg.zone_name,
        "zones": zones,
    }
    save_counting_config()
    return {"message": "Counting configured", "zone_name": cfg.zone_name}


@app.post("/api/counting/{video_name:path}/toggle")
async def toggle_counting(video_name: str):
    """Toggle counting on/off for the active zone."""
    config = counting_config.get(video_name)
    if not config or not config.get("active_zone"):
        raise HTTPException(status_code=400, detail="Counting not configured for this video. Set config first.")

    zone_name = config["active_zone"]
    zone_cfg = config.get("zones", {}).get(zone_name, {})
    mode = zone_cfg.get("mode", "blob")
    mog2_key = (video_name, zone_name)

    with counting_lock:
        if video_name not in counting_state:
            counting_state[video_name] = {}
        zone_states = counting_state[video_name]
        zs = zone_states.get(zone_name)

        if zs and zs.get("enabled"):
            # Disable this zone — save count to config for persistence
            zone_cfg["count"] = zs.get("count", 0)
            zs["enabled"] = False
            zs["tracked_objects"] = {}
            zs["debug_objects"] = {}
            zs["debug_contours"] = []
            zs["recent_crossings"] = []
            if mog2_key in counting_bg_subtractors:
                del counting_bg_subtractors[mog2_key]
            if mog2_key in counting_debug_masks:
                del counting_debug_masks[mog2_key]
            print(f"[COUNTING] Disabled {zone_name} for {video_name} (count={zone_cfg['count']})")
        else:
            # Enable this zone — restore count from runtime state or persisted config
            prev_count = zs.get("count", 0) if zs else zone_cfg.get("count", 0)
            zone_states[zone_name] = {
                "enabled": True,
                "count": prev_count,
                "tracked_objects": {},
                "debug_objects": {},
                "debug_contours": [],
                "recent_crossings": [],
            }
            if mode == "blob":
                zone_states[zone_name]["bg_frame_count"] = 0
                mog2 = cv2.createBackgroundSubtractorMOG2(
                    history=counting_params["mog2_history"],
                    varThreshold=counting_params["mog2_var_threshold"],
                    detectShadows=counting_params["mog2_detect_shadows"],
                )
                counting_bg_subtractors[mog2_key] = mog2
                print(f"[COUNTING-BLOB] Created MOG2 for {video_name}/{zone_name} (count={prev_count})")
            else:
                zone_states[zone_name]["gradient_frame_count"] = 0
                zone_states[zone_name]["gradient_baseline_acc"] = 0.0
                zone_states[zone_name]["gradient_state"] = "idle"
                print(f"[COUNTING-GRADIENT] Enabled {video_name}/{zone_name} (count={prev_count})")

    save_counting_config()
    with counting_lock:
        zs = counting_state.get(video_name, {}).get(zone_name, {})
    return {"enabled": zs.get("enabled", False), "count": zs.get("count", 0)}


@app.post("/api/counting/{video_name:path}/flip")
async def flip_counting_direction(video_name: str):
    """Rotate counting direction by 90 degrees for the active zone"""
    config = counting_config.get(video_name)
    if not config or not config.get("active_zone"):
        raise HTTPException(status_code=400, detail="Counting not configured")

    zone_name = config["active_zone"]
    zone_cfg = config.get("zones", {}).get(zone_name, {})
    zone_cfg["flip_count"] = (zone_cfg.get("flip_count", 0) + 1) % 4
    config["zones"][zone_name] = zone_cfg
    save_counting_config()

    # Reset tracked objects for this zone since the line moved
    with counting_lock:
        zs = counting_state.get(video_name, {}).get(zone_name)
        if zs:
            zs["tracked_objects"] = {}
            zs["debug_objects"] = {}
            zs["debug_contours"] = []

    return {"flip_count": zone_cfg["flip_count"]}


@app.post("/api/counting/{video_name:path}/reset")
async def reset_counting(video_name: str):
    """Reset the counter to 0 for the active zone and reinitialize background model"""
    config = counting_config.get(video_name, {})
    zone_name = config.get("active_zone", "")
    zone_cfg = config.get("zones", {}).get(zone_name, {})
    mode = zone_cfg.get("mode", "blob")
    mog2_key = (video_name, zone_name)

    # Reset persisted count
    zone_cfg["count"] = 0
    save_counting_config()

    with counting_lock:
        zs = counting_state.get(video_name, {}).get(zone_name)
        if zs:
            zs["count"] = 0
            zs["tracked_objects"] = {}
            zs["debug_objects"] = {}
            zs["debug_contours"] = []
            zs["recent_crossings"] = []
            if mode == "blob":
                zs["bg_frame_count"] = 0
            else:
                zs["gradient_frame_count"] = 0
                zs["gradient_baseline_acc"] = 0.0
                zs["gradient_state"] = "idle"
                zs.pop("gradient_baseline", None)
                zs.pop("gradient_baseline_profile", None)
    # Reinitialize MOG2 (blob mode only)
    if mode == "blob" and mog2_key in counting_bg_subtractors:
        counting_bg_subtractors[mog2_key] = cv2.createBackgroundSubtractorMOG2(
            history=counting_params["mog2_history"],
            varThreshold=counting_params["mog2_var_threshold"],
            detectShadows=counting_params["mog2_detect_shadows"],
        )
        print(f"[COUNTING-BLOB] Reset MOG2 for {video_name}/{zone_name}")
    elif mode == "gradient":
        print(f"[COUNTING-GRADIENT] Reset for {video_name}/{zone_name}")
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
    with open(COUNTING_PARAMS_FILE, "w") as f:
        json.dump(counting_params, f, indent=2)
    return counting_params


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

            # Update counting module (independent from YOLO) — process ALL enabled zones
            c_cfg = counting_config.get(source_name, {})
            for c_zone_name, c_zone_cfg in c_cfg.get("zones", {}).items():
                with counting_lock:
                    zs = counting_state.get(source_name, {}).get(c_zone_name)
                if zs and zs.get("enabled"):
                    if c_zone_cfg.get("mode", "blob") == "gradient":
                        update_counting_gradient(source_name, c_zone_name, frame)
                    else:
                        update_counting_blob(source_name, c_zone_name, frame)

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
        # Clean up counting state and MOG2 when stream stops — save counts and disable all zones
        cfg = counting_config.get(source_name, {})
        with counting_lock:
            zone_states = counting_state.get(source_name, {})
            for zn, zs in zone_states.items():
                # Persist count to config
                zcfg = cfg.get("zones", {}).get(zn)
                if zcfg is not None:
                    zcfg["count"] = zs.get("count", 0)
                zs["enabled"] = False
                zs["tracked_objects"] = {}
                zs["debug_objects"] = {}
                zs["debug_contours"] = []
                zs["recent_crossings"] = []
        # Clean up MOG2 instances and debug masks for this video
        keys_to_del = [k for k in counting_bg_subtractors if k[0] == source_name]
        for k in keys_to_del:
            del counting_bg_subtractors[k]
        keys_to_del = [k for k in counting_debug_masks if k[0] == source_name]
        for k in keys_to_del:
            del counting_debug_masks[k]
        save_counting_config()
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

        # Draw counting overlays for ALL enabled zones
        c_cfg = counting_config.get(video_name, {})
        for c_zn, c_zcfg in c_cfg.get("zones", {}).items():
            line_info = get_counting_line(video_name, c_zn)
            if not line_info:
                continue

            ls = line_info["line_start"]
            le = line_info["line_end"]
            c_dx = line_info["dir_x"]
            c_dy = line_info["dir_y"]
            angle_deg = line_info["angle"]
            r_left, r_top, r_right, r_bottom = line_info["roi_bounds"]
            line_color = (0, 200, 255)
            c_mode = c_zcfg.get("mode", "blob")

            with counting_lock:
                cs = counting_state.get(video_name, {}).get(c_zn, {})
                count_val = cs.get("count", 0)
                debug_objs = cs.get("debug_objects", {}).copy()
                debug_contours = list(cs.get("debug_contours", []))
                recent_cross = list(cs.get("recent_crossings", []))

            if c_mode == "blob":
                # === BLOB MODE overlay ===
                debug_mask = counting_debug_masks.get((video_name, c_zn))
                if debug_mask is not None:
                    mask_colored = np.zeros_like(frame)
                    mask_colored[:, :, 1] = debug_mask
                    mask_bool = debug_mask > 0
                    frame[mask_bool] = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)[mask_bool]

                if debug_contours:
                    cv2.drawContours(frame, debug_contours, -1, (255, 255, 0), 2, cv2.LINE_AA)

                for tid, dobj in debug_objs.items():
                    cx_d, cy_d = int(dobj["cx"]), int(dobj["cy"])
                    counted = dobj["counted"]
                    dot_color = (0, 200, 0) if counted else (0, 165, 255)
                    cv2.circle(frame, (cx_d, cy_d), 6, dot_color, -1, cv2.LINE_AA)
                    proj_val = dobj.get("proj", 0)
                    thr_val = line_info["threshold"]
                    area_val = dobj.get("area", 0)
                    id_label = f"#{tid}" + (" OK" if counted else f" p={proj_val:.0f}/t={thr_val:.0f}")
                    cv2.putText(frame, id_label, (cx_d + 8, cy_d - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"a={area_val:.0f}", (cx_d + 8, cy_d + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
                    if "prev_cx" in dobj:
                        pcx, pcy = int(dobj["prev_cx"]), int(dobj["prev_cy"])
                        if abs(pcx - cx_d) > 1 or abs(pcy - cy_d) > 1:
                            cv2.arrowedLine(frame, (pcx, pcy), (cx_d, cy_d),
                                            dot_color, 1, cv2.LINE_AA, tipLength=0.3)

            # === Common overlays (both modes) ===
            cv2.line(frame, (int(ls[0]), int(ls[1])), (int(le[0]), int(le[1])), line_color, 2, cv2.LINE_AA)

            mid_x = (ls[0] + le[0]) / 2
            mid_y = (ls[1] + le[1]) / 2
            arrow_len = 30
            arr_start = (int(mid_x - c_dx * arrow_len), int(mid_y - c_dy * arrow_len))
            arr_end = (int(mid_x + c_dx * arrow_len), int(mid_y + c_dy * arrow_len))
            cv2.arrowedLine(frame, arr_start, arr_end, line_color, 2, cv2.LINE_AA, tipLength=0.4)

            now = time.time()

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

            if recent_cross:
                log_x = int(r_left) + 5
                log_y = int(r_bottom) - 10
                for i, ev in enumerate(reversed(recent_cross[-5:])):
                    age = now - ev["time"]
                    log_label = f"#{ev['track_id']} crossed ({age:.1f}s ago)"
                    log_col = (0, 255, 255) if age < 0.5 else (180, 180, 180)
                    cv2.putText(frame, log_label, (log_x, log_y - i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, log_col, 1, cv2.LINE_AA)

            # Count label (top-right of ROI) — includes zone name
            count_label = f"{c_zn}: {count_val}"
            (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_x = int(r_right) - tw - 10
            label_y = int(r_top) - 10
            if label_y < th + 5:
                label_y = int(r_top) + th + 10
            cv2.rectangle(frame, (label_x - 5, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, count_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

            mode_label = "BLOB/MOG2" if c_mode == "blob" else "GRADIENT"
            angle_label = f"{angle_deg:.0f} deg | {mode_label}"
            cv2.putText(frame, angle_label, (int(r_left) + 5, int(r_top) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

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
