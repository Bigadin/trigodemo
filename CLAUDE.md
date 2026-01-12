# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zone Presence Tracker - A FastAPI web application for tracking human presence in video zones using YOLO object detection. Users can upload videos, draw polygon zones on video frames, and track how long humans are detected within each zone.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Object Detection**: YOLO (ultralytics) with custom model `human.pt`
- **Video Processing**: OpenCV (cv2)
- **Geometry**: Shapely for polygon intersection calculations
- **Frontend**: Single-page HTML/CSS/JS app (no framework)

## Commands

```bash
# Activate virtual environment
.\trigoPoc\Scripts\activate   # Windows
source trigoPoc/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# Server runs at http://0.0.0.0:8000
```

## Architecture

### Backend (main.py)

Single-file FastAPI application with:

- **Video streaming**: MJPEG streaming via `/api/stream/{video_name}` with real-time YOLO detection overlay
- **Zone management**: CRUD operations for polygon zones stored in `data/zones.json`
- **Presence tracking**: Wall-clock time tracking with 1.5s grace period, stored in `data/presence.json`
- **Threading model**: Separate threads for video processing and YOLO detection per video stream
- **Face blurring**: Optional Gaussian blur on top 30% of detection bboxes

Key globals:
- `zones_by_video`: Dict mapping video names to their defined zones
- `zone_timers`: Dict tracking accumulated presence time per zone
- `active_streams`: Dict tracking active video processing threads
- `shared_frames`: Frames shared between processor and streamer threads

### Frontend (static/index.html)

Single HTML file with embedded CSS/JS. Three-step workflow:
1. Select/upload video
2. Draw polygon zones on video frame
3. Start detection and monitor presence times

Canvas-based zone drawing with coordinate scaling to match original video dimensions.

## Key Configuration

- `INTERSECTION_THRESHOLD = 0.30` - Minimum bbox/zone overlap to count as presence
- `GRACE_PERIOD = 1.5` - Seconds to wait before marking zone as unoccupied
- `YOLO_CONFIDENCE = 0.45` - Detection confidence threshold

## Zone Types

Three polygon zone types are supported:

1. **Presence** (orange): Original YOLO-based human presence tracking with timers
2. **Include** (cyan): ROI mask that limits where processing occurs
3. **Counting** (magenta): Background-subtraction based object counting for conveyor belts

## Conveyor Counting System

Ultra-light object counting using background subtraction (no ML model required).

### How It Works
1. Draw an **Include** zone (optional) to limit processing area
2. Draw a **Counting** zone across the conveyor path
3. Start detection - counter auto-calibrates for ~2 seconds
4. Objects crossing the counting zone increment the counter

### Configuration (conveyor_config.py)

Key parameters to tune:
- `BG_SUBTRACTOR_TYPE`: "MOG2" or "KNN"
- `THRESHOLD_MODE`: "auto" (recommended) or "manual"
- `MANUAL_THRESHOLD`: S_norm threshold if using manual mode (default 0.02 = 2%)
- `WARMUP_FRAMES`: Frames for auto-calibration (default 60)
- `THRESHOLD_K`: Multiplier for auto threshold (default 6.0)
- `MIN_OFF_FRAMES`: Anti-double-counting delay (default 5)
- `MIN_ON_FRAMES`: Minimum frames to confirm presence (default 2)
- `MORPH_OPEN_KERNEL_SIZE`, `MORPH_CLOSE_KERNEL_SIZE`: Noise cleanup

### Tuning Tips
- If missing objects: lower `MANUAL_THRESHOLD` or `THRESHOLD_K`
- If double-counting: increase `MIN_OFF_FRAMES`
- If counting noise: increase `NOISE_FLOOR` or morphology kernel sizes
- If lighting changes: use "Calibrer" button when belt is empty

### API Endpoints
- `GET /api/conveyor/{video}` - Get current count and stats
- `POST /api/conveyor/{video}/reset` - Reset counter to zero
- `POST /api/conveyor/{video}/calibrate` - Re-run threshold calibration
- `POST /api/conveyor/{video}/reset-background` - Reset background model

## File Structure

- `main.py` - All backend logic
- `conveyor_config.py` - Conveyor counting configuration
- `conveyor_counter.py` - ConveyorCounter class
- `static/index.html` - Frontend UI
- `videos/` - Uploaded video files
- `data/zones.json` - Zone polygon definitions (includes type field)
- `data/presence.json` - Accumulated presence times
- `human.pt` - Custom YOLO model for human detection
