# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zone Presence Tracker (YRYS) - A real-time video surveillance application for detecting and tracking human presence in defined zones using YOLOv8 object detection.

**Stack:** Python/FastAPI backend + Vanilla JavaScript frontend

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (starts on http://localhost:8000)
python main.py
```

**Docker (with GPU):**
```bash
docker-compose up --build
```

Requires:
- YOLO model file `human.pt` in root directory
- Video files in `/videos` directory

## Architecture

### Backend (main.py)

**Threading Model:**
- `video_processor()` - Reads video frames into a queue (maxsize=5)
- `detection_worker()` - Runs YOLO inference on queued frames
- `generate_frames()` - Streams processed frames to frontend via MJPEG

**Thread Safety:** Uses locks for shared state:
- `data_lock` - zones and timers
- `streams_lock` - active stream metadata and detections
- `frames_lock` - shared frame buffer (separate to reduce contention)
- `blur_lock` - blur state toggle
- `model_lock` - YOLO model inference

**Key Constants:**
```python
INTERSECTION_THRESHOLD = 0.30  # Bbox/zone overlap required for detection
GRACE_PERIOD = 1.5             # Seconds before occupancy considered ended
YOLO_CONFIDENCE = 0.45         # Detection confidence threshold
```

### Frontend (static/)

- `static/index.html` - Entry point
- `static/js/index.js` - Main SPA logic (~3800 lines)
- `static/css/` - Stylesheets

**Views:**
- Sites View - Multi-camera site management
- Tracker View - Zone editing with canvas-based polygon drawing, presence monitoring

**Polling Intervals:**
- Active streams: 350ms for presence updates
- Idle streams: 1000ms polling
- Zone cache TTL: 2500ms

### Data Storage

- `data/zones.json` - Zone polygon definitions per video
- `data/presence.json` - Accumulated presence times

## API Endpoints

**Streaming:**
- `POST /api/stream/{video_name}/start` - Start video processing
- `GET /api/stream/{video_name}` - MJPEG stream with overlays
- `POST /api/stream/{video_name}/stop` - Stop stream
- `POST /api/streams/stop` - Stop all streams
- `GET /api/streams` - List active streams

**Zones:**
- `GET /api/zones/{video_name}` - Get zones
- `POST /api/zones` - Create zone
- `PUT /api/zones/{video_name}/{zone_name}` - Update zone
- `DELETE /api/zones/{video_name}/{zone_name}` - Delete zone

**Presence:**
- `GET /api/presence` - Get all presence data
- `POST /api/zones/reset/{zone_name}` - Reset zone timer

**Videos:**
- `GET /api/videos` - List videos
- `POST /api/videos/upload` - Upload video
- `GET /api/videos/{video_name}/frame` - Get first frame

**Blur:**
- `GET /api/blur` - Get blur state
- `POST /api/blur/toggle` - Toggle blur
- `POST /api/blur/{state}` - Set blur on/off

## Key Patterns

1. **Producer-Consumer:** Frame producer → detection worker → frame streamer
2. **Grace Period Logic:** 1.5s temporal smoothing prevents flickering in occupancy status
3. **Geometric Intersection:** Uses Shapely polygons to calculate bbox/zone overlap (30% threshold)
4. **Wall-clock Timing:** Presence accumulation based on real elapsed time, not frame counts
