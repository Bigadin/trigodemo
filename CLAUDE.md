# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trigo Poc is a real-time video presence tracking application that uses YOLOv8 for human detection within user-defined polygonal zones. It consists of a FastAPI backend with a single-page HTML/CSS/JS frontend.

## Commands

### Run the Application
```bash
python main.py
```
Starts the FastAPI server on `http://0.0.0.0:8000` using Uvicorn.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Virtual Environment
The project uses a virtual environment in `trigoPoc/`. Activate it before running:
```bash
# Windows
trigoPoc\Scripts\activate

# Linux/Mac
source trigoPoc/bin/activate
```

## Architecture

### Backend (`main.py`)
Single-file FastAPI application with multi-threaded video processing:

- **Threading Model**: Uses separate threads for video reading (`video_processor`) and YOLO inference (`detection_worker`) with thread-safe locks (`data_lock`, `model_lock`, `streams_lock`, `blur_lock`)
- **MJPEG Streaming**: Real-time frame delivery via `generate_frames()` generator
- **Zone Detection**: Uses Shapely for polygon intersection calculations (30% bbox overlap threshold)
- **Presence Tracking**: Accumulates occupancy time per zone with 1.5s grace period

### Frontend (`static/index.html`)
Self-contained SPA with YRYS design system:
- Interactive polygon drawing for zone definition
- Real-time video preview with detection overlays
- Zone presence time visualization
- Light/dark theme support

### Data Flow
1. Video uploaded → `videos/` directory
2. User draws zones → saved to `data/zones.json`
3. Processing starts → frames read at native FPS
4. Detection thread → YOLO inference (humans only, 0.45 confidence)
5. Zone checker → calculates bbox-polygon intersection
6. Presence timer → tracks with grace period
7. MJPEG stream → delivers frames with overlays
8. Auto-save → `data/presence.json` every 2 seconds

### Key Configuration
- `INTERSECTION_THRESHOLD = 0.30` - Required bbox-zone overlap
- `GRACE_PERIOD = 1.5` - Seconds to maintain occupancy without detection
- `YOLO_CONFIDENCE = 0.45` - Detection threshold
- Automatic GPU (CUDA) detection with CPU fallback

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/videos` | GET | List uploaded videos |
| `/api/videos/upload` | POST | Upload video file |
| `/api/videos/{video}/frame` | GET | Get first frame |
| `/api/videos/{video}/info` | GET | Get video metadata |
| `/api/zones/{video}` | GET | Get zones for video |
| `/api/zones` | POST | Create zone |
| `/api/zones/{video}/{zone}` | PUT/DELETE | Update/delete zone |
| `/api/stream/{video}` | GET | MJPEG stream |
| `/api/stream/{video}/start` | POST | Start processing |
| `/api/stream/{video}/stop` | POST | Stop processing |
| `/api/presence` | GET | Get presence data |
| `/api/blur/toggle` | POST | Toggle face blur |
| `/api/zones/reset` | POST | Reset all timers |

## ML Models

Three YOLO models available in project root:
- `human.pt` (19 MB) - Primary custom model
- `human2.pt` (5.3 MB) - Alternative model
- `yolov8n.pt` (6.3 MB) - Standard YOLOv8 Nano

## File Structure

```
├── main.py              # FastAPI backend (all endpoints + processing)
├── static/index.html    # Frontend SPA
├── data/
│   ├── zones.json       # Zone polygon definitions
│   └── presence.json    # Accumulated presence times
├── videos/              # Uploaded video storage
├── *.pt                 # YOLO model files
└── requirements.txt     # Python dependencies
```
