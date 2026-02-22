# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zone Presence Tracker (YRYS) - A real-time video surveillance application for detecting and tracking human presence in defined zones using YOLOv8 object detection, with a blob-based people counting module.

**Stack:** Python/FastAPI backend (single file `main.py`) + Vanilla JavaScript SPA frontend

## Running the Application

```bash
# Activate the existing venv
trigoPoc\Scripts\Activate.ps1   # Windows PowerShell

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
- Video files in `videos/` directory

There are no tests, linter, or build steps configured for this project.

## Architecture

### Backend (`main.py` — monolith ~1900 lines)

Everything is in a single file. The backend has three major subsystems:

**1. Presence Detection (YOLO + Zones)**
- `video_processor()` — reads frames from video/camera into a Queue(maxsize=5)
- `detection_worker()` — creates a **dedicated YOLO model instance per stream** (avoids tracker state pollution between streams), runs `model.track()` with ByteTrack
- `generate_frames()` — reads from `shared_frames`, draws overlays (zones, detections, blur, counting), streams MJPEG
- `check_zones()` — computes bbox/zone overlap using Shapely, manages grace-period timers

**2. Counting Module (Blob/MOG2 — independent from YOLO)**
- `update_counting_blob()` — background subtraction on ROI polygon, contour detection, proximity-based blob tracking, line-crossing counting
- Each video gets its own `cv2.BackgroundSubtractorMOG2` instance
- Counting parameters are configurable at runtime via API and persisted to `data/counting_params.json`
- Direction auto-computed from polygon's minimum bounding rectangle; can be flipped in 90° increments

**3. Camera Sources**
- Supports video files, webcams (by device ID), and RTSP streams
- Camera sources use `camera:{id}` naming convention throughout the codebase
- ONVIF discovery available via WS-Discovery (optional dependency)

**Thread Safety — 6 locks for shared state:**
- `data_lock` — zones and presence timers
- `streams_lock` — active stream metadata and detections
- `frames_lock` — shared frame buffer (separate to reduce contention with streams_lock)
- `blur_lock` — blur state toggle
- `model_lock` — YOLO model (note: each stream now creates its own model instance)
- `counting_lock` — counting state, tracked objects, MOG2 runtime data

**Key Constants:**
```python
INTERSECTION_THRESHOLD = 0.30  # Bbox/zone overlap required for detection
GRACE_PERIOD = 0.5             # Seconds before occupancy considered ended
YOLO_CONFIDENCE = 0.45         # Detection confidence threshold
```

### Frontend (`static/`)

- `static/index.html` — entry point, served at `/`
- `static/js/index.js` — entire SPA logic (~4000+ lines, all in one file)
- `static/css/index.css` — styles

**Views:** Sites View (multi-camera site management) and Tracker View (zone drawing with canvas polygons, presence monitoring, counting controls).

**Frontend polling intervals:**
- Active streams: 2500ms for presence updates
- Idle streams: 5000ms polling
- Zone definitions cache TTL: 2500ms

### Data Storage (JSON files in `data/`)

- `zones.json` — zone polygon definitions keyed by video name
- `presence.json` — accumulated presence times per zone
- `cameras.json` — configured camera sources (webcam/RTSP)
- `counting.json` — counting module config per video (zone name, flip count)
- `counting_params.json` — tunable MOG2/blob algorithm parameters

## API Endpoints

**Streaming:** `POST /api/stream/{video_name}/start`, `GET /api/stream/{video_name}` (MJPEG), `POST /api/stream/{video_name}/stop`, `POST /api/streams/stop`, `GET /api/streams`

**Zones:** `GET /api/zones/{video_name}`, `POST /api/zones`, `PUT /api/zones/{video_name}/{zone_name}`, `DELETE /api/zones/{video_name}/{zone_name}`, `DELETE /api/zones/{video_name}` (delete all)

**Presence:** `GET /api/presence`, `GET /api/presence/{video_name}`, `POST /api/zones/reset` (all), `POST /api/zones/reset/{zone_name}`

**Videos:** `GET /api/videos`, `POST /api/videos/upload`, `GET /api/videos/{video_name}/frame`, `GET /api/videos/{video_name}/info`

**Counting:** `GET /api/counting/{video_name}`, `POST /api/counting/{video_name}/config`, `POST /api/counting/{video_name}/toggle`, `POST /api/counting/{video_name}/flip`, `POST /api/counting/{video_name}/reset`, `GET /api/counting/params`, `PUT /api/counting/params`

**Cameras:** `GET /api/cameras`, `POST /api/cameras`, `DELETE /api/cameras/{id}`, `GET /api/cameras/detect/webcams`, `GET /api/cameras/detect/onvif`, `POST /api/cameras/test-rtsp`, `POST /api/cameras/{id}/test`, `GET /api/cameras/{id}/frame`

**Blur:** `GET /api/blur`, `POST /api/blur/toggle`, `POST /api/blur/{state}`

## Key Patterns

1. **Producer-Consumer:** Frame producer → detection worker (YOLO) → frame streamer. Counting runs in the producer thread, independent of YOLO.
2. **Per-Stream Model Instances:** Each active stream gets its own YOLO model to prevent ByteTrack state leaking between videos.
3. **Grace Period Logic:** 0.5s temporal smoothing prevents flickering in occupancy status.
4. **Geometric Intersection:** Uses Shapely polygons to calculate bbox/zone overlap (30% threshold).
5. **Wall-clock Timing:** Presence accumulation based on real elapsed time, not frame counts.
6. **MOG2 Background Learning:** Counting module learns background for N frames, then freezes the model (`learningRate=0`).

## Icônes SVG (synchronisation systématique)

**Dossier source des icônes :** `static/assets_youn/SvIcons/SVGnew/`

Toutes les références d’icônes dans `index.js` doivent pointer vers des fichiers présents dans ce dossier. Lors de l’ajout ou du renommage d’icônes :

1. **Vérifier la présence des fichiers** dans `static/assets_youn/SvIcons/SVGnew/`
2. **Mettre à jour les chemins** dans `getBenefitSkillGroups()`, `getBenefitCategoryGroupsBySkill()` et `BENEFIT_SUBCATEGORIES`
3. **Fichiers avec espaces** : utiliser l’encodage URL (`Ypublic%20transport.svg`)
4. **Incrémenter le cache** : `index.js?v=X` dans `static/index.html`

**Références actuelles des sous-skills :**
- Counting : `Ycountingppl.svg`, `polygon-line-check-svgrepo-com.svg`, `polygon-svgrepo-com.svg`
- Heatmap : `Yheatmapdense.svg`, `Ytraj.svg`
- Quality : `Yqualitydefect.svg`, `Yobstruction.svg`, `Yfissure.svg`, `Yhumidity.svg`
- Transport (catégorie) : `Ytransport2.svg` (ou `Ycar.svg` si absent)
- Public transport : `Ypublic%20transport.svg`

**Important :** Si le projet utilise des worktrees Git, s’assurer que les modifications sont bien appliquées dans le dossier depuis lequel le serveur est lancé (`python main.py`).

**Script de synchronisation :** `scripts/sync-assets.ps1` — exécuter après modifications pour garder Documents et worktree vpi alignés :
```powershell
.\scripts\sync-assets.ps1 both
```

## UI Design System (YRYS Brand)

See `static/assets_youn/YrysUIPackage/YRYS_UI_Best_Practices.md` for full details. Key rules:
- Use CSS design tokens (variables) only — no hardcoded colors
- Font: Manrope family
- Brand colors: Orange `#F08321`, Cyan `#10B0F9`, Blue `#062DB6`, Purple `#BD44D5`
- Light/dark theme support via `data-theme` attribute
- Spacing scale: 4/8/12/16/24/32/48px only
