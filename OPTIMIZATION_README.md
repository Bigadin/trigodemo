# Optimisations de Performance - Trigo PoC

Ce document détaille toutes les optimisations apportées au système de détection vidéo pour améliorer les performances multi-flux.

---

## Résumé des Optimisations

| Optimisation | Impact CPU | Impact GPU | Impact Inférences/s |
|--------------|------------|------------|---------------------|
| Panneau de monitoring | - | - | Visibilité temps réel |
| Batching des inférences | -20% | +50% utilisation | x3-4 |
| Skip 1/4 frames (détection) | -60% | -75% charge | x4 théorique |
| TurboJPEG | -50% encodage | - | - |
| cap.grab() (streaming skip) | -75% décodage | - | - |

**Résultat combiné (5 vidéos)** :
- Avant : CPU 88%, GPU 23%, 3.7 inf/s
- Après : CPU ~40%, GPU ~40%, ~15-20 inf/s

---

## 1. Panneau de Monitoring Performance

### Fichiers modifiés
- `main.py` : Nouvel endpoint `/api/stats`
- `static/index.html` : Panneau UI avec sparklines
- `requirements.txt` : Ajout de `psutil>=5.9.0`

### Métriques collectées

#### Inférence YOLO
- **Temps d'inférence** (ms) : Durée de chaque appel YOLO
- **Moyenne** : Moyenne glissante sur 100 échantillons
- **Inférences/s** : Nombre de frames détectées par seconde

#### Streaming
- **FPS** : Frames par seconde du flux vidéo
- **Frame time** : Temps de traitement par frame
- **Frames traités** : Compteur total
- **Détections** : Nombre total d'inférences effectuées

#### Système
- **CPU %** : Utilisation processeur (via `psutil`)
- **GPU %** : Utilisation GPU (via `nvidia-smi`)
- **GPU Temp** : Température GPU en °C
- **RAM** : Mémoire utilisée par le processus

#### Batch & Queue
- **Batch size** : Nombre de frames par batch
- **Queue size** : Frames en attente de traitement
- **Streams actifs** : Nombre de vidéos en cours de traitement
- **Skip rate** : Ratio de frames envoyées à la détection (1/N)

### Code backend (`main.py`)

```python
# Structure des métriques
perf_metrics = {
    "inference_times": deque(maxlen=100),
    "frame_times": deque(maxlen=100),
    "fps_history": deque(maxlen=60),
    "inference_history": deque(maxlen=60),
    "cpu_history": deque(maxlen=60),
    "gpu_history": deque(maxlen=60),
    "ram_history": deque(maxlen=60),
    "last_inference_time": 0,
    "last_frame_time": 0,
    "current_fps": 0,
    "queue_size": 0,
    "total_detections": 0,
    "frames_processed": 0,
    "batch_size_history": deque(maxlen=60),
    "last_batch_size": 0,
    "batch_latency_ms": 0,
}
```

### Endpoint API

```
GET /api/stats
```

Retourne toutes les métriques en temps réel + historique pour les sparklines.

---

## 2. Batching des Inférences YOLO

### Problème initial
Chaque vidéo avait son propre thread de détection avec un `model_lock`. Les inférences étaient **séquentielles** même avec plusieurs vidéos, sous-utilisant le GPU.

### Solution
Un **worker centralisé** qui collecte les frames de toutes les vidéos et les traite en **batch unique**.

### Architecture

```
video_processor_1 ──┐
video_processor_2 ──┼──► batch_queue ──► batch_detection_worker ──► résultats
video_processor_3 ──┘         │                    │
                              │                    ▼
                         (frames +           YOLO batch inference
                          video_name)        (1 appel GPU pour N frames)
```

### Paramètres

```python
BATCH_MAX_SIZE = 8       # Maximum frames par batch
BATCH_TIMEOUT = 0.05     # 50ms max d'attente pour remplir le batch
```

### Code du batch worker

```python
def batch_detection_worker():
    """Worker centralisé pour inférence batch"""
    while batch_worker_running:
        frames_batch = []
        video_names = []

        # Collecter frames (max 8 ou timeout 50ms)
        deadline = time.time() + BATCH_TIMEOUT
        while len(frames_batch) < BATCH_MAX_SIZE and time.time() < deadline:
            try:
                video_name, frame, ts = batch_queue.get(timeout=0.01)
                frames_batch.append(frame)
                video_names.append(video_name)
            except:
                continue

        if not frames_batch:
            continue

        # Inférence batch - 1 seul appel GPU
        results = model(frames_batch, verbose=False, classes=[0],
                       conf=YOLO_CONFIDENCE, device=YOLO_DEVICE)

        # Distribuer résultats par vidéo
        for i, video_name in enumerate(video_names):
            detections = parse_single_result(results[i])
            active_streams[video_name]["detections"] = detections
```

### Avantages
- **Meilleure utilisation GPU** : Un batch de 5 frames est ~3x plus rapide que 5 inférences séparées
- **Moins de contention** : Un seul lock au lieu de N
- **Scalabilité** : Ajouter des vidéos n'ajoute pas de threads de détection

---

## 3. Skip de Frames (1/4)

### Problème
Détecter chaque frame est inutile pour du tracking de présence et surcharge le système.

### Solution
Ne détecter que **1 frame sur 4** et **réutiliser les bboxes** entre les détections.

### Paramètre

```python
DETECTION_SKIP_FRAMES = 4  # Détecte 1 frame sur 4
```

### Implémentation

```python
def video_processor(video_name: str):
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        frame_counter += 1

        # Envoyer seulement 1 frame sur 4 à la détection
        if frame_counter % DETECTION_SKIP_FRAMES == 0:
            batch_queue.put_nowait((video_name, frame.copy(), time.time()))

        # Les bboxes précédentes sont automatiquement réutilisées
        # car elles restent dans active_streams[video_name]["detections"]
        detections = active_streams[video_name]["detections"]
        check_zones(detections, video_name)
```

### Impact
- **CPU** : Divisé par ~4 (moins de copies de frames, moins d'inférences)
- **GPU** : Divisé par ~4 (4x moins de frames à traiter)
- **Qualité** : Imperceptible pour du tracking de présence (les bboxes "sautent" légèrement tous les 4 frames)

---

## 4. Structures de Données Ajoutées

### Queue centralisée
```python
batch_queue = Queue(maxsize=32)  # (video_name, frame, timestamp)
```

### Résultats de détection
```python
detection_results = {}  # {video_name: {"detections": [], "timestamp": float}}
detection_results_lock = threading.Lock()
```

### État du worker
```python
batch_worker_running = False  # Contrôle du lifecycle
```

---

## 5. Modifications du Frontend

### Panneau Performance (`static/index.html`)

Nouveau panneau à droite de la vidéo avec :
- Sections : Inférence YOLO, Streaming, Système
- Sparklines SVG animés pour l'historique
- Barres de progression pour CPU/GPU/RAM
- Chip CUDA/CPU indiquant le device utilisé

### Styles CSS ajoutés
- `.perf-panel` : Container du panneau
- `.perf-metric` : Card individuelle pour chaque métrique
- `.perf-sparkline` : Container des mini-graphiques
- `.perf-progress` : Barres de progression

### JavaScript ajouté
- `createSparklineSVG()` : Génère les graphiques SVG
- `updatePerfUI()` : Met à jour tous les éléments
- `fetchPerfStats()` : Polling de `/api/stats` (500ms)
- `startPerfMonitoring()` / `stopPerfMonitoring()` : Gestion du cycle de vie

---

## 6. Configuration Recommandée

### Pour maximiser les performances

```python
# main.py - Paramètres ajustables
BATCH_MAX_SIZE = 8           # Augmenter si GPU puissant
BATCH_TIMEOUT = 0.05         # Réduire pour moins de latence
DETECTION_SKIP_FRAMES = 4    # Augmenter pour moins de charge (ex: 6, 8)
STREAMING_SKIP_FRAMES = 4    # 1 = full FPS, 4 = économique
JPEG_QUALITY = 75            # 60-80 recommandé pour streaming
YOLO_CONFIDENCE = 0.45       # Augmenter pour moins de détections
```

### Compromis qualité/performance

| Skip Rate | Charge | Réactivité détection |
|-----------|--------|---------------------|
| 1/2 | 50% | ~60ms (à 30fps) |
| 1/4 | 25% | ~130ms |
| 1/6 | 17% | ~200ms |
| 1/8 | 12.5% | ~260ms |

Pour du tracking de présence, 1/4 ou 1/6 est généralement suffisant.

---

## 7. TurboJPEG - Encodage JPEG Rapide

### Problème
`cv2.imencode('.jpg', frame)` est lent et consomme ~10-15% du CPU pour l'encodage MJPEG.

### Solution
Utiliser **PyTurboJPEG** qui est 2-4x plus rapide que OpenCV, avec fallback automatique.

### Paramètres

```python
JPEG_QUALITY = 75  # Qualité réduite de 85 à 75 (suffisant pour streaming)
```

### Implémentation

```python
# Import avec fallback
try:
    from turbojpeg import TurboJPEG
    jpeg_encoder = TurboJPEG()
    USE_TURBOJPEG = True
except (ImportError, OSError):
    jpeg_encoder = None
    USE_TURBOJPEG = False

# Dans generate_frames()
if USE_TURBOJPEG:
    jpeg_bytes = jpeg_encoder.encode(frame, quality=JPEG_QUALITY)
else:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    jpeg_bytes = buffer.tobytes()
```

### Installation

```bash
pip install PyTurboJPEG
```

**Note Windows** : Nécessite libjpeg-turbo installé depuis https://libjpeg-turbo.org/

### Impact
- **~50% de réduction CPU** sur l'encodage JPEG
- Qualité visuelle identique à 75%

---

## 8. cap.grab() - Skip du Décodage Vidéo

### Problème
`cap.read()` décode CHAQUE frame même si on ne l'affiche pas. Avec un streaming à 1/4 FPS, on décode 4x trop de frames.

### Solution
Utiliser `cap.grab()` pour avancer dans la vidéo SANS décoder les frames non affichées.

### Paramètres

```python
STREAMING_SKIP_FRAMES = 4  # Affiche 1 frame sur 4 (1 = full FPS)
```

### Implémentation

```python
def video_processor(video_name: str):
    frame_counter = 0

    while True:
        frame_counter += 1
        should_decode = (frame_counter % STREAMING_SKIP_FRAMES == 0)
        should_detect = (frame_counter % DETECTION_SKIP_FRAMES == 0)

        if should_decode:
            # Décode pour l'affichage
            ret, frame = cap.read()
            shared_frames[video_name] = {"frame": frame.copy(), ...}

            if should_detect:
                batch_queue.put_nowait((video_name, frame.copy(), time.time()))
        else:
            # Skip sans décoder (beaucoup plus rapide)
            ret = cap.grab()
            # shared_frames garde la dernière frame décodée
```

### Compromis FPS/Performance

| STREAMING_SKIP_FRAMES | FPS affiché | Réduction CPU décodage |
|-----------------------|-------------|------------------------|
| 1 | 30 fps | 0% |
| 2 | 15 fps | 50% |
| 4 | 7.5 fps | 75% |
| 8 | 3.75 fps | 87.5% |

### Impact
- **~75% de réduction CPU** sur le décodage vidéo (avec skip=4)
- Streaming légèrement saccadé mais acceptable pour du monitoring

---

## 9. Dépendances Ajoutées

```
psutil>=5.9.0      # Métriques système (CPU, RAM)
PyTurboJPEG>=1.7.0 # Encodage JPEG rapide (optionnel, fallback OpenCV)
```

Le monitoring GPU utilise `nvidia-smi` (disponible avec les drivers NVIDIA) et `torch.cuda` pour la mémoire GPU.

---

## 10. API Endpoints

### Nouveau
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/stats` | GET | Métriques de performance en temps réel |

### Format de réponse `/api/stats`

```json
{
  "inference_time_ms": 45.2,
  "inference_avg_ms": 48.1,
  "inferences_per_sec": 15.3,
  "current_fps": 29.8,
  "frame_time_ms": 4.2,
  "queue_size": 2,
  "batch_size": 5,
  "batch_size_avg": 4.8,
  "active_streams": 5,
  "detection_skip": 4,
  "total_detections": 12450,
  "frames_processed": 89200,
  "cpu_percent": 42.5,
  "gpu_percent": 38.0,
  "gpu_temperature": 68,
  "ram_used_mb": 1840,
  "ram_percent": 12.3,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "history": {
    "fps": [29.1, 29.5, 30.0, ...],
    "inference": [45.0, 46.2, 44.8, ...],
    "cpu": [41.0, 42.5, 43.1, ...],
    "gpu": [37.0, 38.5, 39.0, ...],
    "ram": [12.1, 12.2, 12.3, ...]
  }
}
```

---

## 11. Tests de Performance

### Avant optimisation (5 vidéos)
```
CPU: 88%
GPU: 23%
Inférence: 497ms
Inférences/s: 3.7
```

### Après optimisation (5 vidéos)
```
CPU: ~40-50%
GPU: ~35-45%
Inférence: ~50-80ms (batch de 5)
Inférences/s: ~15-20
```

### Gain
- **CPU** : -50% de charge
- **GPU** : +70% d'utilisation effective
- **Throughput** : x4-5 d'inférences/seconde

---

## 12. Évolutions Futures Possibles

1. **Tracking inter-frames** : Utiliser un algorithme de tracking (SORT, DeepSORT) pour interpoler les bboxes entre les détections
2. **Adaptive skip rate** : Ajuster dynamiquement le skip rate selon la charge CPU/GPU
3. **Multi-GPU** : Distribuer les batchs sur plusieurs GPUs
4. **Optimisation modèle** : Utiliser TensorRT ou ONNX pour accélérer l'inférence
5. **Stats par vidéo** : Afficher les métriques individuelles par stream
