# Onboarding équipe — Intégration backend & tests

> Guide pour intégrer et tester le backend sans modifier le frontend.

---

## 1. Architecture globale

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite) — frontend-v2/                                 │
│  ├── Views : SitesView, LieuView, TrackerView, AnalyticsView, LogsView    │
│  ├── API client : fetch vers /api/*                                     │
│  └── Données : HierarchyContext (hiérarchie, lieux, sites, caméras)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP GET/POST/PUT/DELETE
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI) — main.py                                             │
│  ├── /api/hierarchy    → hiérarchie complète lieux/sites/caméras/bénéfices│
│  ├── /api/benefits     → CRUD bénéfices                                  │
│  ├── /api/zones        → zones par vidéo                                 │
│  ├── /api/presence     → présence temps réel                             │
│  ├── /api/counting     → config comptage                                 │
│  ├── /api/videos       → stream, frame, info                             │
│  └── /api/skills       → config skills (détection, comptage, etc.)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Données — data/*.json, fichiers vidéo                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Principes pour l'intégration

1. **Backend = source de vérité** : le frontend consomme les API. Les réponses doivent respecter les schémas attendus.
2. **Pas de logique métier côté front** : le front affiche et envoie les données. Les calculs sont côté backend.
3. **Fichiers de référence** : `frontend-v2/src/api/` contient tous les appels HTTP et les types attendus.

---

## 2. API contract — Endpoints utilisés par le frontend

| Endpoint | Méthode | Rôle | Fichier front |
|----------|---------|------|---------------|
| `/api/hierarchy` | GET | Hiérarchie lieux → sites → caméras → bénéfices | `api/hierarchy.ts` |
| `/api/benefits` | POST | Créer un bénéfice | `api/benefits.ts` |
| `/api/benefits/:id` | PUT, DELETE | Modifier / supprimer un bénéfice | `api/benefits.ts` |
| `/api/benefits/:id` | PUT `{ active }` | Activer/désactiver un bénéfice | `api/benefits.ts` |
| `/api/zones/:videoPath` | GET | Zones par vidéo (polygones, total_time, is_occupied) | `api/tracker.ts` |
| `/api/presence/:videoPath` | GET | Présence temps réel par zone | `api/tracker.ts` |
| `/api/counting/:videoPath` | GET | Config comptage (zone_name, mode, count, enabled) | `api/tracker.ts` |
| `/api/videos/:path/info` | GET | Dimensions vidéo (width, height) | `api/tracker.ts` |
| `/api/videos/:path/frame` | GET | Image frame (pour éditeur zones) | `api/tracker.ts` |
| `/api/streams` | GET | Liste des streams actifs | `api/tracker.ts` |
| `/api/stream/:path/start` | POST | Démarrer stream | `api/tracker.ts` |
| `/api/stream/:path/stop` | POST | Arrêter stream | `api/tracker.ts` |
| `/api/skills` | GET | Config skills (groupes, items, catégories) | `api/skills.ts` |
| `/api/lieux` | GET | Liste des lieux (si utilisé) | `api/lieux.ts` |

---

## 3. Schémas de données attendus

### Hierarchy (GET /api/hierarchy)

```json
{
  "hierarchy": {
    "usine": {
      "lieu_id": "usine",
      "name": "OG Logistics",
      "icon": "/static/assets_youn/OG.jpg",
      "sites": {
        "site_entree": {
          "site_id": "site_entree",
          "name": "Entrée",
          "lieu_id": "usine",
          "cameras": {
            "cam_entree": {
              "camera_id": "cam_entree",
              "name": "Caméra entrée",
              "path": "/videos/entree.mp4",
              "benefits": {
                "ben_xxx": {
                  "benefit_id": "ben_xxx",
                  "name": "Détection présence entrée",
                  "skill": "detection",
                  "skill_item": "detection_presence",
                  "categories": ["human::silhouette", "transport::voiture"],
                  "camera_id": "cam_entree",
                  "zone_polygons": [[[x,y],[x,y],...]],
                  "zone_polygon_types": ["include", "exclude"],
                  "active": true
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### Zones (GET /api/zones/:videoPath)

```json
{
  "zones": {
    "ben_xxx": {
      "polygons": [[[x,y],[x,y],...]],
      "total_time": 120,
      "is_occupied": true
    }
  }
}
```

### Counting (GET /api/counting/:videoPath)

```json
{
  "configured": true,
  "zone_name": "ben_counting_convoyeur",
  "mode": "complex",
  "enabled": true,
  "count": 42
}
```

### Presence (GET /api/presence/:videoPath)

```json
{
  "zones": {
    "ben_xxx": {
      "is_occupied": true,
      "total_time": 120
    }
  }
}
```

---

## 4. Comment tester sans toucher au front

1. **Démarrer le backend** : `python main.py` (ou équivalent)
2. **Démarrer le frontend** : `cd frontend-v2 && npm run dev`
3. **Proxy** : Vite proxy `/api` vers le backend (voir `vite.config.ts`)
4. **Modifier les réponses** : adapter les endpoints dans `main.py` pour que les schémas correspondent à ceux ci-dessus.
5. **Données de démo** : `data/benefits.json`, `data/lieux.json`, `data/hierarchy.json` (ou équivalent) alimentent les réponses.

---

## 5. Assets — Images lieux

| Fichier | Chemin complet | Usage |
|---------|---------------|-------|
| OG.jpg | `/static/assets_youn/OG.jpg` | Image lieu "OG Logistics" (usine) |
| westfield.jpg | `/static/assets_youn/westfield.jpg` | Image lieu "Galerie Westfield" (mall) |

Ces images sont utilisées dans les cartes Vue d'ensemble et Vue Lieu. Le champ `icon` dans l'API hierarchy peut les surcharger.

---

## 6. Documents à consulter

| Document | Contenu |
|----------|---------|
| [VOCABULAIRE_ET_MAPPING.md](./VOCABULAIRE_ET_MAPPING.md) | Mots-clés pour ajouter une carte Data Room ou un paramètre modal bénéfice |
| [SCREENS_ET_ELEMENTS.md](./SCREENS_ET_ELEMENTS.md) | Schémas des pages et noms des éléments UI |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Architecture globale du projet |
| [SPEC_CANVAS_BENEFICES.md](./SPEC_CANVAS_BENEFICES.md) | Spéc éditeur de zones |
