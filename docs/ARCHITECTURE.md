# Architecture YRYS — Zone Presence Tracker

## 1. Programmation orientée objet : les objets du domaine

### Entités principales (modèle de données)

| Objet | Rôle | Propriétés clés | Stockage |
|-------|------|-----------------|----------|
| **Site** | Lieu physique regroupant des caméras | `name`, `location`, `cameras[]` | Mémoire (DEMO) / API future |
| **Camera** | Source vidéo (fichier, webcam, RTSP) | `id`, `name`, `video`, `benefits[]`, `sourceType` | `cameras.json` + mémoire |
| **Benefit** | Bénéfice métier lié à une caméra | `id`, `name`, `skill`, `categories`, `polygons`, `forme` | Dans `camera.benefits` |
| **Zone** | Zone géométrique sur une vidéo | `name`, `polygons[]`, `type` (poly/line) | `zones.json` |
| **Presence** | Temps de présence dans une zone | `zone_name`, `total_time`, `is_occupied` | `presence.json` + API temps réel |
| **CountingConfig** | Config du module comptage | `zone_name`, `mode`, `flip_count` | `counting.json` |

### Objets techniques (services / état)

| Objet | Rôle |
|-------|------|
| **Stream** | Flux vidéo actif (détection YOLO, MJPEG) |
| **Detection** | Résultat YOLO (bbox, track_id, confidence) |
| **EditorState** | État de l’éditeur de zones (outil, points, undo) |
| **BenefitConfigState** | État du wizard création/édition de bénéfice |

### Relations

```
Site 1──* Camera 1──* Benefit
                │
                └── video ──* Zone (définitions)
                          ──* Presence (temps réel)
                          ──* CountingConfig
```

---

## 2. Architecture modulaire proposée

### Vue d’ensemble

```
trigodemo/
├── main.py                 # Backend FastAPI (inchangé pour l’instant)
├── static/
│   ├── index.html          # Point d’entrée (chargement des modules)
│   ├── css/
│   └── js/
│       ├── app.js          # Bootstrap, router, state global
│       ├── api.js          # Client HTTP (fetch wrappers)
│       ├── state.js        # Store central (sites, currentView, etc.)
│       └── modules/
│           ├── sidebar/
│           │   ├── sidebar.js
│           │   └── sidebar.css
│           ├── benefit-modal/
│           │   ├── benefit-modal.js
│           │   └── benefit-modal.css
│           ├── tracker/
│           │   ├── tracker.js
│           │   └── tracker.css
│           ├── sites/
│           │   ├── sites.js
│           │   └── sites.css
│           ├── drawing/
│           │   ├── drawing.js
│           │   └── drawing.css
│           └── counting/
│               ├── counting.js
│               └── counting.css
├── docs/
│   └── ARCHITECTURE.md     # Ce fichier
└── scripts/
    └── sync-assets.ps1
```

### Principes

1. **Un module = une responsabilité** : sidebar, modal bénéfice, tracker, etc.
2. **API publique** : chaque module expose `init(container, options)` et éventuellement `render()`, `destroy()`.
3. **Communication** : via un bus d’événements ou un store partagé (`state.js`).
4. **Pas de framework** : Vanilla JS, chargement par `<script>` ou bundler léger plus tard.

---

## 3. Découpage des modules (extraction depuis index.js)

| Module | Responsabilités | Lignes estimées |
|--------|----------------|-----------------|
| **sidebar** | Arbre Lieu > Site > Caméra > Bénéfice, toggles, sélection | ~400 |
| **benefit-modal** | Wizard création/édition bénéfice, steps, catégories | ~600 |
| **tracker** | Vue tracker, onglets Overview/Details/Settings, bénéfices | ~350 |
| **sites** | Vue home, grille/liste sites, création site | ~400 |
| **drawing** | Canvas zones, polygones, lignes, éditeur | ~800 |
| **counting** | Panel comptage, toggle, flip, reset | ~150 |
| **video** | Lecteur vidéo, stream MJPEG, détection | ~300 |
| **api** | `fetch` vers `/api/*` | ~150 |
| **state** | `sitesCache`, `currentView`, `currentCameraId`, etc. | ~100 |

---

## 4. Plan de migration progressif

### Phase 1 — Préparation (sans casser le code)

1. Créer la structure `static/js/modules/`.
2. Créer `api.js` et `state.js` en parallèle de `index.js`.
3. Migrer progressivement les appels `fetch` vers `api.js`.

### Phase 2 — Extraction du premier module

1. Extraire **sidebar** : copier `renderSitesSidebar`, handlers de clic, état `sidebar*Collapsed`.
2. Exposer `SidebarModule.init(zoneListSidebar, { onSelectCamera, onSelectBenefit })`.
3. Remplacer dans `index.js` par un appel à `SidebarModule.init(...)`.

### Phase 3 — Modules suivants

1. **benefit-modal** (le plus gros).
2. **tracker** (Overview, Details, Settings).
3. **sites**, **drawing**, **counting**.

### Phase 4 — Nettoyage

1. Supprimer le code dupliqué dans `index.js`.
2. Réduire `index.js` à un bootstrap + orchestration.

---

## 5. Conventions de code

- **Nommage** : `camelCase` pour les fonctions/variables, `PascalCase` pour les "classes" ou modules.
- **Événements** : préfixe `yrys:` (ex. `yrys:site-selected`, `yrys:benefit-created`).
- **State** : objet unique `window.YRYS` ou `window.YrysState` pour éviter les variables globales éparpillées.

---

## 6. Références

- `CLAUDE.md` — Contexte projet, commandes, stack
- `static/assets_youn/YrysUIPackage/YRYS_UI_Best_Practices.md` — Design system
