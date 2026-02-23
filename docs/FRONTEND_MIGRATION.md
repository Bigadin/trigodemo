# Plan de migration Frontend — React + TypeScript + Vite

> Réf : [DEPENDENCY_MAP.md](./DEPENDENCY_MAP.md) · [REGRESSION_CHECKLIST.md](./REGRESSION_CHECKLIST.md)

---

## Stack cible

| Outil | Version | Rôle |
|---|---|---|
| **Vite** | 6.x | Bundler, dev server, HMR |
| **React** | 19.x | UI composants |
| **TypeScript** | 5.x | Typage strict |
| **Zustand** | 5.x | State management |
| **TanStack Query** | 5.x | Cache API, polling, mutations |
| **CSS Modules** | (natif Vite) | Styles scopés par composant |

### Pas de dépendances lourdes

- Pas de Redux, pas de MUI, pas de Tailwind (design system custom existant)
- Canvas editor : vanilla Canvas API (pas de lib externe — trop spécifique)
- Graphiques analytics : vanilla Canvas API (déjà implémenté)

---

## Architecture cible

```
src/
├── main.tsx                  # Point d'entrée React
├── App.tsx                   # Router + Layout
├── api/                      # Couche API typée
│   ├── client.ts             # fetch wrapper (_api.get/post/put/del)
│   ├── hierarchy.ts          # lieux, sites, hierarchy
│   ├── benefits.ts           # CRUD benefits
│   ├── cameras.ts            # CRUD cameras + detect/RTSP/ONVIF
│   ├── zones.ts              # CRUD zones + reset
│   ├── streams.ts            # start/stop/list streams
│   ├── videos.ts             # list/upload/info/optimize
│   ├── counting.ts           # config/toggle/flip/reset/params
│   ├── presence.ts           # GET presence
│   ├── metrics.ts            # perf metrics + logs + blur
│   └── skills.ts             # GET skills config
├── types/                    # Interfaces TypeScript
│   ├── site.ts               # Lieu, Site, Camera, Benefit
│   ├── zone.ts               # Zone, Polygon, LineArrow
│   ├── editor.ts             # EditorState, Tool, UndoEntry
│   ├── counting.ts           # CountingState, CountingMode
│   ├── skills.ts             # Skill, SkillGroup, CategoryGroup
│   └── api.ts                # ApiResponse<T>, ErrorResponse
├── stores/                   # Zustand stores (slices par domaine)
│   ├── siteStore.ts          # sitesCache, currentSite, CRUD
│   ├── videoStore.ts         # currentVideo, currentCameraId, streams, timers
│   ├── editorStore.ts        # tool, zone, polygons, undo/redo
│   ├── benefitStore.ts       # wizard state (step, skill, categories)
│   ├── countingStore.ts      # countingState par vidéo
│   └── uiStore.ts            # sidebar collapsed, view, filters
├── hooks/                    # React hooks custom
│   ├── useZonesPolling.ts    # Polling zones + présence (1.2s / 5s adaptatif)
│   ├── useStreamsPolling.ts   # Polling streams actifs (2s)
│   ├── useLogsPolling.ts     # Polling logs (2s)
│   ├── useVideoInfo.ts       # Charge info vidéo (dimensions)
│   └── useKeyboardShortcuts.ts
├── components/
│   ├── layout/
│   │   ├── AppLayout.tsx     # Sidebar + Main + Topbar
│   │   ├── TopBar.tsx        # Navigation principale
│   │   ├── Sidebar.tsx       # Conteneur sidebar
│   │   └── Breadcrumb.tsx    # Fil d'ariane tracker
│   ├── sites/
│   │   ├── SitesView.tsx     # Vue "Home" — grille/liste sites
│   │   ├── SiteCard.tsx      # Carte individuelle site
│   │   └── CreateSiteForm.tsx
│   ├── sidebar/
│   │   ├── ExplorerTree.tsx  # Arbre lieux → sites → caméras → bénéfices
│   │   ├── TreeNode.tsx      # Noeud arbre (collapse/expand)
│   │   └── BenefitFilterPill.tsx
│   ├── tracker/
│   │   ├── TrackerView.tsx   # Vue principale tracker
│   │   ├── TrackerHeader.tsx # Header avec KPIs (étapes 1/2/3)
│   │   ├── DataRoom.tsx      # Panneau mesures ROI
│   │   ├── RoiCard.tsx       # Carte mesure individuelle
│   │   ├── BenefitsPanel.tsx # Liste bénéfices caméra
│   │   ├── BenefitRow.tsx    # Ligne bénéfice avec badges
│   │   ├── SiteOverview.tsx  # Vue d'ensemble site (toutes caméras)
│   │   └── TrackerTabs.tsx   # Onglets Overview/Details/Settings
│   ├── video/
│   │   ├── VideoPlayer.tsx   # Lecteur vidéo/stream + canvas overlay
│   │   ├── VideoSelector.tsx # Select vidéo + upload
│   │   ├── CameraGrid.tsx    # Grille caméras du site
│   │   ├── CameraTile.tsx    # Tuile caméra individuelle
│   │   └── StreamBadges.tsx  # Badges streams actifs
│   ├── editor/
│   │   ├── EditorOverlay.tsx # Modal plein écran éditeur
│   │   ├── EditorCanvas.tsx  # Canvas interactif (pointer events)
│   │   ├── EditorToolbar.tsx # Barre outils (select, include, exclude, line)
│   │   ├── EditorZoneList.tsx# Liste zones dans l'éditeur
│   │   ├── EditorHoverBar.tsx# Barre contextuelle au survol
│   │   ├── ShapeLabels.tsx   # Labels des formes sur le canvas
│   │   └── useEditorCanvas.ts# Hook logique canvas (pointer, draw, undo)
│   ├── benefits/
│   │   ├── BenefitWizard.tsx # Wizard multi-étapes création bénéfice
│   │   ├── SkillStep.tsx     # Étape 1 : choix skill
│   │   ├── ConfigStep.tsx    # Étape 2 : sous-skill + catégories
│   │   ├── DrawStep.tsx      # Étape 3 : dessin zones (intégré éditeur)
│   │   ├── FinalizeStep.tsx  # Étape 4 : récap + sauvegarde
│   │   ├── CategoryTree.tsx  # Arbre sélection catégories
│   │   └── SkillCard.tsx     # Carte skill (detection, counting…)
│   ├── detection/
│   │   ├── DetectionControls.tsx # Start/Stop/StopAll
│   │   └── BlurControls.tsx     # Toggle blur + anonymize
│   ├── counting/
│   │   ├── CountingPanel.tsx    # Config comptage (zone, mode)
│   │   ├── CountingDisplay.tsx  # Affichage compteur live
│   │   └── CountingParams.tsx   # Sliders threshold/cooldown
│   ├── analytics/
│   │   ├── AnalyticsView.tsx    # Vue dashboard analytics
│   │   ├── HeatmapCard.tsx      # Widget heatmap (canvas)
│   │   ├── DivergingBarCard.tsx # Widget barres divergentes
│   │   ├── StatsBarCard.tsx     # Widget barres stats
│   │   ├── CircularCard.tsx     # Widget jauge circulaire
│   │   └── CardMenu.tsx         # Menu contextuel widget
│   ├── logs/
│   │   ├── LogsView.tsx         # Console logs + filtres
│   │   ├── LogEntry.tsx         # Ligne de log
│   │   └── PerfChart.tsx        # Graphique performance (canvas)
│   ├── modals/
│   │   ├── AppModal.tsx         # Alert/Confirm réutilisable
│   │   └── AddCameraModal.tsx   # Modal ajout caméra
│   └── shared/
│       ├── Badge.tsx            # Badge/chip réutilisable
│       ├── LovDropdown.tsx      # Select custom (ex lov-dropdown.js)
│       └── IconSvg.tsx          # Wrapper SVG icons
├── styles/
│   ├── tokens.css               # Variables CSS extraites de index.css
│   ├── reset.css                # Reset + typographie de base
│   └── *.module.css             # Styles scopés par composant
└── utils/
    ├── format.ts                # formatHMS, truncateFilename…
    ├── geometry.ts              # dist2, pointInPoly, nearestVertex…
    └── colors.ts                # getLocColor, getBenefitColor, colorsForType
```

---

## Phases de migration

### Phase 0 — Scaffolding (1 session)
**But :** Initialiser le projet Vite+React+TS à côté du code existant

- [ ] `npm create vite@latest` dans le repo
- [ ] Configurer `vite.config.ts` (proxy API vers `:8000`, alias `@/`)
- [ ] Configurer `tsconfig.json` (strict, paths)
- [ ] Extraire `styles/tokens.css` depuis `index.css` (lignes 1–137 : variables CSS)
- [ ] Créer `api/client.ts` (copie typée de `_api`)
- [ ] Créer `types/site.ts`, `types/zone.ts` (d'après les specs)
- [ ] Créer `App.tsx` avec router basique (4 vues)
- [ ] **Régression** : Le frontend vanilla existant (`/static/index.html`) continue de marcher en parallèle

### Phase 1 — Layout + Navigation (1 session)
**But :** Reproduire le shell "client lourd" (sidebar + topbar + routing)

Dépendances : Phase 0
- [ ] `AppLayout.tsx` : sidebar fixe + main content
- [ ] `TopBar.tsx` : navigation Sites / Tracker / Analytics / Logs
- [ ] `Sidebar.tsx` : conteneur (contenu vide pour l'instant)
- [ ] Routing : `/` → Sites, `/tracker` → Tracker, `/analytics` → Analytics, `/logs` → Logs
- [ ] `siteStore.ts` : `loadHierarchy()` depuis `/api/hierarchy`
- [ ] `uiStore.ts` : `currentView`, `sidebarCollapsed`
- [ ] **Régression** : Navigation fluide entre les 4 vues, sidebar visible

### Phase 2 — Sites + Sidebar (1–2 sessions)
**But :** Vue "Home" fonctionnelle + arbre explorateur

Dépendances : Phase 1
- [ ] `SitesView.tsx` : grille/liste sites avec toggle
- [ ] `SiteCard.tsx` : carte site avec badges (caméras, bénéfices)
- [ ] `CreateSiteForm.tsx` : formulaire création site
- [ ] `ExplorerTree.tsx` : arbre lieux → sites → caméras → bénéfices
- [ ] `TreeNode.tsx` : noeud avec collapse/expand animé
- [ ] `siteStore.ts` : CRUD complet (create, delete, select)
- [ ] `Breadcrumb.tsx` : fil d'ariane avec LOV dropdowns
- [ ] `LovDropdown.tsx` : migration de `lov-dropdown.js`
- [ ] **Régression** : Créer/supprimer site, naviguer dans l'arbre, breadcrumb fonctionnel

### Phase 3 — Tracker core (2–3 sessions)
**But :** Vue Tracker avec vidéo, caméras, bénéfices

Dépendances : Phase 2
- [ ] `TrackerView.tsx` : layout principal tracker
- [ ] `TrackerHeader.tsx` : KPIs (étapes 1/2/3 avec compteurs)
- [ ] `VideoPlayer.tsx` : `<img>` pour stream MJPEG + canvas overlay
- [ ] `CameraGrid.tsx` + `CameraTile.tsx` : tuiles caméras du site
- [ ] `AddCameraModal.tsx` : modal ajout caméra (webcam/RTSP/ONVIF/vidéo)
- [ ] `videoStore.ts` : `currentVideo`, `selectCamera`, `selectVideo`
- [ ] `BenefitsPanel.tsx` + `BenefitRow.tsx` : liste bénéfices avec badges
- [ ] `TrackerTabs.tsx` : onglets Overview / Details / Settings
- [ ] `hooks/useStreamsPolling.ts` : polling streams (2s)
- [ ] `hooks/useZonesPolling.ts` : polling zones + présence (adaptatif)
- [ ] **Régression** : Sélectionner caméra, voir vidéo, lister bénéfices, voir présences live

### Phase 4 — DataRoom + Detection (1–2 sessions)
**But :** Cartes ROI, contrôles détection, comptage

Dépendances : Phase 3
- [ ] `DataRoom.tsx` + `RoiCard.tsx` : cartes mesures avec métriques live
- [ ] `DetectionControls.tsx` : start/stop détection + stopAll
- [ ] `BlurControls.tsx` : toggle blur + anonymize
- [ ] `CountingPanel.tsx` : config comptage (zone, mode, toggle, flip)
- [ ] `CountingDisplay.tsx` : affichage compteur live
- [ ] `CountingParams.tsx` : sliders threshold/cooldown
- [ ] `SiteOverview.tsx` : vue multi-caméras du site
- [ ] `countingStore.ts` : état comptage par vidéo
- [ ] **Régression** : Démarrer détection, voir présences, voir comptage, blur fonctionne

### Phase 5 — Editor canvas (2–3 sessions) ⚠️ LE PLUS COMPLEXE
**But :** Éditeur de polygones/lignes plein écran

Dépendances : Phase 3
- [ ] `EditorOverlay.tsx` : modal plein écran
- [ ] `EditorCanvas.tsx` : canvas interactif (pointer events)
- [ ] `useEditorCanvas.ts` : hook avec toute la logique (draw, move, resize, undo)
- [ ] `EditorToolbar.tsx` : outils (select, include, exclude, count_line)
- [ ] `EditorZoneList.tsx` : liste zones avec sélection
- [ ] `EditorHoverBar.tsx` : barre contextuelle (delete point/shape)
- [ ] `ShapeLabels.tsx` : labels des formes sur le canvas
- [ ] `editorStore.ts` : tool, zone, polygons, undo stack
- [ ] Migrer les ~1 300 lignes de logique géométrique dans `utils/geometry.ts`
- [ ] `api/zones.ts` : CRUD zones (POST/PUT/DELETE)
- [ ] **Régression** : Ouvrir éditeur, dessiner polygone, dessiner ligne, undo, sauvegarder, fermer, zones persistées

### Phase 6 — Wizard bénéfice (1–2 sessions)
**But :** Création/édition bénéfice multi-étapes

Dépendances : Phase 5 (utilise l'éditeur)
- [ ] `BenefitWizard.tsx` : orchestrateur des 4 étapes
- [ ] `SkillStep.tsx` : sélection skill (detection/counting/heatmap/quality)
- [ ] `ConfigStep.tsx` : sous-skill + arbre catégories
- [ ] `DrawStep.tsx` : intégration éditeur canvas
- [ ] `FinalizeStep.tsx` : récap + sauvegarde
- [ ] `CategoryTree.tsx` : arbre avec checkbox (family → class)
- [ ] `benefitStore.ts` : état wizard complet
- [ ] `api/benefits.ts` : CRUD bénéfices
- [ ] **Régression** : Créer bénéfice bout en bout (skill → config → draw → save), éditer existant, supprimer

### Phase 7 — Analytics + Logs (1 session)
**But :** Dashboard analytics et console logs

Dépendances : Phase 1 (indépendant du tracker)
- [ ] `AnalyticsView.tsx` : grille widgets drag/reorganize
- [ ] `HeatmapCard.tsx` : widget heatmap canvas
- [ ] `DivergingBarCard.tsx`, `StatsBarCard.tsx`, `CircularCard.tsx`
- [ ] `CardMenu.tsx` : menu contextuel (fullscreen, delete, export)
- [ ] `LogsView.tsx` : console + filtres catégorie
- [ ] `LogEntry.tsx` : ligne de log formatée
- [ ] `PerfChart.tsx` : graphique performance canvas
- [ ] `hooks/useLogsPolling.ts` : polling logs (2s)
- [ ] **Régression** : Widgets analytics s'affichent, logs chargent, filtres fonctionnent, perf chart dessine

### Phase 8 — Polish + Nettoyage (1 session)
**But :** Supprimer l'ancien code, optimiser

- [ ] Supprimer `static/js/index.js`, `static/js/skills-adapter.js`, `static/js/lov-dropdown.js`, `static/js/utils.js`
- [ ] Supprimer l'ancien `static/index.html` (remplacé par le build Vite)
- [ ] Migrer `static/css/index.css` → CSS Modules par composant + `tokens.css`
- [ ] `static/spec.html` reste en standalone (pas de migration)
- [ ] Vérifier que toutes les fonctionnalités sont couvertes (cf. REGRESSION_CHECKLIST)
- [ ] Build de production `vite build`
- [ ] Configurer le backend pour servir le build Vite (`/static/dist/`)

---

## Ordre de migration — Justification

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
  (infra)    (shell)    (sites)    (tracker)   (dataroom)
                                       │
                                       ├──► Phase 5 ──► Phase 6
                                       │    (editor)    (wizard)
                                       │
                                       └──► Phase 7
                                            (analytics)
                                                 │
                                            Phase 8
                                           (cleanup)
```

**Pourquoi cet ordre :**
1. **Phase 0–1** : Le shell doit exister avant tout composant
2. **Phase 2** : Sites + Sidebar = le squelette de navigation, nécessaire pour tout le reste
3. **Phase 3** : Le Tracker est le coeur métier, il doit marcher avant les sous-modules
4. **Phase 4** : DataRoom dépend de zones polling (Phase 3)
5. **Phase 5** : L'éditeur est le plus complexe mais ne bloque pas les phases 3-4
6. **Phase 6** : Le wizard utilise l'éditeur (Phase 5)
7. **Phase 7** : Analytics/Logs sont indépendants, parallélisables avec Phase 5-6
8. **Phase 8** : Nettoyage uniquement quand tout est migré

---

## Coexistence ancien/nouveau

Pendant la migration, les deux frontends coexistent :

| URL | Sert |
|---|---|
| `http://localhost:8000/` | Ancien frontend (static/index.html) — INCHANGÉ |
| `http://localhost:5173/` | Nouveau frontend React (Vite dev server, proxy API → :8000) |

Le backend Python n'a **aucune modification** à subir. Les mêmes APIs servent les deux frontends.

Quand Phase 8 est terminée, on remplace la route `/` du backend pour servir le build Vite.
