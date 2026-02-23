# Carte des interdépendances — Frontend Trigodemo

> Référence pour la migration React+TS+Vite. Chaque module listé ici deviendra un composant/hook/store React.

---

## 1. Inventaire du monolithe `index.js` (7 740 lignes)

### Découpage cible en modules

| Module cible | Lignes actuelles | Fonctions clés | Complexité |
|---|---|---|---|
| `api/client.ts` | 12–17 | `_api.get/post/put/del` | Faible |
| `api/hierarchy.ts` | 25–112 | `_syncSitesToBackend`, `_loadHierarchyFromBackend` | Moyenne |
| `stores/siteStore.ts` | 140–180, 3529–3560, 4353–4420 | `sitesCache`, `currentSite`, `loadSites`, `createSite`, `deleteSiteByName` | Moyenne |
| `stores/videoStore.ts` | 182–276 | `currentVideo`, `markVideoRunStart/Stop`, `activeVideoStreams`, timers | Moyenne |
| `stores/editorStore.ts` | ~650 | `editorState` (tool, zone, points, polygons, dirty, undo) | Élevée |
| `stores/benefitStore.ts` | ~1950 | `benefitConfigState` (step, skill, skillSub, category…) | Moyenne |
| `stores/countingStore.ts` | 267–276 | `countingStateByVideo`, `fetchCountingState` | Faible |
| `components/editor/` | 610–1920 | 40+ fonctions editor* | **Très élevée** |
| `components/benefits/` | 1979–2810 | Wizard multi-étapes, config bénéfices | Élevée |
| `components/tracker/` | 3166–3520 | DataRoom, cartes ROI, overview | Moyenne |
| `components/sidebar/` | 3808–4350 | Arbre explorateur sites/caméras/bénéfices | Moyenne |
| `components/sites/` | 3598–3808 | Grille/liste sites, filtres | Faible |
| `components/video/` | 5501–5680, 6411–6840 | Player, canvas, streams | Moyenne |
| `components/zones/` | 5750–6107 | loadZones, polling, cache | Moyenne |
| `components/detection/` | 7078–7291 | Start/stop détection, blur, anonymize | Faible |
| `components/counting/` | 7309–7493 | UI comptage, config, toggle | Faible |
| `components/analytics/` | 7537–8452 | Dashboard widgets, heatmap, graphiques canvas | Élevée |
| `components/logs/` | 4608–4660 | Console logs, auto-refresh, filtres | Faible |
| `hooks/usePolling.ts` | Dispersé | `setInterval` zones, streams, logs | Moyenne |
| `hooks/useVideoStream.ts` | 5579–5668 | `selectVideo`, `selectCamera` | Moyenne |

---

## 2. Graphe de dépendances entre modules

```
┌─────────────┐
│  api/client │ ← Tous les modules
└──────┬──────┘
       │
┌──────▼──────────────┐     ┌───────────────────┐
│  stores/siteStore   │◄────│  api/hierarchy    │
│  (sitesCache,       │     └───────────────────┘
│   currentSite)      │
└──┬───┬───┬──────────┘
   │   │   │
   │   │   └──────────────────────────────────────┐
   │   │                                          │
┌──▼───▼──────────┐  ┌──────────────────┐  ┌─────▼─────────────┐
│ stores/videoStore│  │stores/editorStore│  │stores/benefitStore│
│ (currentVideo,   │  │(editorState)     │  │(benefitConfigState)│
│  timers, streams)│  └────┬─────────────┘  └────┬──────────────┘
└──┬──┬────────────┘       │                     │
   │  │              ┌─────▼──────────┐    ┌─────▼──────────────┐
   │  │              │ editor/        │    │ benefits/          │
   │  │              │ (40 fonctions) │    │ (wizard, config)   │
   │  │              └─────┬──────────┘    └─────┬──────────────┘
   │  │                    │                     │
   │  │              ┌─────▼─────────────────────▼──┐
   │  └──────────────► tracker/                     │
   │                 │ (DataRoom, ROI, overview)     │
   │                 └──────┬───────────────────────┘
   │                        │
┌──▼────────────┐    ┌──────▼──────────┐
│ video/        │    │ sidebar/        │
│ (player,      │◄───│ (arbre sites/   │
│  canvas)      │    │  caméras)       │
└──┬────────────┘    └────────────────┘
   │
┌──▼────────────┐  ┌───────────────┐  ┌────────────┐
│ zones/        │  │ detection/    │  │ counting/  │
│ (polling,     │  │ (start/stop)  │  │ (UI, ROI)  │
│  cache)       │  └───────────────┘  └────────────┘
└───────────────┘

┌───────────────┐  ┌───────────────┐
│ analytics/    │  │ logs/         │
│ (indépendant) │  │ (indépendant) │
└───────────────┘  └───────────────┘
```

---

## 3. Variables d'état globales → Zustand stores

### `siteStore` — Source de vérité pour la hiérarchie

| Variable actuelle | Type | Consommateurs |
|---|---|---|
| `sitesCache` | `Site[]` | sidebar, tracker, sites, benefits, editor, video |
| `currentSite` | `Site \| null` | tracker, sidebar, header, zones, detection, counting |
| `sidebarLocationsCollapsed` | `Record<string, boolean>` | sidebar |
| `sidebarSitesCollapsed` | `Record<string, boolean>` | sidebar |
| `sidebarCamerasCollapsed` | `Record<string, boolean>` | sidebar |
| `sidebarLocationFilter` | `string \| null` | sidebar |
| `selectedBenefitId` | `string \| null` | tracker, zones, editor |
| `selectedBenefitCamId` | `string \| null` | tracker, zones |

### `videoStore` — État vidéo et détection

| Variable actuelle | Type | Consommateurs |
|---|---|---|
| `currentVideo` | `string \| null` | video, zones, detection, counting, editor, tracker |
| `currentCameraId` | `string \| null` | tracker, benefits, editor, camera |
| `activeVideoStreams` | `Set<string>` | video, detection, zones, tracker |
| `videoRunAccumSecByVideo` | `Record<string, number>` | tracker, ROI |
| `videoRunStartTsByVideo` | `Record<string, number>` | video |
| `videoHasRunByVideo` | `Record<string, boolean>` | tracker, detection |
| `lastPresenceByVideo` | `Record<string, object>` | tracker, ROI |
| `zoneLiveTimersByVideo` | `Record<string, object>` | tracker, ROI, zones |
| `zonesCacheByVideo` | `Record<string, object>` | zones, counting, detection, tracker |
| `countingStateByVideo` | `Record<string, object>` | counting, ROI, zones |

### `editorStore` — État de l'éditeur canvas

| Variable actuelle | Type | Consommateurs |
|---|---|---|
| `editorState.tool` | `'select' \| 'include' \| 'exclude' \| 'count_line'` | editor seulement |
| `editorState.zone` | `string \| null` | editor seulement |
| `editorState.points` | `number[][]` | editor seulement |
| `editorState.polygons` | `Polygon[]` | editor seulement |
| `editorState.dirty` | `boolean` | editor seulement |
| `editorState.undo` | `UndoEntry[]` | editor seulement |

### `benefitStore` — État du wizard bénéfice

| Variable actuelle | Type | Consommateurs |
|---|---|---|
| `benefitConfigState.step` | `'skill' \| 'config' \| 'draw' \| 'finalize'` | benefits seulement |
| `benefitConfigState.skill` | `string` | benefits, editor |
| `benefitConfigState.skillSub` | `string` | benefits |
| `benefitConfigState.selectedCategories` | `CategorySelection[]` | benefits |
| `benefitConfigState.editingBenefitId` | `string \| null` | benefits |

---

## 4. Contrats API (45 endpoints) → `api/*.ts`

### `api/hierarchy.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/hierarchy` | `siteStore.loadHierarchy()` |
| POST | `/api/lieux` | `siteStore.createSite()` |
| POST | `/api/sites` | `siteStore.createSite()` |
| DELETE | `/api/sites/{id}` | `siteStore.deleteSite()` |

### `api/benefits.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/benefits` | sync |
| POST | `/api/benefits` | `benefitStore.save()` |
| PUT | `/api/benefits/{id}` | `benefitStore.update()` |
| DELETE | `/api/benefits/{id}` | `benefitStore.delete()` |

### `api/cameras.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/cameras` | `videoStore.loadBackendCameras()` |
| POST | `/api/cameras` | addCameraModal |
| DELETE | `/api/cameras/{id}` | deleteCamera |
| GET | `/api/cameras/detect/webcams` | addCameraModal |
| POST | `/api/cameras/test-rtsp` | addCameraModal |
| GET | `/api/cameras/detect/onvif` | addCameraModal |

### `api/zones.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/zones/{video}` | zones, editor, tracker (×8 occurrences) |
| POST | `/api/zones` | editor, zones |
| PUT | `/api/zones/{video}/{zone}` | editor, zones |
| DELETE | `/api/zones/{video}/{zone}` | editor, zones |
| POST | `/api/zones/reset/{name}` | zones |
| POST | `/api/zones/reset` | zones |

### `api/streams.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/streams` | polling (2s) |
| POST | `/api/stream/{video}/start` | detection |
| POST | `/api/stream/{video}/stop` | detection, editor |
| POST | `/api/streams/stop` | stopAll |

### `api/videos.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/videos` | loadVideos |
| GET | `/api/videos/{video}/info` | editor, video |
| POST | `/api/videos/upload` | addCameraModal, upload |
| POST | `/api/videos/optimize/{video}` | optimisation |

### `api/counting.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/counting/{video}` | polling |
| POST | `/api/counting/{video}/reset` | counting |
| POST | `/api/counting/{video}/config` | counting |
| POST | `/api/counting/{video}/toggle` | counting |
| POST | `/api/counting/{video}/flip` | counting |
| GET | `/api/counting/params` | counting |
| PUT | `/api/counting/params` | counting |

### `api/presence.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/presence/{video}` | polling, editor (×4) |

### `api/metrics.ts`
| Méthode | Endpoint | Utilisé par |
|---|---|---|
| GET | `/api/metrics` | analytics |
| GET | `/api/logs` | logs |
| DELETE | `/api/logs` | logs |
| GET | `/api/blur` | detection |
| POST | `/api/blur/toggle` | detection |
| GET | `/api/skills` | skills-adapter |

---

## 5. Polling & Timers → `hooks/`

| Timer | Intervalle | Endpoint | Module cible |
|---|---|---|---|
| `updateActiveStreams` | 2 000 ms | `GET /api/streams` | `useStreamsPolling` |
| Zone loading loop | 1 200 ms (actif) / 5 000 ms (idle) | `GET /api/zones/{video}` + `GET /api/presence/{video}` | `useZonesPolling` |
| Log auto-refresh | 2 000 ms | `GET /api/logs` | `useLogsPolling` |
| Editor preview | 280 ms | Frame refresh (local) | `useEditorPreview` |

---

## 6. Événements DOM → Composants React

| Groupe d'événements | Nombre | Composant cible |
|---|---|---|
| Canvas editor (pointer/dblclick) | 8 | `<EditorCanvas />` |
| Boutons éditeur | 16 | `<EditorToolbar />` |
| Modal app (alert/confirm) | 4 | `<AppModal />` |
| Navigation (nav*) | 6 | `<TopBar />` |
| Sites (create, filter) | 3 | `<SitesView />` |
| Caméras (add, detect, RTSP) | 8 | `<AddCameraModal />` |
| Vidéo (select, upload) | 2 | `<VideoSelector />` |
| Dessin (draw, undo, finish) | 12 | `<DrawPanel />` |
| Détection (start, stop, blur) | 5 | `<DetectionControls />` |
| Comptage (mode, zone, toggle) | 7 | `<CountingPanel />` |
| Zones/Bénéfices (click, change) | 6 | `<BenefitsPanel />` |
| Logs (filter, refresh, clear) | 3 | `<LogsView />` |
| Analytics (canvas hover) | 4 | `<AnalyticsCard />` |
| Window resize | 2 | Global (useEffect) |
| Document click (close menus) | 5 | Global (useEffect) |
| Keyboard (shortcuts) | 2 | Global (useEffect) |
| **TOTAL** | **~93** | |

---

## 7. Fichiers externes consommés

| Fichier | Rôle | Migrer vers |
|---|---|---|
| `skills-adapter.js` (88 lignes) | Charge config skills, expose `window.skillsAdapter` | `api/skills.ts` + `stores/skillsStore.ts` |
| `lov-dropdown.js` (237 lignes) | Select custom glassmorphism | Composant React `<LovDropdown />` |
| `utils.js` (59 lignes) | `formatHMS`, `escapeHtml`, `truncateFilename`… | `utils/format.ts` (sans `escapeHtml` — React escape auto) |
| `index.css` (6 681 lignes) | Styles globaux + design tokens | Garder tokens dans `styles/tokens.css`, reste en CSS Modules |
