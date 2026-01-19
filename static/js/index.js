// State
        let currentVideo = null;
        let currentCameraId = null;
        let currentView = 'home'; // 'home' | 'tracker'
        // DEMO mode: sites live only in memory (refresh => reset)
        const DEMO_SITES = [
            {
                name: 'Site Démo',
                cameras: [
                    { id: 'cam1', name: 'Entrepôt', hint: 'Déchargement & présence', video: 'entr1.mp4' },
                    { id: 'cam2', name: 'Convoyeur', hint: 'Tapis roulant & contrôle', video: 'video_01.mp4' }
                ]
            }
        ];
        let sitesCache = structuredClone ? structuredClone(DEMO_SITES) : JSON.parse(JSON.stringify(DEMO_SITES));
        let currentSite = null; // { name, cameras, created_at }
        let availableVideosList = [];
        // Détection: on calcule les % / absences uniquement pendant la détection (pas "depuis que je regarde la page")
        const videoRunAccumSecByVideo = {}; // { [videoName]: seconds } cumulé (hors session courante)
        const videoRunStartTsByVideo = {};  // { [videoName]: epochMs } start de la session courante si en cours
        const videoHasRunByVideo = {};      // { [videoName]: boolean } true dès qu'on a lancé au moins une fois pendant cette session
        const lastPresenceByVideo = {};     // { [videoName]: presenceZones } snapshot gelé quand en pause
        const zoneLiveTimersByVideo = {};  // { [videoName]: { lastTs:number, zones:{[zone]:{occ:number, abs:number}} } }
        const zonesCacheByVideo = {};      // { [videoName]: zonesWithPolygons } (définitions)
        let zonesCacheRefreshTs = 0;
        const linePassByVideo = {};        // { [videoName]: { [zoneName]: { count:number, lastOcc:boolean, series:number[] } } }
        const presencePreviewsCollapsedByVideo = {}; // { [videoName]: { [zoneName]: boolean } }
        const sidebarZonesCollapsedByVideo = {}; // { [videoName]: { [zoneName]: boolean } } pour replier les zones dans la sidebar
        const zonesDefsFetchTsByVideo = {}; // { [videoName]: epochMs } pour throttle /api/zones/{video}
        const zonesDefsFetchedByVideo = {}; // { [videoName]: boolean } pour distinguer "0 zones" vs "pas encore fetch"
        const presenceOkTsByVideo = {};     // { [videoName]: epochMs } dernier /api/presence OK (anti-stale)
        let loadZonesInFlight = false;
        let loadZonesLoopTimer = null;
        const PRESENCE_POLL_ACTIVE_MS = 1000;  // Reduced from 350ms to avoid server saturation
        const PRESENCE_POLL_IDLE_MS = 2000;   // Reduced frequency when idle
        const ZONES_DEF_TTL_MS = 2500;
        const PRESENCE_STALE_MS = 1400; // si on n'a pas eu de /presence OK depuis trop longtemps, ne pas afficher "Occupé"

        function markVideoRunStart(video) {
            if (!video) return;
            if (videoRunAccumSecByVideo[video] == null) videoRunAccumSecByVideo[video] = 0;
            if (!videoRunStartTsByVideo[video]) videoRunStartTsByVideo[video] = Date.now();
            videoHasRunByVideo[video] = true;
        }

        function markVideoRunStop(video) {
            if (!video) return;
            const start = videoRunStartTsByVideo[video];
            if (!start) return;
            const dt = Math.max(0, (Date.now() - start) / 1000);
            videoRunAccumSecByVideo[video] = Number(videoRunAccumSecByVideo[video] || 0) + dt;
            delete videoRunStartTsByVideo[video];
        }

        function markAllRunsStop() {
            try {
                for (const v of Array.from(activeVideoStreams || [])) markVideoRunStop(v);
            } catch {}
        }

        function getVideoRunTotalSec(video) {
            const acc = Number(videoRunAccumSecByVideo[video] || 0);
            const start = videoRunStartTsByVideo[video];
            if (activeVideoStreams.has(video) && start) {
                return acc + Math.max(0, (Date.now() - start) / 1000);
            }
            return acc;
        }

        function formatHMS(totalSeconds) {
            const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const ss = s % 60;
            const pad = (n) => String(n).padStart(2, '0');
            return `${pad(h)}:${pad(m)}:${pad(ss)}`;
        }

        function ensureZoneLive(video) {
            if (!zoneLiveTimersByVideo[video]) zoneLiveTimersByVideo[video] = { lastTs: 0, zones: {} };
            if (!zoneLiveTimersByVideo[video].zones) zoneLiveTimersByVideo[video].zones = {};
            return zoneLiveTimersByVideo[video];
        }

        function updateZoneLiveTimers(video, presenceZones) {
            if (!video) return;
            if (!activeVideoStreams.has(video)) return; // ne compte que pendant la détection
            const v = ensureZoneLive(video);
            const now = Date.now();
            const last = Number(v.lastTs || 0);
            const dt = last ? Math.max(0, (now - last) / 1000) : 0;
            v.lastTs = now;
            if (!dt) return;

            // union des zones connues + zones présentes
            const names = new Set([
                ...Object.keys(v.zones || {}),
                ...Object.keys(presenceZones || {})
            ]);
            for (const name of names) {
                if (!v.zones[name]) v.zones[name] = { occ: 0, abs: 0 };
                const isOcc = !!(presenceZones?.[name]?.is_occupied);
                if (isOcc) v.zones[name].occ += dt;
                else v.zones[name].abs += dt;
            }
        }

        function resetLocalTimersAll() {
            try {
                for (const k of Object.keys(zoneLiveTimersByVideo)) delete zoneLiveTimersByVideo[k];
                for (const k of Object.keys(videoRunAccumSecByVideo)) delete videoRunAccumSecByVideo[k];
                for (const k of Object.keys(videoRunStartTsByVideo)) delete videoRunStartTsByVideo[k];
                for (const k of Object.keys(videoHasRunByVideo)) delete videoHasRunByVideo[k];
                for (const k of Object.keys(lastPresenceByVideo)) delete lastPresenceByVideo[k];
                for (const k of Object.keys(zonesCacheByVideo)) delete zonesCacheByVideo[k];
            } catch {}
        }

        function resetLocalTimersZone(video, zoneName) {
            if (!video || !zoneName) return;
            const v = zoneLiveTimersByVideo[video];
            if (v?.zones?.[zoneName]) v.zones[zoneName] = { occ: 0, abs: 0 };
            // ne touche pas lastTs: le tick continue si détection tourne
            try {
                const lp = linePassByVideo?.[video]?.[zoneName];
                if (lp) { lp.count = 0; lp.series = []; lp.lastOcc = false; }
            } catch {}
        }

        function ensureLinePass(video) {
            if (!linePassByVideo[video]) linePassByVideo[video] = {};
            return linePassByVideo[video];
        }

        function updateLinePass(video, presenceZones, isLineZoneFn) {
            if (!video || !presenceZones) return;
            const store = ensureLinePass(video);
            for (const [zoneName, info] of Object.entries(presenceZones || {})) {
                if (!isLineZoneFn(zoneName)) continue;
                if (!store[zoneName]) store[zoneName] = { count: 0, lastOcc: false, series: [] };
                const cur = !!info?.is_occupied;
                if (cur && !store[zoneName].lastOcc) store[zoneName].count += 1;
                store[zoneName].lastOcc = cur;
                store[zoneName].series.push(cur ? 1 : 0);
                if (store[zoneName].series.length > 80) store[zoneName].series.splice(0, store[zoneName].series.length - 80);
            }
        }

        function buildSparklineSvg(series) {
            const s = Array.isArray(series) ? series : [];
            const n = s.length;
            const w = 120, h = 26;
            if (n < 2) {
                return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><line class="base" x1="0" y1="${h-4}" x2="${w}" y2="${h-4}" /></svg>`;
            }
            const padY = 4;
            const xStep = w / (n - 1);
            const pts = s.map((v, i) => {
                const x = i * xStep;
                const y = (h - padY) - (v ? 14 : 0);
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(' ');
            return `
                <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
                    <line class="base" x1="0" y1="${h-4}" x2="${w}" y2="${h-4}" />
                    <polyline class="wave" points="${pts}" />
                </svg>
            `;
        }

        async function refreshZonesCacheForSite(force = false) {
            if (!currentSite?.cameras?.length) return;
            const now = Date.now();
            if (!force && (now - zonesCacheRefreshTs) < 2000) return; // Throttle to 2s to reduce server load
            zonesCacheRefreshTs = now;
            // Only refresh for current video, not all cameras (reduces requests)
            if (currentVideo) {
                try {
                    const res = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}`);
                    const data = await res.json();
                    zonesCacheByVideo[currentVideo] = data.zones || {};
                } catch {}
            }
        }

        function computeSiteStats() {
            const cams = (currentSite?.cameras || []);
            const camCount = cams.length;

            // total zones sur toutes les caméras (définitions)
            let zoneCount = 0;
            const videos = cams.map(c => c.video).filter(Boolean);
            for (const v of videos) {
                zoneCount += Object.keys(zonesCacheByVideo[v] || {}).length;
            }

            // taux d'occupation moyen pondéré par le temps de détection:
            // moyenne = (Σ occSec) / (Σ (occSec+absSec)) sur toutes les zones/toutes les caméras
            let totalOcc = 0;
            let totalDen = 0;
            for (const v of videos) {
                const z = zoneLiveTimersByVideo?.[v]?.zones || {};
                // Important: ne calculer QUE sur les zones existantes (définitions),
                // sinon une zone supprimée peut continuer à polluer la moyenne via les timers locaux.
                const names = Object.keys(zonesCacheByVideo[v] || {});
                for (const name of names) {
                    const occ = Number(z?.[name]?.occ || 0);
                    const abs = Number(z?.[name]?.abs || 0);
                    const den = occ + abs;
                    totalOcc += occ;
                    totalDen += den;
                }
            }
            const avgOcc = totalDen > 0 ? (totalOcc / totalDen) : 0;
            return { camCount, zoneCount, avgOcc };
        }

        function updateHeaderStepsKpis() {
            if (!step1Num || !step2Num || !step3Num) return;
            if (currentView !== 'tracker' || !currentSite) {
                step1Num.textContent = '—';
                step2Num.textContent = '—';
                step3Num.textContent = '—';
                return;
            }
            const { camCount, zoneCount, avgOcc } = computeSiteStats();
            step1Num.textContent = String(camCount);
            step2Num.textContent = String(zoneCount);
            step3Num.textContent = `${Math.round(avgOcc * 100)}%`;
        }
        let isDrawing = false;
        let isCurrentVideoStreaming = false;
        let drawPoints = [];
        let videoWidth = 0;
        let videoHeight = 0;
        let activeVideoStreams = new Set();
        let drawMode = 'poly'; // 'poly' | 'line'
        let activeDrawZoneName = '';
        let zonePolygonCounts = {}; // { [zoneName]: number }
        let selectedAsset = null; // { zone: string, idx?: number }
        let editMode = false;
        let editPoints = null; // points being edited (array of [x,y])
        let editDragging = null; // { idx: number }
        const HANDLE_RADIUS = 10;

        // Elements
        const videoSelect = document.getElementById('videoSelect');
        const videoUpload = document.getElementById('videoUpload');
        const videoFrame = document.getElementById('videoFrame');
        const videoStream = document.getElementById('videoStream');
        const videoContainer = document.getElementById('videoContainer');
        const drawCanvas = document.getElementById('drawCanvas');
        const ctx = drawCanvas.getContext('2d');
        const placeholder = document.getElementById('placeholder');
        const toggleDrawPanelBtn = document.getElementById('toggleDrawPanelBtn');
        const drawFab = document.getElementById('drawFab');
        const editZonesBtn = document.getElementById('editZonesBtn');
        const drawPanel = document.getElementById('drawPanel');
        const drawZoneSelect = document.getElementById('drawZoneSelect');
        const drawZoneNameGroup = document.getElementById('drawZoneNameGroup');
        const drawZoneName = document.getElementById('drawZoneName');
        const toolPolyBtn = document.getElementById('toolPolyBtn');
        const toolLineBtn = document.getElementById('toolLineBtn');
        const startDrawBtn = document.getElementById('startDrawBtn');
        const stopDrawBtn = document.getElementById('stopDrawBtn');
        const editSelectedBtn = document.getElementById('editSelectedBtn');
        const addPointBtn = document.getElementById('addPointBtn');
        const deletePointBtn = document.getElementById('deletePointBtn');
        const saveEditBtn = document.getElementById('saveEditBtn');
        const drawHud = document.getElementById('drawHud');
        const drawHudTitle = document.getElementById('drawHudTitle');
        const undoBtn = document.getElementById('undoBtn');
        const finishBtn = document.getElementById('finishBtn');
        const cancelBtn = document.getElementById('cancelBtn');
        const drawInstructions = document.getElementById('drawInstructions');
        const startDetectionBtn = document.getElementById('startDetectionBtn');
        const stopAllBtn = document.getElementById('stopAllBtn');
        const zonesGrid = document.getElementById('zonesGrid');
        const statusBadge = document.getElementById('statusBadge');
        const statusText = document.getElementById('statusText');
        const step1Num = document.querySelector('#step1 .step-num');
        const step2Num = document.querySelector('#step2 .step-num');
        const step3Num = document.querySelector('#step3 .step-num');
        const activeStreamsDiv = document.getElementById('activeStreams');
        const cameraGrid = document.getElementById('cameraGrid');
        const currentVideoTitle = document.getElementById('currentVideoTitle');
        const zoneListSidebar = document.getElementById('zoneListSidebar');
        const sidebarTreeLabel = document.getElementById('sidebarTreeLabel');
        const recapCameras = document.getElementById('recapCameras');
        const recapCamerasSub = document.getElementById('recapCamerasSub');
        const recapZones = document.getElementById('recapZones');
        const recapZonesSub = document.getElementById('recapZonesSub');
        const recapDrawings = document.getElementById('recapDrawings');
        const recapDrawingsSub = document.getElementById('recapDrawingsSub');
        const recapActive = document.getElementById('recapActive');
        const recapActiveSub = document.getElementById('recapActiveSub');

        // Multi-site UI
        const homeView = document.getElementById('homeView');
        const trackerView = document.getElementById('trackerView');
        const sitesGrid = document.getElementById('sitesGrid');
        const newSiteNameInput = document.getElementById('newSiteName');
        const createSiteBtn = document.getElementById('createSiteBtn');
        const navSites = document.getElementById('navSites');
        const navTracker = document.getElementById('navTracker');
        const pageTitleEl = document.getElementById('pageTitle');
        const pageSubtitleEl = document.getElementById('pageSubtitle');
        const stepsEl = document.querySelector('.steps');

        // Cameras (per site)
        const addCameraBtn = document.getElementById('addCameraBtn');
        const addCameraForm = document.getElementById('addCameraForm');
        const newCamName = document.getElementById('newCamName');
        const newCamVideo = document.getElementById('newCamVideo');
        const newCamHint = document.getElementById('newCamHint');
        const saveCamBtn = document.getElementById('saveCamBtn');
        const cancelCamBtn = document.getElementById('cancelCamBtn');

        // Camera source tabs and panels
        const camSourceTabs = document.querySelectorAll('.cam-source-tab');
        const camSourcePanels = document.querySelectorAll('.cam-source-panel');
        let currentCamSourceType = 'video'; // 'video' | 'webcam' | 'rtsp'

        // Webcam source elements
        const newCamWebcam = document.getElementById('newCamWebcam');
        const detectWebcamsBtn = document.getElementById('detectWebcamsBtn');

        // RTSP source elements
        const newCamRtspUrl = document.getElementById('newCamRtspUrl');
        const testRtspBtn = document.getElementById('testRtspBtn');
        const scanOnvifBtn = document.getElementById('scanOnvifBtn');
        const onvifScanStatus = document.getElementById('onvifScanStatus');

        // Backend cameras (webcam/rtsp) - loaded from /api/cameras
        let backendCameras = {};

        async function loadBackendCameras() {
            try {
                const res = await fetch('/api/cameras');
                const data = await res.json();
                backendCameras = data.cameras || {};
                // Sync backend cameras to a special "Caméras" site
                syncBackendCamerasToSite();
            } catch (e) {
                console.error('Failed to load backend cameras:', e);
                backendCameras = {};
            }
        }

        function syncBackendCamerasToSite() {
            // Create or update a "Caméras" site with backend cameras (webcam/rtsp)
            const CAMERAS_SITE_NAME = 'Caméras';
            let camerasSite = sitesCache.find(s => s.name === CAMERAS_SITE_NAME);

            if (!camerasSite) {
                camerasSite = { name: CAMERAS_SITE_NAME, cameras: [] };
                sitesCache.push(camerasSite);
            }

            // Build cameras array from backend cameras
            const syncedCameras = [];
            for (const [backendId, camData] of Object.entries(backendCameras)) {
                // Check if this camera already exists in the site
                const existing = camerasSite.cameras.find(c => c.backendCameraId === backendId);
                if (existing) {
                    syncedCameras.push(existing);
                } else {
                    // Create new camera entry
                    const camType = camData.type; // 'webcam' or 'rtsp'
                    syncedCameras.push({
                        id: backendId,
                        name: camData.name || backendId,
                        hint: camType === 'webcam' ? 'Webcam' : 'RTSP',
                        sourceType: camType,
                        backendCameraId: backendId
                    });
                }
            }

            camerasSite.cameras = syncedCameras;

            // Also sync cameras across all sites - if a camera in any site matches a backend camera ID pattern,
            // ensure it has the correct sourceType and backendCameraId
            for (const site of sitesCache) {
                if (site.name === CAMERAS_SITE_NAME) continue;
                for (const cam of site.cameras || []) {
                    // Check if this camera's ID matches a backend camera
                    if (backendCameras[cam.id]) {
                        const backendCam = backendCameras[cam.id];
                        cam.backendCameraId = cam.id;
                        cam.sourceType = backendCam.type;
                    }
                    // Or if backendCameraId was set but we need to verify it still exists
                    if (cam.backendCameraId && backendCameras[cam.backendCameraId]) {
                        const backendCam = backendCameras[cam.backendCameraId];
                        cam.sourceType = backendCam.type;
                    }
                }
            }
        }

        async function addBackendCamera(cameraId, name, type, deviceIdOrUrl) {
            const body = { camera_id: cameraId, name, type };
            if (type === 'webcam') {
                body.device_id = parseInt(deviceIdOrUrl, 10);
            } else if (type === 'rtsp') {
                body.url = deviceIdOrUrl;
            }
            const res = await fetch('/api/cameras', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to add camera');
            }
            await loadBackendCameras();
        }

        // Upload video elements
        const uploadVideoInput = document.getElementById('uploadVideoInput');
        const uploadVideoLabelText = document.getElementById('uploadVideoLabelText');
        const uploadVideoForm = document.getElementById('uploadVideoForm');
        const uploadProgress = document.getElementById('uploadProgress');
        console.log('[Init] Upload elements:', { uploadVideoInput, uploadVideoLabelText, uploadVideoForm, uploadProgress });

        // Editor modal elements
        const editorOverlay = document.getElementById('editorOverlay');
        const editorCloseBtn = document.getElementById('editorCloseBtn');
        const editorCloseBtn2 = document.getElementById('editorCloseBtn2');
        const editorSaveBtn = document.getElementById('editorSaveBtn');
        const editorTitle = document.getElementById('editorTitle');
        const editorSubtitle = document.getElementById('editorSubtitle');
        const editorFrame = document.getElementById('editorFrame');
        const editorCanvas = document.getElementById('editorCanvas');
        const editorCtx = editorCanvas.getContext('2d');
        const editorCamId = document.getElementById('editorCamId');
        const editorCamSource = document.getElementById('editorCamSource');
        const editorCamRes = document.getElementById('editorCamRes');
        const editorCamStatus = document.getElementById('editorCamStatus');
        const editorZoneList = document.getElementById('editorZoneList');
        const editorNewZoneName = document.getElementById('editorNewZoneName');
        const editorAddZoneBtn = document.getElementById('editorAddZoneBtn');
        const editorDeleteZoneBtn = document.getElementById('editorDeleteZoneBtn');
        const toolSelectBtn = document.getElementById('toolSelectBtn');
        const toolCountLineBtn = document.getElementById('toolCountLineBtn');
        const toolIncludeBtn = document.getElementById('toolIncludeBtn');
        const toolExcludeBtn = document.getElementById('toolExcludeBtn');
        const toolUndoBtn = document.getElementById('toolUndoBtn');
        const toolClearBtn = document.getElementById('toolClearBtn');
        const toolSaveBtn = document.getElementById('toolSaveBtn');
        const editorGuide = document.getElementById('editorGuide');
        const editorHoverBar = document.getElementById('editorHoverBar');
        const hoverDeletePointBtn = document.getElementById('hoverDeletePointBtn');
        const hoverDeleteShapeBtn = document.getElementById('hoverDeleteShapeBtn');
        const editorToolsEl = editorOverlay?.querySelector?.('.editor-tools') || null;

        const editorToolbarButtons = [
            toolSelectBtn,
            toolCountLineBtn,
            toolIncludeBtn,
            toolExcludeBtn,
            toolUndoBtn,
            toolClearBtn,
            toolSaveBtn,
        ].filter(Boolean);

        function editorUpdateToolbarEnabled() {
            const zoneName = editorState.zone;
            const zones = editorState.zones || {};
            const enabled = !!zoneName && !!zones[zoneName];
            editorToolsEl?.classList.toggle('is-disabled', !enabled);
            editorToolbarButtons.forEach((b) => { try { b.disabled = !enabled; } catch {} });
        }

        // App modal (internal alert/confirm)
        const appModalOverlay = document.getElementById('appModalOverlay');
        const appModalTitle = document.getElementById('appModalTitle');
        const appModalSubtitle = document.getElementById('appModalSubtitle');
        const appModalMessage = document.getElementById('appModalMessage');
        const appModalOkBtn = document.getElementById('appModalOkBtn');
        const appModalCancelBtn = document.getElementById('appModalCancelBtn');
        const appModalCloseBtn = document.getElementById('appModalCloseBtn');

        let __modalResolve = null;

        function uiModal({ title = 'Message', subtitle = 'Zone Tracker', message = '', okText = 'OK', cancelText = null } = {}) {
            if (!appModalOverlay) return Promise.resolve(true);
            appModalTitle.textContent = title;
            appModalSubtitle.textContent = subtitle;
            appModalMessage.textContent = message;
            appModalOkBtn.textContent = okText;
            appModalCancelBtn.textContent = cancelText || '';
            appModalCancelBtn.classList.toggle('hidden', !cancelText);
            appModalOverlay.classList.remove('hidden');

            return new Promise((resolve) => {
                __modalResolve = resolve;
            });
        }

        function uiAlert(message, title = 'Info') {
            return uiModal({ title, message, okText: 'OK', cancelText: null });
        }

        function uiConfirm(message, title = 'Confirmer') {
            return uiModal({ title, message, okText: 'Confirmer', cancelText: 'Annuler' });
        }

        const editorState = {
            open: false,
            tool: 'select', // select | line | poly
            zone: null,
            polygonIdx: null,
            mode: 'idle', // idle | creating | editing
            points: [],
            lineDirEnd: null, // [x,y] canvas coords: extrémité de la flèche (brouillon ligne)
            drag: null, // { kind: 'vertex'|'poly', vIdx?:number, start?:[x,y] }
            undo: [],
            zones: {}, // from /api/zones/{video}
            presence: {}, // from /api/presence/{video}
            w: 0,
            h: 0,
            dirty: false,
            autosaveTimer: null,
            didDrag: false,
            lastSavedTs: 0
        };

        function setTool(tool) {
            drawMode = tool;
            toolPolyBtn.classList.toggle('active', tool === 'poly');
            toolLineBtn.classList.toggle('active', tool === 'line');
            updateFinishButtonState();
        }

        // ===== Editor modal logic (paint-like) =====
        function editorSetTool(tool) {
            editorState.tool = tool;
            toolSelectBtn.classList.toggle('active', tool === 'select');
            toolCountLineBtn.classList.toggle('active', tool === 'line');
            toolIncludeBtn.classList.toggle('active', tool === 'include');
            toolExcludeBtn.classList.toggle('active', tool === 'exclude');
            editorState.lineDirEnd = null;
            editorGuide.textContent =
                tool === 'select'
                    ? "Sélection: cliquez un point (hit zone large) puis glissez pour déplacer. Shift + clic près d'une arête = ajouter un point."
                    : tool === 'line'
                        ? "Ligne de comptage: 2 clics pour placer départ/arrivée. Ensuite tirez la flèche depuis le centre pour le sens. Puis Sauver."
                        : tool === 'exclude'
                            ? "Zone d'exclusion: cliquez pour placer des points (3+), puis Sauver."
                            : "Zone d'inclusion: cliquez pour placer des points (3+), puis Sauver.";
        }

        function editorPushUndo() {
            const z = editorState.zone;
            const snapshot = {
                tool: editorState.tool,
                zone: z,
                polygonIdx: editorState.polygonIdx,
                mode: editorState.mode,
                points: clonePoints(editorState.points),
                // IMPORTANT: inclure l'état des polygones de la zone pour que ↶ annule aussi
                // un ajout de point (Shift + clic) ou tout changement de forme.
                zonePolys: z ? clonePointsArray(editorState.zones?.[z]?.polygons || []) : null,
            };
            editorState.undo.push(snapshot);
            if (editorState.undo.length > 50) editorState.undo.shift();
        }

        function editorPopUndo() {
            const s = editorState.undo.pop();
            if (!s) return;
            editorState.tool = s.tool;
            editorState.zone = s.zone;
            editorState.polygonIdx = s.polygonIdx;
            editorState.mode = s.mode;
            editorState.points = clonePoints(s.points);
            if (s.zone && s.zonePolys && editorState.zones?.[s.zone]) {
                editorState.zones[s.zone].polygons = clonePointsArray(s.zonePolys);
            }
            editorSetTool(editorState.tool);
            editorRender();
            // Garder backend en cohérence quand on annule une modification de forme
            // (ex: ajout point via Shift). La direction de ligne (meta localStorage) ne déclenche pas de PUT.
            try { editorScheduleAutosave(); } catch {}
        }

        function editorClearTemp() {
            editorState.mode = 'idle';
            editorState.points = [];
            editorState.drag = null;
            editorState.polygonIdx = null;
            editorState.lineDirEnd = null;
            editorRender();
        }

        function editorScheduleAutosave() {
            if (!currentVideo || !editorState.zone) return;
            editorState.dirty = true;
            if (editorState.autosaveTimer) clearTimeout(editorState.autosaveTimer);
            editorState.autosaveTimer = setTimeout(async () => {
                try {
                    await editorPutNow();
                } catch (e) {
                    console.warn('Autosave failed', e);
                }
            }, 650);
        }

        async function editorPutNow() {
            if (!currentVideo || !editorState.zone) return;
            const zoneName = editorState.zone;
            const polygons = editorState.zones?.[zoneName]?.polygons || [];
            const putRes = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}/${encodeURIComponent(zoneName)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ polygons })
            });
            if (!putRes.ok) throw new Error('PUT zone failed');
            editorState.dirty = false;
            editorState.lastSavedTs = Date.now();
            // sync main view quietly (force pour refléter immédiatement les changements)
            try { await refreshMainAfterEditor(true); } catch {}
        }

        async function editorPostNewPolygon(type, poly, extraMeta = null) {
            if (!currentVideo || !editorState.zone) return;
            const zoneName = editorState.zone;
            const prevCount = (editorState.zones?.[zoneName]?.polygons || []).length;
            const res = await fetch('/api/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: zoneName, polygons: [poly], video: currentVideo })
            });
            if (!res.ok) throw new Error('POST zone failed');
            setDrawType(currentVideo, zoneName, prevCount, type);
            if (type === 'line' && extraMeta) {
                setLineMeta(currentVideo, zoneName, prevCount, extraMeta);
            }
            // refresh editor state from server
            const [zonesRes, presenceRes] = await Promise.all([
                fetch(`/api/zones/${encodeURIComponent(currentVideo)}`),
                fetch(`/api/presence/${encodeURIComponent(currentVideo)}`)
            ]);
            const zdata = await zonesRes.json();
            const pdata = await presenceRes.json();
            editorState.zones = zdata.zones || {};
            editorState.presence = pdata.zones || {};
            // UX: après création, on sélectionne automatiquement la nouvelle forme pour pouvoir l'ajuster
            editorSetTool('select');
            editorState.zone = zoneName;
            editorState.polygonIdx = prevCount;
            editorRenderZoneList();
            editorRender();
            try { await refreshMainAfterEditor(true); } catch {}
        }

        async function refreshMainAfterEditor(force = false) {
            // Soft refresh (sans recharger la page) pour éviter de perdre l'état multi-site (in-memory)
            if (!currentVideo) return;
            // Force invalidation defs (très important après sauvegarde/suppression)
            if (force) {
                try {
                    zonesDefsFetchedByVideo[currentVideo] = false;
                    zonesDefsFetchTsByVideo[currentVideo] = 0;
                    zonesCacheRefreshTs = 0;
                } catch {}
            }

            // refresh KPI / sidebar
            await refreshZonesCacheForSite(true);
            await loadZones();

            // IMPORTANT: s'assurer que l'UI principale repasse bien en "frame + overlay" si la détection est stoppée en éditant
            if (!isCurrentVideoStreaming) {
                videoStream.src = '';
                videoStream.classList.add('hidden');
                videoFrame.classList.remove('hidden');
                drawCanvas.classList.remove('hidden');

                // refresh frame (cache-bust) + redraw overlay *après* chargement image
                videoFrame.onload = async () => {
                    syncCanvasSize();
                    try { await drawExistingZones(true); } catch {}
                };
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
            } else {
                // si stream: on ne redessine pas l'overlay (canvas masqué), mais on garde l'UI à jour
                try { await drawExistingZones(false); } catch {}
            }
        }

        function editorSelectZone(zoneName) {
            editorState.zone = zoneName;
            editorState.polygonIdx = null;
            editorState.mode = 'idle';
            editorState.points = [];
            editorRenderZoneList();
            editorUpdateToolbarEnabled();
            editorRender();
        }

        function editorRenderZoneList() {
            const zones = editorState.zones || {};
            const presence = editorState.presence || {};
            const keys = Object.keys(zones).sort((a,b) => a.localeCompare(b));
            editorZoneList.innerHTML = keys.length
                ? keys.map((z) => {
                    const info = presence[z] || { formatted_time: '00:00:00' };
                    const active = editorState.zone === z ? 'active' : '';
                    const count = (zones[z]?.polygons || []).length;
                    return `
                        <div class="editor-zone-row ${active}" data-editor-select-zone="${z.replace(/'/g, "\\'")}">
                            <div class="editor-zone-left">
                                <span class="editor-dot"></span>
                                <span class="editor-zone-name">${z}</span>
                            </div>
                            <span class="editor-zone-meta">${info.formatted_time} • ${count}</span>
                        </div>
                    `;
                }).join('')
                : `<div style="color: rgba(245,247,255,0.65); font-size: var(--text-sm);">Aucune zone</div>`;
            editorUpdateToolbarEnabled();
        }

        window.__editorSelectZone = (z) => editorSelectZone(z);

        async function editorDeleteZone() {
            if (!currentVideo || !editorState.zone) {
                uiAlert('Sélectionnez une zone à supprimer.', 'Zones');
                return;
            }
            const zoneName = editorState.zone;
            const ok = await uiConfirm(`Supprimer la zone "${zoneName}" (tous ses dessins) ?`, 'Suppression');
            if (!ok) return;

            const res = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}/${encodeURIComponent(zoneName)}`, { method: 'DELETE' });
            if (!res.ok) {
                uiAlert('Erreur suppression zone.', 'Suppression');
                return;
            }

            // Refresh editor + main UI
            editorState.zone = null;
            editorState.polygonIdx = null;
            hideHoverBar();
            await editorOpen();
            try { await loadZones(); await drawExistingZones(); } catch {}
        }

        async function editorOpen() {
            if (!currentVideo) {
                uiAlert('Choisissez une caméra avant de dessiner.', 'Dessin');
                return;
            }
            if (isCurrentVideoStreaming) {
                // On stoppe pour éditer sur frame fixe (plus fiable)
                await fetch(`/api/stream/${encodeURIComponent(currentVideo)}/stop`, { method: 'POST' });
                isCurrentVideoStreaming = false;
            }

            const cam = getCameraByVideo(currentVideo);
            editorTitle.textContent = `Video Editing — ${cam ? cam.name : currentVideo}`;
            editorSubtitle.textContent = `Source: ${currentVideo}`;

            // Load info + zones
            const [infoRes, zonesRes, presenceRes] = await Promise.all([
                fetch(`/api/videos/${encodeURIComponent(currentVideo)}/info`),
                fetch(`/api/zones/${encodeURIComponent(currentVideo)}`),
                fetch(`/api/presence/${encodeURIComponent(currentVideo)}`)
            ]);
            const info = await infoRes.json();
            const zdata = await zonesRes.json();
            const pdata = await presenceRes.json();

            editorState.w = info.width;
            editorState.h = info.height;
            editorState.zones = zdata.zones || {};
            editorState.presence = pdata.zones || {};
            if (!editorState.zone) {
                const firstZone = Object.keys(editorState.zones)[0] || null;
                editorState.zone = firstZone;
            }

            editorCamId.textContent = cam ? cam.id : '—';
            editorCamSource.textContent = currentVideo;
            editorCamRes.textContent = `${info.width}×${info.height}`;
            editorCamStatus.textContent = 'Online';

            editorFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
            editorCanvas.width = info.width;
            editorCanvas.height = info.height;

            editorSetTool('select');
            editorClearTemp();
            editorRenderZoneList();
            editorUpdateToolbarEnabled();

            editorOverlay.classList.remove('hidden');
            editorState.open = true;
            // sync canvas display size after image load
            editorFrame.onload = () => {
                const r = editorFrame.getBoundingClientRect();
                editorCanvas.style.width = r.width + 'px';
                editorCanvas.style.height = r.height + 'px';
                editorRender();
            };
        }

        async function editorClose() {
            // Auto-save before closing: commit any draft shape and save all
            try {
                // If there's a shape being drawn, commit it first
                if (editorState.points && editorState.points.length >= 2) {
                    await editorCommitDraftShape();
                }
                // Save all changes
                if (currentVideo && editorState.zone) {
                    await editorPutNow();
                }
            } catch (e) {
                console.warn('Auto-save on close failed:', e);
            }

            editorOverlay.classList.add('hidden');
            editorState.open = false;
            editorClearTemp();
            // Important: quand on quitte l'éditeur, on force un refresh visuel (évite l'impression que les anciennes formes "persistaient")
            refreshMainAfterEditor(true).catch(() => {});
        }

        function editorRender() {
            // clear
            editorCtx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);

            // draw existing drawings for all zones (selected zone full, others ghosted)
            const selectedZone = editorState.zone;
            const zonesObj = editorState.zones || {};
            const zoneKeys = Object.keys(zonesObj).sort((a, b) => a.localeCompare(b));
            zoneKeys.forEach((zoneName) => {
                const polys = zonesObj?.[zoneName]?.polygons || [];
                const isGhost = !!(selectedZone && zoneName !== selectedZone);
                const prevAlpha = editorCtx.globalAlpha;
                editorCtx.globalAlpha = isGhost ? 0.22 : 1;

                polys.forEach((poly, idx) => {
                    if (!poly || poly.length < 3) return;
                    const isActive = (!isGhost) && (editorState.polygonIdx === idx);
                    const type = getDrawType(currentVideo, zoneName, idx);
                    const c = colorsForType(type, isActive);

                    if (type === 'line' && poly.length === 4) {
                        // Render backend quadrilateral as a center line (editor view)
                        const a = [(poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2];
                        const b = [(poly[2][0] + poly[3][0]) / 2, (poly[2][1] + poly[3][1]) / 2];
                        editorCtx.beginPath();
                        editorCtx.moveTo(a[0], a[1]);
                        editorCtx.lineTo(b[0], b[1]);
                        editorCtx.strokeStyle = c.stroke;
                        editorCtx.lineWidth = isActive ? 4 : (isGhost ? 1.6 : 2);
                        editorCtx.setLineDash([]);
                        editorCtx.stroke();

                        // Arrow: show on active line with handle; show ghost arrow without handle
                        const meta = getLineMeta(currentVideo, zoneName, idx);
                        const arrow = computeLineArrowFromMeta(meta, poly);
                        if (arrow?.mid && arrow?.end) {
                            drawArrow(
                                editorCtx,
                                arrow.mid,
                                arrow.end,
                                c.stroke,
                                { shaftWidth: isGhost ? 1.4 : 2, dashed: false, head: 22, wing: 13, outline: true, handle: !isGhost }
                            );
                        }
                    } else {
                        editorCtx.beginPath();
                        editorCtx.moveTo(poly[0][0], poly[0][1]);
                        for (let i = 1; i < poly.length; i++) editorCtx.lineTo(poly[i][0], poly[i][1]);
                        editorCtx.closePath();
                        editorCtx.fillStyle = c.fill;
                        editorCtx.fill();
                        editorCtx.strokeStyle = c.stroke;
                        editorCtx.lineWidth = isActive ? 4 : (isGhost ? 1.6 : 2);
                        editorCtx.stroke();
                    }
                });

                editorCtx.globalAlpha = prevAlpha;
            });

            // draw current temp tool
            if (editorState.points.length) {
                if (editorState.tool === 'line') {
                    const p1 = editorState.points[0];
                    const p2 = editorState.points[1];
                    if (p1) {
                        editorCtx.beginPath();
                        editorCtx.arc(p1[0], p1[1], 8, 0, Math.PI * 2);
                        editorCtx.fillStyle = '#10B0F9';
                        editorCtx.fill();
                    }
                    if (p2) {
                        editorCtx.beginPath();
                        editorCtx.arc(p2[0], p2[1], 8, 0, Math.PI * 2);
                        editorCtx.fillStyle = '#10B0F9';
                        editorCtx.fill();
                        editorCtx.beginPath();
                        editorCtx.moveTo(p1[0], p1[1]);
                        editorCtx.lineTo(p2[0], p2[1]);
                        editorCtx.strokeStyle = '#10B0F9';
                        editorCtx.lineWidth = 2; /* plus fin */
                        editorCtx.setLineDash([]); /* continu */
                        editorCtx.stroke();
                        editorCtx.setLineDash([]);
                    }
                    // Flèche de direction depuis le centre (seulement quand on a 2 points)
                    if (p1 && p2) {
                        const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
                        const def = getLineDraftMidAndDefaultDir();
                        const end = editorState.lineDirEnd || def?.end;
                        if (end) drawArrow(editorCtx, mid, end, '#10B0F9', { shaftWidth: 2, dashed: false, head: 22, wing: 13, outline: true });
                    }
                } else if (editorState.tool === 'include' || editorState.tool === 'exclude') {
                    const pts = editorState.points;
                    const t = editorState.tool === 'exclude' ? 'exclude' : 'include';
                    const c = colorsForType(t, false);
                    editorCtx.beginPath();
                    editorCtx.moveTo(pts[0][0], pts[0][1]);
                    for (let i = 1; i < pts.length; i++) editorCtx.lineTo(pts[i][0], pts[i][1]);
                    if (pts.length >= 3) {
                        editorCtx.closePath();
                        editorCtx.fillStyle = c.fill;
                        editorCtx.fill();
                    }
                    editorCtx.strokeStyle = c.stroke;
                    editorCtx.lineWidth = 3;
                    editorCtx.setLineDash([6, 4]);
                    editorCtx.stroke();
                    editorCtx.setLineDash([]);

                    pts.forEach((p, i) => {
                        editorCtx.beginPath();
                        editorCtx.arc(p[0], p[1], 8, 0, Math.PI * 2);
                        editorCtx.fillStyle = i === 0 ? '#10B0F9' : '#22c55e';
                        editorCtx.fill();
                        editorCtx.strokeStyle = '#000';
                        editorCtx.lineWidth = 2;
                        editorCtx.stroke();
                    });
                }
            }

            // draw handles for selected polygon (edit)
            if (editorState.tool === 'select' && editorState.zone && typeof editorState.polygonIdx === 'number') {
                const poly = editorState.zones?.[editorState.zone]?.polygons?.[editorState.polygonIdx];
                if (poly && poly.length) {
                    poly.forEach((p, i) => {
                        editorCtx.beginPath();
                        editorCtx.arc(p[0], p[1], 9, 0, Math.PI * 2);
                        editorCtx.fillStyle = '#10B0F9';
                        editorCtx.fill();
                        editorCtx.strokeStyle = '#fff';
                        editorCtx.lineWidth = 2;
                        editorCtx.stroke();
                    });
                }
            }
        }

        let hoverHideTimer = null;
        let hoverKey = '';

        function cancelHoverHide() {
            if (hoverHideTimer) clearTimeout(hoverHideTimer);
            hoverHideTimer = null;
        }

        function scheduleHoverHide(ms = 7000) {
            cancelHoverHide();
            // Ne pas auto-masquer si déjà caché
            if (editorHoverBar.classList.contains('hidden')) return;
            hoverHideTimer = setTimeout(() => {
                hideHoverBar();
            }, ms);
        }

        function hideHoverBar() {
            cancelHoverHide();
            editorHoverBar.classList.add('hidden');
            editorHoverBar.dataset.kind = '';
            editorHoverBar.dataset.vIdx = '';
            hoverKey = '';
        }

        function showHoverBar(xCss, yCss, kind, vIdx, key, avoidX = xCss, avoidY = yCss) {
            // Le hoverbar ne doit JAMAIS obstruer le point (sinon impossible de le déplacer),
            // y compris quand on est près des bords (clamp).
            editorHoverBar.classList.remove('hidden');
            editorHoverBar.dataset.kind = kind;
            editorHoverBar.dataset.vIdx = typeof vIdx === 'number' ? String(vIdx) : '';
            hoverDeletePointBtn.classList.toggle('hidden', kind !== 'vertex');
            hoverKey = key || '';

            // Mesure réelle (selon si le bouton "point" est visible ou non)
            const barW = editorHoverBar.offsetWidth || 92;
            const barH = editorHoverBar.offsetHeight || 46;

            // Bounds: canvas wrap (coordonnées CSS)
            const wrapRect = editorCanvas.getBoundingClientRect();
            const maxLeft = Math.max(0, wrapRect.width - barW);
            const maxTop = Math.max(0, wrapRect.height - barH);

            // Zone interdite autour du point (évite le recouvrement)
            const avoid = 44; // px CSS (généreux)
            const ax1 = avoidX - avoid, ay1 = avoidY - avoid;
            const ax2 = avoidX + avoid, ay2 = avoidY + avoid;

            function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
            function noOverlap(l, t) {
                const r = l + barW, b = t + barH;
                // pas d'intersection avec le carré d'évitement
                return (r < ax1) || (l > ax2) || (b < ay1) || (t > ay2);
            }

            function overlapArea(l, t) {
                const r = l + barW, b = t + barH;
                const ox = Math.max(0, Math.min(r, ax2) - Math.max(l, ax1));
                const oy = Math.max(0, Math.min(b, ay2) - Math.max(t, ay1));
                return ox * oy;
            }

            // Essaye plusieurs positions autour du point (8 directions), puis on choisit le moindre chevauchement.
            const off = 12;
            const candidates = [
                { l: avoidX + off, t: avoidY - barH - off },                 // NE
                { l: avoidX + off, t: avoidY + off },                        // SE
                { l: avoidX - barW - off, t: avoidY - barH - off },          // NW
                { l: avoidX - barW - off, t: avoidY + off },                 // SW
                { l: avoidX - barW / 2, t: avoidY - barH - off },            // N
                { l: avoidX - barW / 2, t: avoidY + off },                   // S
                { l: avoidX + off, t: avoidY - barH / 2 },                   // E
                { l: avoidX - barW - off, t: avoidY - barH / 2 },            // W
            ];

            let placed = null;
            let best = null;
            for (const c of candidates) {
                const l = clamp(c.l, 0, maxLeft);
                const t = clamp(c.t, 0, maxTop);
                if (noOverlap(l, t)) { placed = { l, t }; break; }
                const a = overlapArea(l, t);
                if (!best || a < best.a) best = { l, t, a };
            }
            // Fallback: position la moins mauvaise (chevauchement minimal) plutôt que de recouvrir le point
            if (!placed && best) placed = { l: best.l, t: best.t };
            if (!placed) placed = { l: clamp(avoidX + off, 0, maxLeft), t: clamp(avoidY + off, 0, maxTop) };

            editorHoverBar.style.left = `${placed.l}px`;
            editorHoverBar.style.top = `${placed.t}px`;
            // Persistance généreuse (7s) pour permettre le déplacement de la souris + clic
            scheduleHoverHide(7000);
        }

        function editorEventPoint(e) {
            const rect = editorCanvas.getBoundingClientRect();
            const scaleX = editorCanvas.width / rect.width;
            const scaleY = editorCanvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            return [x, y];
        }

        function pointInPoly(poly, p) {
            // ray casting
            let inside = false;
            for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
                const xi = poly[i][0], yi = poly[i][1];
                const xj = poly[j][0], yj = poly[j][1];
                const intersect = ((yi > p[1]) !== (yj > p[1])) &&
                    (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-9) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        function editorPickPolygon(p) {
            const zoneName = editorState.zone;
            const polys = editorState.zones?.[zoneName]?.polygons || [];
            for (let idx = polys.length - 1; idx >= 0; idx--) {
                const poly = polys[idx];
                if (poly?.length >= 3 && pointInPoly(poly, p)) return idx;
            }
            return null;
        }

        function editorPickTarget(p, radiusPx = 34) {
            // Objectif UX: pouvoir attraper un point même si le clic est *en dehors* de la forme.
            // Priorité:
            // 1) hitbox autour des sommets / endpoints (ligne)
            // 2) poignée de flèche (ligne)
            // 3) clic sur segment (ligne) pour déplacer la ligne
            // 4) point-in-poly (fallback)
            const zoneName = editorState.zone;
            if (!zoneName) return null;
            const polys = editorState.zones?.[zoneName]?.polygons || [];
            const r2 = radiusPx * radiusPx;

            let best = null; // { idx, kind, vIdx?, point?, d2? }

            // 1/2/3) Proximité (sommets, endpoints, poignée flèche, segment)
            for (let idx = polys.length - 1; idx >= 0; idx--) {
                const poly = polys[idx];
                if (!poly || poly.length < 3) continue;
                const type = getDrawType(currentVideo, zoneName, idx);

                if (type === 'line') {
                    const meta = getLineMeta(currentVideo, zoneName, idx);
                    const arrow = computeLineArrowFromMeta(meta, poly);
                    if (arrow?.end && lineHandleHit(p, arrow.end)) {
                        // poignée = priorité max (permet de la saisir même hors quad)
                        return { idx, kind: 'lineHandle', point: arrow.end, mid: arrow.mid };
                    }

                    const ends = editorLineEndpointsFromPoly(poly);
                    const a = (meta?.p1 && Array.isArray(meta.p1)) ? meta.p1 : ends?.a;
                    const b = (meta?.p2 && Array.isArray(meta.p2)) ? meta.p2 : ends?.b;
                    if (a && b) {
                        const dA = dist2(a, p);
                        const dB = dist2(b, p);
                        if (dA <= r2 || dB <= r2) {
                            const which = dA <= dB ? 0 : 1;
                            const d2 = Math.min(dA, dB);
                            if (!best || d2 < best.d2) best = { idx, kind: 'lineEndpoint', vIdx: which, point: which === 0 ? a : b, d2 };
                        } else {
                            // hit sur segment (tolérance plus large) pour déplacer la ligne entière
                            const dSeg2 = editorPointToSegmentDist2(p, a, b);
                            if (dSeg2 <= (22 * 22)) {
                                // priorité plus faible qu'un endpoint, mais permet de sélectionner/drag même hors forme
                                if (!best || dSeg2 < best.d2) best = { idx, kind: 'lineSegment', point: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], d2: dSeg2 };
                            }
                        }
                    }
                    continue;
                }

                // Polygone: hitbox autour des sommets
                const vIdx = editorNearestVertex(poly, p, radiusPx);
                if (vIdx >= 0) {
                    const d2 = dist2(poly[vIdx], p);
                    if (!best || d2 < best.d2) best = { idx, kind: 'vertex', vIdx, point: poly[vIdx], d2 };
                    continue;
                }

                // Polygone: hitbox autour des arêtes (permet de sélectionner / Shift+ajout même si le clic est à l'extérieur)
                const edgeD2 = editorNearestEdgeDist2(poly, p);
                const edgeHitR = 22; // px canvas (tolérance visuelle)
                if (edgeD2 <= edgeHitR * edgeHitR) {
                    if (!best || edgeD2 < best.d2) best = { idx, kind: 'edge', d2: edgeD2 };
                }
            }
            if (best) return best;

            // 4) Fallback: point dans le polygone
            for (let idx = polys.length - 1; idx >= 0; idx--) {
                const poly = polys[idx];
                if (poly?.length >= 3 && pointInPoly(poly, p)) return { idx, kind: 'poly' };
            }
            return null;
        }

        function editorNearestVertex(poly, p, radius = 32) {
            let best = -1;
            let bestD = Infinity;
            for (let i = 0; i < poly.length; i++) {
                const d = dist2(poly[i], p);
                if (d < bestD) { bestD = d; best = i; }
            }
            return bestD <= radius * radius ? best : -1;
        }

        function editorLineEndpointsFromPoly(poly) {
            // Le backend stocke une ligne comme un quadrilatère fin (4 pts).
            // On veut 2 poignées seulement = milieux des 2 bords.
            if (!poly || poly.length !== 4) return null;
            const a = [(poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2];
            const b = [(poly[2][0] + poly[3][0]) / 2, (poly[2][1] + poly[3][1]) / 2];
            return { a, b };
        }

        function editorPointToSegmentDist2(p, a, b) {
            const vx = b[0] - a[0];
            const vy = b[1] - a[1];
            const wx = p[0] - a[0];
            const wy = p[1] - a[1];
            const c1 = vx * wx + vy * wy;
            if (c1 <= 0) return dist2(p, a);
            const c2 = vx * vx + vy * vy;
            if (c2 <= c1) return dist2(p, b);
            const t = c1 / (c2 || 1);
            const proj = [a[0] + t * vx, a[1] + t * vy];
            return dist2(p, proj);
        }

        function editorNearestEdgeDist2(poly, p) {
            if (!poly || poly.length < 3) return Infinity;
            let best = Infinity;
            for (let i = 0; i < poly.length; i++) {
                const a = poly[i];
                const b = poly[(i + 1) % poly.length];
                const d2 = editorPointToSegmentDist2(p, a, b);
                if (d2 < best) best = d2;
            }
            return best;
        }

        function editorUpdateHoverUI(e) {
            if (!editorState.open) return;
            if (editorState.tool !== 'select') {
                hideHoverBar();
                return;
            }
            const zoneName = editorState.zone;
            if (!zoneName) { scheduleHoverHide(7000); return; }
            const p = editorEventPoint(e);
            const target = editorPickTarget(p, 36);
            if (!target) { scheduleHoverHide(7000); return; }

            // Prefer hovered polygon as selected (lightweight)
            editorState.polygonIdx = target.idx;
            const poly = editorState.zones?.[zoneName]?.polygons?.[target.idx];
            if (!poly) { scheduleHoverHide(7000); return; }
            const vIdx = (target.kind === 'vertex') ? target.vIdx : -1;

            const wrapRect = editorCanvas.getBoundingClientRect();
            // Hoverbar:
            // - vertex => bouton "supprimer point" (sauf lignes)
            // - poly/line* => uniquement "supprimer forme"
            const kind = vIdx >= 0 ? 'vertex' : 'poly';
            const key = `${target.idx}:${kind === 'vertex' ? `v${vIdx}` : (target.kind || 'p')}`;

            // IMPORTANT UX:
            // - Le hoverbar doit rester "ancré" (pas suivre la souris) pour pouvoir cliquer dessus.
            // - On ne repositionne que si la cible change.
            if (key !== hoverKey) {
                let xCss = 0;
                let yCss = 0;

                if (kind === 'vertex') {
                    // Anchor sur le sommet (hitbox invisible), même si clic hors polygone
                    const vx = poly[vIdx][0];
                    const vy = poly[vIdx][1];
                    const scaleX = wrapRect.width / editorCanvas.width;
                    const scaleY = wrapRect.height / editorCanvas.height;
                    // Important: éviter le vrai point (pas un point décalé), sinon au bord le clamp peut le recouvrir.
                    const px = vx * scaleX;
                    const py = vy * scaleY;
                    xCss = px;
                    yCss = py;
                    // avoidX/avoidY = centre du point
                    showHoverBar(xCss, yCss, kind, kind === 'vertex' ? vIdx : null, key, px, py);
                    // showHoverBar déjà appelé, on sort
                    editorRender();
                    return;
                } else if (target.point) {
                    // Anchor sur point “spécial” (ligne: poignée flèche / endpoint / segment)
                    const vx = target.point[0];
                    const vy = target.point[1];
                    const scaleX = wrapRect.width / editorCanvas.width;
                    const scaleY = wrapRect.height / editorCanvas.height;
                    const px = vx * scaleX;
                    const py = vy * scaleY;
                    xCss = px;
                    yCss = py;
                    showHoverBar(xCss, yCss, 'poly', null, key, px, py);
                    editorRender();
                    return;
                } else {
                    // Anchor near initial hover position (cursor at the time), but do not track afterwards
                    xCss = (e.clientX - wrapRect.left) + 10;
                    yCss = (e.clientY - wrapRect.top) + 10;
                }

                // showHoverBar gère clamp + évitement du point, donc on lui passe le point d'ancrage brut
                showHoverBar(
                    xCss,
                    yCss,
                    kind,
                    kind === 'vertex' ? vIdx : null,
                    key
                );
            } else {
                // Keep alive while hovering same target
                scheduleHoverHide(7000);
            }
            editorRender();
        }

        editorCanvas.addEventListener('dblclick', (e) => {
            if (!editorState.open) return;
            // IMPORTANT UX: pas d'auto-save au double-clic (trop surprenant).
            // Le flow fiable: dessiner → cliquer "Sauver".
            e.preventDefault();
        });

        editorCanvas.addEventListener('pointerdown', (e) => {
            if (!editorState.open) return;

            const p = editorEventPoint(e);

            if (editorState.tool === 'line') {
                // Ligne de comptage: 2 points strict.
                // - si déjà 2 points: clic sur la poignée flèche => drag direction, sinon redémarre une nouvelle ligne
                if (editorState.points.length === 2) {
                    const p1 = editorState.points[0];
                    const p2 = editorState.points[1];
                    const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
                    const def = getLineDraftMidAndDefaultDir();
                    const end = editorState.lineDirEnd || def?.end;
                    if (end && lineHandleHit(p, end)) {
                        editorState.drag = { kind: 'lineDir' };
                        editorState.didDrag = false;
                        editorCanvas.setPointerCapture(e.pointerId);
                        editorRender();
                        return;
                    }
                    // redémarre
                    editorPushUndo();
                    editorState.points = [p];
                    editorState.lineDirEnd = null;
                    editorRender();
                    return;
                }
                editorPushUndo();
                editorState.points.push(p);
                // dès qu'on a 2 points, initialise une direction par défaut
                if (editorState.points.length === 2) {
                    const def = getLineDraftMidAndDefaultDir();
                    if (def?.end) editorState.lineDirEnd = def.end;
                }
                editorRender();
                return;
            }

            if (editorState.tool === 'include' || editorState.tool === 'exclude') {
                editorPushUndo();
                editorState.points.push(p);
                editorRender();
                return;
            }

            // select/edit
            const zoneName = editorState.zone;
            if (!zoneName) return;
            const target = editorPickTarget(p, 36);
            if (!target) {
                editorState.polygonIdx = null;
                hideHoverBar();
                editorRender();
                return;
            }
            const idx = target.idx;
            editorState.polygonIdx = idx;
            hideHoverBar();

            const poly = editorState.zones[zoneName].polygons[idx];
            // Ligne de comptage sélectionnée: drag la flèche de direction (sans toucher aux points)
            if (getDrawType(currentVideo, zoneName, idx) === 'line') {
                // 1) flèche (vecteur) : fait partie du même objet
                const meta = getLineMeta(currentVideo, zoneName, idx);
                const arrow = computeLineArrowFromMeta(meta, poly);
                if (arrow?.end && lineHandleHit(p, arrow.end)) {
                    editorPushUndo();
                    editorState.drag = { kind: 'lineDirSaved', zoneName, idx, mid: arrow.mid };
                    editorState.didDrag = false;
                    editorCanvas.setPointerCapture(e.pointerId);
                    editorRender();
                    return;
                }

                // 2) endpoints + déplacement global (pas de coins du quad)
                const ends = editorLineEndpointsFromPoly(poly);
                const a = meta?.p1 && Array.isArray(meta.p1) ? meta.p1 : (ends?.a || null);
                const b = meta?.p2 && Array.isArray(meta.p2) ? meta.p2 : (ends?.b || null);
                if (a && b) {
                    const r = 34;
                    const hitA = dist2(a, p) <= r * r;
                    const hitB = dist2(b, p) <= r * r;
                    if (hitA || hitB) {
                        editorPushUndo();
                        editorState.drag = { kind: 'lineEnd', zoneName, idx, which: hitA ? 'a' : 'b', a: [...a], b: [...b] };
                        editorState.didDrag = false;
                        editorCanvas.setPointerCapture(e.pointerId);
                        editorRender();
                        return;
                    }
                    // hit sur le segment => move whole line
                    const d2 = editorPointToSegmentDist2(p, a, b);
                    if (d2 <= (22 * 22)) {
                        editorPushUndo();
                        editorState.drag = { kind: 'lineMove', zoneName, idx, start: p, a: [...a], b: [...b] };
                        editorState.didDrag = false;
                        editorCanvas.setPointerCapture(e.pointerId);
                        editorRender();
                        return;
                    }
                }
            }
            const vIdx = editorNearestVertex(poly, p, 34);
            if (vIdx >= 0) {
                editorPushUndo();
                editorState.drag = { kind: 'vertex', vIdx };
                editorState.didDrag = false;
                editorCanvas.setPointerCapture(e.pointerId);
            } else {
                // insertion de point sur arête: volontaire (Shift) pour éviter les insertions accidentelles
                const copy = clonePointsArray(editorState.zones[zoneName].polygons);
                const polyCopy = copy[idx];
                const inserted = e.shiftKey ? insertPointOnNearestEdge(polyCopy, p) : false;
                if (inserted) {
                    editorPushUndo();
                    editorState.zones[zoneName].polygons = copy;
                    editorScheduleAutosave();
                } else {
                    editorPushUndo();
                    editorState.drag = { kind: 'poly', start: p };
                    editorState.didDrag = false;
                    editorCanvas.setPointerCapture(e.pointerId);
                }
            }
            editorRender();
        });

        editorCanvas.addEventListener('pointermove', (e) => {
            if (!editorState.open) return;
            if (!editorState.drag) {
                editorUpdateHoverUI(e);
                return;
            }
            if (editorState.drag.kind === 'lineDir' && editorState.tool === 'line' && editorState.points.length === 2) {
                const p = editorEventPoint(e);
                const p1 = editorState.points[0];
                const p2 = editorState.points[1];
                const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
                const v = sub(p, mid);
                // évite une flèche trop courte
                if (Math.hypot(v[0], v[1]) > 8) {
                    editorState.lineDirEnd = p;
                    editorState.didDrag = true;
                }
                editorRender();
                return;
            }
            if (editorState.drag.kind === 'lineDirSaved') {
                const p = editorEventPoint(e);
                const zoneName = editorState.drag.zoneName;
                const idx = editorState.drag.idx;
                const mid = editorState.drag.mid;
                if (!zoneName || typeof idx !== 'number' || !mid) return;
                const v = sub(p, mid);
                if (Math.hypot(v[0], v[1]) > 8) {
                    const d = norm(v);
                    const meta = getLineMeta(currentVideo, zoneName, idx) || {};
                    // conserve p1/p2 si déjà présents
                    meta.dir = d;
                    setLineMeta(currentVideo, zoneName, idx, meta);
                    editorState.didDrag = true;
                }
                editorRender();
                return;
            }
            if (editorState.drag.kind === 'lineMove' || editorState.drag.kind === 'lineEnd') {
                const p = editorEventPoint(e);
                const zoneName = editorState.drag.zoneName;
                const idx = editorState.drag.idx;
                if (!zoneName || typeof idx !== 'number') return;

                let a = editorState.drag.a;
                let b = editorState.drag.b;
                if (!a || !b) return;

                if (editorState.drag.kind === 'lineMove') {
                    const start = editorState.drag.start;
                    if (!start) return;
                    const dx = p[0] - start[0];
                    const dy = p[1] - start[1];
                    a = [a[0] + dx, a[1] + dy];
                    b = [b[0] + dx, b[1] + dy];
                    editorState.drag.start = p;
                } else {
                    const which = editorState.drag.which;
                    if (which === 'a') a = [p[0], p[1]];
                    else b = [p[0], p[1]];
                }

                editorState.drag.a = a;
                editorState.drag.b = b;
                // Rebuild a clean quad (no distortion)
                const polyNew = lineToPolygon(a, b, 12);
                if (editorState.zones?.[zoneName]?.polygons?.[idx]) {
                    editorState.zones[zoneName].polygons[idx] = polyNew;
                }
                // Update meta endpoints (dir kept)
                const meta = getLineMeta(currentVideo, zoneName, idx) || {};
                meta.p1 = a;
                meta.p2 = b;
                setLineMeta(currentVideo, zoneName, idx, meta);

                editorState.didDrag = true;
                editorRender();
                return;
            }
            const zoneName = editorState.zone;
            if (!zoneName || typeof editorState.polygonIdx !== 'number') return;
            const p = editorEventPoint(e);
            // Perf: ne pas cloner à chaque move (ça rendait le drag "mou").
            const poly = editorState.zones?.[zoneName]?.polygons?.[editorState.polygonIdx];
            if (!poly) return;

            if (editorState.drag.kind === 'vertex') {
                const vIdx = editorState.drag.vIdx;
                if (typeof vIdx !== 'number' || !poly[vIdx]) return;
                poly[vIdx] = [p[0], p[1]];
                editorState.didDrag = true;
            } else if (editorState.drag.kind === 'poly') {
                const start = editorState.drag.start;
                if (!start) return;
                const dx = p[0] - start[0];
                const dy = p[1] - start[1];
                for (let i = 0; i < poly.length; i++) {
                    poly[i] = [poly[i][0] + dx, poly[i][1] + dy];
                }
                editorState.drag.start = p;
                editorState.didDrag = true;
            }
            editorRender();
        });

        editorCanvas.addEventListener('pointerup', () => {
            if (!editorState.open) return;
            const dragKind = editorState.drag?.kind;
            editorState.drag = null;
            if (editorState.didDrag) {
                editorState.didDrag = false;
                // Ne pas autosave (PUT) quand on ne fait que tirer la flèche direction (meta localStorage).
                if (dragKind !== 'lineDir' && dragKind !== 'lineDirSaved') {
                    editorScheduleAutosave();
                }
            }
        });

        async function editorCommitDraftShape() {
            // Toolbar "Sauvegarder": confirme la forme actuellement tracée (draft -> forme)
            if (!currentVideo) {
                uiAlert('Choisissez une caméra avant de dessiner.', 'Sauvegarde');
                return;
            }
            // UX: si aucune zone n'est sélectionnée mais qu'il en existe, on en choisit une automatiquement.
            if (!editorState.zone) {
                const keys = Object.keys(editorState.zones || {}).sort();
                if (keys.length >= 1) {
                    editorSelectZone(keys[0]);
                } else {
                    uiAlert('Créez d\'abord une zone à droite (bouton +), puis cliquez sur \"Sauvegarder\".', 'Sauvegarde');
                    try { editorNewZoneName?.focus(); } catch {}
                    return;
                }
            }

            // UX demandé: en mode "Sélection" (ajustement d'une forme existante), pas de modal.
            // On fait un "save normal" (PUT) uniquement si quelque chose a changé.
            if (editorState.tool === 'select') {
                if (!editorState.dirty) return; // rien à faire
                try {
                    await editorPutNow(); // silencieux (pas de modal de succès)
                } catch (e) {
                    uiAlert('Erreur: sauvegarde (PUT).', 'Sauvegarde');
                }
                return;
            }

            // Commit seulement si un tracé est en cours
            if (editorState.tool === 'line' && editorState.points.length === 2) {
                const p1 = editorState.points[0];
                const p2 = editorState.points[1];
                const poly = lineToPolygon(p1, p2, 12);
                const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
                const end = editorState.lineDirEnd || getLineDraftMidAndDefaultDir()?.end || add(mid, [0, -60]);
                const d = norm(sub(end, mid));
                const meta = { p1, p2, dir: d };
                editorState.points = [];
                editorState.lineDirEnd = null;
                editorRender();
                await editorPostNewPolygon('line', poly, meta);
                return;
            }

            if ((editorState.tool === 'include' || editorState.tool === 'exclude') && editorState.points.length >= 3) {
                const poly = clonePoints(editorState.points);
                const type = editorState.tool === 'exclude' ? 'exclude' : 'include';
                editorState.points = [];
                editorRender();
                await editorPostNewPolygon(type, poly);
                return;
            }

            // Ne pas pop de modal si l'utilisateur n'est pas en train de tracer.
            // (le bouton sert surtout à valider un tracé en cours)
            return;
        }

        async function editorSaveAll() {
            // Bouton "Sauvegarder tout" (bas): sauvegarde la configuration entière (PUT)
            if (!currentVideo) {
                uiAlert('Choisissez une caméra avant de sauvegarder.', 'Sauvegarde');
                return;
            }
            if (!editorState.zone) {
                const keys = Object.keys(editorState.zones || {}).sort();
                if (keys.length >= 1) editorSelectZone(keys[0]);
            }
            try {
                await editorPutNow();
                const [zonesRes, presenceRes] = await Promise.all([
                    fetch(`/api/zones/${encodeURIComponent(currentVideo)}`),
                    fetch(`/api/presence/${encodeURIComponent(currentVideo)}`)
                ]);
                const zdata = await zonesRes.json();
                const pdata = await presenceRes.json();
                editorState.zones = zdata.zones || {};
                editorState.presence = pdata.zones || {};
                editorRenderZoneList();
                editorRender();
                await refreshMainAfterEditor(true);
            } catch (e) {
                uiAlert('Erreur: sauvegarde (PUT).', 'Sauvegarde');
            }
        }

        editorAddZoneBtn.addEventListener('click', async () => {
            const name = editorNewZoneName.value.trim();
            if (!name) return;
            const res = await fetch('/api/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, polygons: [], video: currentVideo })
            });
            if (!res.ok) { uiAlert('Erreur création zone', 'Zones'); return; }
            editorNewZoneName.value = '';
            await editorOpen(); // refresh editor state
            editorSelectZone(name);
        });

        editorDeleteZoneBtn?.addEventListener('click', editorDeleteZone);
        
        // Gestionnaire de délégation pour les zones dans l'éditeur
        editorZoneList?.addEventListener('click', (e) => {
            const zoneRow = e.target?.closest?.('[data-editor-select-zone]');
            if (zoneRow) {
                e.preventDefault();
                e.stopPropagation();
                const zoneName = zoneRow.getAttribute('data-editor-select-zone') || '';
                if (zoneName && window.__editorSelectZone) {
                    window.__editorSelectZone(zoneName);
                }
                return;
            }
        });

        toolSelectBtn.addEventListener('click', () => editorSetTool('select'));
        toolCountLineBtn.addEventListener('click', () => { editorSetTool('line'); editorState.points = []; editorRender(); });
        toolIncludeBtn.addEventListener('click', () => { editorSetTool('include'); editorState.points = []; editorRender(); });
        toolExcludeBtn.addEventListener('click', () => { editorSetTool('exclude'); editorState.points = []; editorRender(); });
        toolUndoBtn.addEventListener('click', () => {
            // Annulation "intelligente" façon paint:
            // - si on trace, on retire le dernier point
            // - sinon on annule le dernier snapshot
            if ((editorState.tool === 'include' || editorState.tool === 'exclude' || editorState.tool === 'line') && editorState.points.length) {
                editorState.points.pop();
                editorRender();
                return;
            }
            editorPopUndo();
        });
        toolClearBtn.addEventListener('click', () => { editorPushUndo(); editorClearTemp(); });
        toolSaveBtn?.addEventListener('click', editorCommitDraftShape);

        editorSaveBtn?.addEventListener('click', editorSaveAll);
        editorCloseBtn.addEventListener('click', editorClose);
        editorCloseBtn2?.addEventListener('click', editorClose);

        // Hoverbar: keep it clickable (pause auto-hide while hovering it)
        editorHoverBar?.addEventListener('pointerenter', cancelHoverHide);
        editorHoverBar?.addEventListener('pointerleave', () => scheduleHoverHide(7000));

        // Hover actions
        hoverDeletePointBtn.addEventListener('click', () => {
            if (!editorState.zone || typeof editorState.polygonIdx !== 'number') return;
            const kind = editorHoverBar.dataset.kind;
            const vIdx = Number(editorHoverBar.dataset.vIdx);
            if (kind !== 'vertex' || !Number.isFinite(vIdx)) return;
            const copy = clonePointsArray(editorState.zones?.[editorState.zone]?.polygons || []);
            const poly = copy[editorState.polygonIdx];
            if (!poly || poly.length <= 3) {
                uiAlert('Un polygone doit garder au moins 3 points.', 'Edition');
                return;
            }
            editorPushUndo();
            poly.splice(vIdx, 1);
            editorState.zones[editorState.zone].polygons = copy;
            editorScheduleAutosave();
            hideHoverBar();
            editorRender();
        });

        hoverDeleteShapeBtn.addEventListener('click', async () => {
            if (!editorState.zone || typeof editorState.polygonIdx !== 'number') return;
            const copy = clonePointsArray(editorState.zones?.[editorState.zone]?.polygons || []);
            if (!copy[editorState.polygonIdx]) return;
            const ok = await uiConfirm('Supprimer ce dessin ?', 'Suppression');
            if (!ok) return;
            editorPushUndo();
            copy.splice(editorState.polygonIdx, 1);
            editorState.zones[editorState.zone].polygons = copy;
            editorState.polygonIdx = null;
            editorScheduleAutosave();
            hideHoverBar();
            editorRenderZoneList();
            editorRender();
        });

        // Raccourcis clavier (paint-like)
        document.addEventListener('keydown', (e) => {
            if (!editorState.open) return;
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
                e.preventDefault();
                toolUndoBtn.click();
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                editorClose();
            }
        });

        // Modal buttons
        function closeAppModal(result) {
            if (!appModalOverlay) return;
            appModalOverlay.classList.add('hidden');
            const r = __modalResolve;
            __modalResolve = null;
            if (typeof r === 'function') r(result);
        }
        appModalOkBtn?.addEventListener('click', () => closeAppModal(true));
        appModalCancelBtn?.addEventListener('click', () => closeAppModal(false));
        appModalCloseBtn?.addEventListener('click', () => closeAppModal(false));
        appModalOverlay?.addEventListener('click', (e) => {
            if (e.target === appModalOverlay) closeAppModal(false);
        });
        document.addEventListener('keydown', (e) => {
            if (appModalOverlay?.classList.contains('hidden')) return;
            if (e.key === 'Escape') closeAppModal(false);
        });

        // Open editor when clicking the drawing button (now: overlay icon)
        toggleDrawPanelBtn?.addEventListener('click', () => editorOpen());
        drawFab?.addEventListener('click', () => editorOpen());
        editZonesBtn?.addEventListener('click', () => editorOpen());

        function clonePoints(pts) {
            return (pts || []).map(p => [p[0], p[1]]);
        }

        function dist2(a, b) {
            const dx = a[0] - b[0];
            const dy = a[1] - b[1];
            return dx * dx + dy * dy;
        }

        function norm(v) {
            const l = Math.hypot(v[0], v[1]) || 1;
            return [v[0] / l, v[1] / l];
        }

        function add(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
        function sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
        function mul(a, k) { return [a[0] * k, a[1] * k]; }

        function drawArrow(ctx, from, to, color = '#10B0F9', opts = {}) {
            const v = sub(to, from);
            const u = norm(v);
            const head = Number(opts.head ?? 22);
            const wing = Number(opts.wing ?? 13);
            const shaftWidth = Number(opts.shaftWidth ?? 2);
            const dashed = !!opts.dashed;
            const outline = opts.outline !== false; // default true
            const handle = opts.handle !== false;  // default true
            // shaft
            ctx.beginPath();
            ctx.moveTo(from[0], from[1]);
            ctx.lineTo(to[0], to[1]);
            ctx.strokeStyle = color;
            ctx.lineWidth = shaftWidth;
            ctx.setLineDash(dashed ? [6, 4] : []);
            ctx.stroke();
            ctx.setLineDash([]);

            // head
            const left = add(to, add(mul(u, -head), mul([-u[1], u[0]], wing)));
            const right = add(to, add(mul(u, -head), mul([u[1], -u[0]], wing)));
            ctx.beginPath();
            ctx.moveTo(to[0], to[1]);
            ctx.lineTo(left[0], left[1]);
            ctx.lineTo(right[0], right[1]);
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
            if (outline) {
                ctx.strokeStyle = 'rgba(255,255,255,0.9)';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // handle
            if (handle) {
                ctx.beginPath();
                ctx.arc(to[0], to[1], 7, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffff';
                ctx.globalAlpha = 0.9;
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        function getLineDraftMidAndDefaultDir() {
            if (editorState.tool !== 'line') return null;
            if (editorState.points.length !== 2) return null;
            const p1 = editorState.points[0];
            const p2 = editorState.points[1];
            const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
            const v = sub(p2, p1);
            const u = norm(v);
            // default direction = perp to line
            const perp = [-u[1], u[0]];
            const end = add(mid, mul(perp, 60));
            return { mid, end };
        }

        function lineHandleHit(p, handleEnd) {
            return dist2(p, handleEnd) <= (12 * 12);
        }

        function getCanvasPointFromEvent(e) {
            const rect = drawCanvas.getBoundingClientRect();
            const scaleX = drawCanvas.width / rect.width;
            const scaleY = drawCanvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            return [x, y];
        }

        function nearestVertexIndex(pts, p, radiusPx = HANDLE_RADIUS) {
            const r2 = radiusPx * radiusPx;
            let bestIdx = -1;
            let bestD = Infinity;
            for (let i = 0; i < pts.length; i++) {
                const d = dist2(pts[i], p);
                if (d < bestD) {
                    bestD = d;
                    bestIdx = i;
                }
            }
            return bestD <= r2 ? bestIdx : -1;
        }

        function pointToSegmentDistanceSquared(p, a, b) {
            const vx = b[0] - a[0];
            const vy = b[1] - a[1];
            const wx = p[0] - a[0];
            const wy = p[1] - a[1];
            const c1 = vx * wx + vy * wy;
            if (c1 <= 0) return dist2(p, a);
            const c2 = vx * vx + vy * vy;
            if (c2 <= c1) return dist2(p, b);
            const t = c1 / c2;
            const proj = [a[0] + t * vx, a[1] + t * vy];
            return dist2(p, proj);
        }

        function insertPointOnNearestEdge(pts, p) {
            if (!pts || pts.length < 3) return false;
            let bestIdx = -1;
            let bestD = Infinity;
            for (let i = 0; i < pts.length; i++) {
                const a = pts[i];
                const b = pts[(i + 1) % pts.length];
                const d = pointToSegmentDistanceSquared(p, a, b);
                if (d < bestD) {
                    bestD = d;
                    bestIdx = i;
                }
            }
            // Seuil d'insertion (en px canvas)
            if (bestD > (18 * 18)) return false;
            pts.splice(bestIdx + 1, 0, [p[0], p[1]]);
            return true;
        }

        function setEditMode(on) {
            editMode = on;
            editDragging = null;
        }

        async function saveEditedPolygon() {
            if (!currentVideo || !selectedAsset || !selectedAsset.zone) return;
            if (!editPoints || !Array.isArray(editPoints)) return;

            const zoneName = selectedAsset.zone;
            const idx = selectedAsset.idx;
            const polygons = clonePointsArray((cachedZones[zoneName]?.polygons) || []);
            if (typeof idx !== 'number' || !polygons[idx]) return;

            polygons[idx] = clonePoints(editPoints);

            const res = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}/${encodeURIComponent(zoneName)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ polygons })
            });
            if (!res.ok) {
                uiAlert('Erreur sauvegarde (PUT zone).', 'Zones');
                return;
            }
            await loadZones();
            await drawExistingZones();
            setEditMode(false);
        }

        function clonePointsArray(polys) {
            return (polys || []).map(poly => clonePoints(poly));
        }
        const steps = {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3')
        };

        function setView(view) {
            currentView = view;
            const isHome = view === 'home';

            homeView.classList.toggle('hidden', !isHome);
            trackerView.classList.toggle('hidden', isHome);

            navSites.classList.toggle('active', isHome);
            navTracker.classList.toggle('active', !isHome);

            if (stepsEl) stepsEl.style.display = isHome ? 'none' : '';

            if (isHome) {
                pageTitleEl.textContent = 'Sites';
                pageSubtitleEl.textContent = 'Choisissez un site ou créez-en un pour démarrer la configuration';
                renderSitesSidebar();
            } else {
                const siteLabel = currentSite?.name ? ` — ${currentSite.name}` : '';
                pageTitleEl.textContent = `Zone Presence Tracker${siteLabel}`;
                pageSubtitleEl.textContent = 'Surveillance et analyse du temps de présence';
            }
            updateSidebarTreeLabel();
        }

        function updateSidebarTreeLabel() {
            if (!sidebarTreeLabel) return;
            if (currentView === 'tracker') {
                sidebarTreeLabel.textContent = currentSite?.name ? String(currentSite.name) : 'Site';
            } else {
                sidebarTreeLabel.textContent = 'Sites';
            }
        }

        function loadSites() {
            renderSitesHome();
            if (currentView === 'home') renderSitesSidebar();
        }

        function escapeHtml(text) {
            return String(text ?? '')
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        // Helper to truncate filename for display
        function truncateFilename(name, maxLen = 20) {
            if (!name || name.length <= maxLen) return name;
            const ext = name.lastIndexOf('.') > 0 ? name.slice(name.lastIndexOf('.')) : '';
            const base = name.slice(0, name.length - ext.length);
            const availableLen = maxLen - ext.length - 3; // 3 for "..."
            if (availableLen <= 0) return name.slice(0, maxLen - 3) + '...';
            return base.slice(0, availableLen) + '...' + ext;
        }

        function renderSitesHome() {
            if (!sitesGrid) return;
            if (!sitesCache || sitesCache.length === 0) {
                sitesGrid.innerHTML = '<div class="no-zones">Aucun site. Créez-en un pour commencer.</div>';
                return;
            }

            sitesGrid.innerHTML = sitesCache.map((s) => {
                const cams = s.cameras || [];
                const camLines = cams.map(c => `${c.id}: ${c.name} • ${c.video}`).slice(0, 3);
                const siteName = escapeHtml(s.name);
                const siteKey = encodeURIComponent(String(s.name ?? ''));
                const demoName = DEMO_SITES?.[0]?.name;
                const canDelete = !(demoName && s.name === demoName);
                return `
                    <div class="zone-card site-card" style="cursor:pointer;" data-site="${siteKey}">
                        <div class="zone-card-header">
                            <div class="zone-card-name">${siteName}</div>
                            <div style="display:flex; align-items:center; gap: var(--space-2);">
                                <div class="zone-card-status empty">${cams.length} caméra(s)</div>
                                ${canDelete ? `<button class="btn btn-ghost btn-icon" data-delete-site="${siteKey}" title="Supprimer" type="button">✕</button>` : ''}
                            </div>
                        </div>
                        <div style="margin-top: var(--space-2); color: var(--color-text-muted); font-size: var(--text-xs);">
                            ${camLines.map(l => `<div>${l}</div>`).join('')}
                        </div>
                        <div class="zone-card-time" style="font-size: var(--text-sm);">Configurer</div>
                    </div>
                `;
            }).join('');
        }

        function renderSitesSidebar() {
            if (!zoneListSidebar) return;
            if (!sitesCache || sitesCache.length === 0) {
                zoneListSidebar.innerHTML = '<div style="color: var(--sidebar-text-subtle); font-size: var(--text-sm);">Aucun site</div>';
                return;
            }
            zoneListSidebar.innerHTML = `
                <div class="draw-tree">
                    ${sitesCache.map((s) => `
                        <div class="tree-row site-row" data-site="${encodeURIComponent(String(s.name ?? ''))}">
                            <div class="left">
                                <span class="label">🏢 ${escapeHtml(s.name)}</span>
                            </div>
                            <span class="meta">${(s.cameras || []).length} cam</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        function createSite(name) {
            const n = String(name || '').trim();
            if (!n) throw new Error('Nom de site requis');
            if ((sitesCache || []).some(s => s.name === n)) throw new Error('Site déjà existant');
            // nouveau site vide (0 caméra)
            sitesCache.push({ name: n, cameras: [] });
        }

        function cameraIdFromName(name) {
            const base = String(name || '')
                .trim()
                .toLowerCase()
                .replaceAll(/[^a-z0-9]+/g, '-')
                .replaceAll(/(^-|-$)/g, '');
            return base || 'cam';
        }

        function makeUniqueCameraId(cameras, desiredId) {
            const used = new Set((cameras || []).map(c => c.id));
            let id = desiredId;
            let n = 2;
            while (used.has(id)) {
                id = `${desiredId}-${n}`;
                n += 1;
            }
            return id;
        }

        function saveCurrentSiteCameras(cameras) {
            if (!currentSite?.name) throw new Error('Aucun site sélectionné');
            const idx = (sitesCache || []).findIndex(s => s.name === currentSite.name);
            if (idx < 0) throw new Error('Site introuvable');
            sitesCache[idx].cameras = cameras;
            currentSite = sitesCache[idx];
        }

        function deleteSiteByName(name) {
            const n = String(name || '').trim();
            const demoName = DEMO_SITES?.[0]?.name;
            if (n && demoName && n === demoName) throw new Error('Impossible de supprimer le site démo');
            const idx = (sitesCache || []).findIndex(s => s.name === n);
            if (idx >= 0) sitesCache.splice(idx, 1);
            if (currentSite?.name === n) {
                currentSite = null;
                currentVideo = null;
                currentCameraId = null;
                selectedAsset = null;
                setView('home');
            }
        }

        async function selectSiteByName(name) {
            const found = (sitesCache || []).find(s => s.name === name);
            currentSite = found || { name, cameras: [] };

            // Reset selection when switching sites
            currentVideo = null;
            currentCameraId = null;
            selectedAsset = null;

            setView('tracker');
            await loadVideos();

            // Auto select first camera if possible
            const cams = getActiveCameras();
            if (cams.length > 0) {
                selectCamera(cams[0].id);
            }
            await loadZones();
            updateSteps();
        }

        // Initialize
        init();

        async function init() {
            // Important: récupérer d'abord les streams, puis construire l'UI (évite les resets)
            await updateActiveStreams();
            await loadBackendCameras();
            loadSites();

            // Nav
            navSites?.addEventListener('click', () => setView('home'));
            navTracker?.addEventListener('click', async () => {
                if (!currentSite) {
                    setView('home');
                    return;
                }
                setView('tracker');
                await loadVideos();
                await loadZones();
                updateSteps();
            });

            // Create site
            async function handleCreateSite() {
                const name = (newSiteNameInput?.value || '').trim();
                if (!name) return;
                try {
                    createSite(name);
                    newSiteNameInput.value = '';
                    loadSites();
                    await selectSiteByName(name);
                } catch (e) {
                    uiAlert(`Erreur création site: ${e?.message || e}`, 'Sites');
                }
            }
            createSiteBtn?.addEventListener('click', handleCreateSite);
            newSiteNameInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') handleCreateSite();
            });

            // Add camera (site)
            function switchCamSourceTab(sourceType) {
                currentCamSourceType = sourceType;
                // Update tabs
                camSourceTabs.forEach(tab => {
                    if (tab.dataset.source === sourceType) {
                        tab.classList.add('active');
                    } else {
                        tab.classList.remove('active');
                    }
                });
                // Update panels
                camSourcePanels.forEach(panel => {
                    if (panel.dataset.source === sourceType) {
                        panel.classList.remove('hidden');
                    } else {
                        panel.classList.add('hidden');
                    }
                });
            }

            async function detectWebcams() {
                if (newCamWebcam) newCamWebcam.innerHTML = '<option value="">Détection...</option>';
                try {
                    const res = await fetch('/api/cameras/detect/webcams');
                    const data = await res.json();
                    if (newCamWebcam) {
                        newCamWebcam.innerHTML = '';
                        if (data.webcams && data.webcams.length > 0) {
                            data.webcams.forEach(w => {
                                newCamWebcam.innerHTML += `<option value="${w.device_id}">${escapeHtml(w.name)} (${w.resolution})</option>`;
                            });
                        } else {
                            newCamWebcam.innerHTML = '<option value="">Aucune webcam détectée</option>';
                        }
                    }
                } catch (e) {
                    console.error('Webcam detection failed:', e);
                    if (newCamWebcam) newCamWebcam.innerHTML = '<option value="">Erreur de détection</option>';
                }
            }

            async function testRtspConnection() {
                const url = newCamRtspUrl?.value?.trim();
                if (!url) {
                    uiAlert('Entrez une URL RTSP', 'Test RTSP');
                    return;
                }
                if (testRtspBtn) testRtspBtn.textContent = '...';
                try {
                    const res = await fetch(`/api/cameras/test-rtsp?url=${encodeURIComponent(url)}`, { method: 'POST' });
                    const data = await res.json();
                    if (data.success) {
                        uiAlert(`Connexion réussie ! Résolution: ${data.resolution}`, 'Test RTSP');
                        if (testRtspBtn) testRtspBtn.textContent = '\u2713';
                    } else {
                        uiAlert(`Échec: ${data.error}`, 'Test RTSP');
                        if (testRtspBtn) testRtspBtn.textContent = '\u2717';
                    }
                } catch (e) {
                    uiAlert(`Erreur: ${e.message}`, 'Test RTSP');
                    if (testRtspBtn) testRtspBtn.textContent = '\u2717';
                }
                setTimeout(() => { if (testRtspBtn) testRtspBtn.textContent = '\u2713'; }, 2000);
            }

            async function scanOnvifCameras() {
                if (onvifScanStatus) onvifScanStatus.textContent = 'Scan en cours...';
                try {
                    const res = await fetch('/api/cameras/detect/onvif');
                    const data = await res.json();
                    if (data.error) {
                        if (onvifScanStatus) onvifScanStatus.textContent = data.error;
                        return;
                    }
                    if (data.cameras && data.cameras.length > 0) {
                        const names = data.cameras.map(c => c.name).join(', ');
                        if (onvifScanStatus) onvifScanStatus.textContent = `Trouvé: ${names}`;
                        // TODO: Could show a picker modal
                        uiAlert(`${data.cameras.length} caméra(s) ONVIF trouvée(s):\n${data.cameras.map(c => `${c.name}: ${c.xaddr}`).join('\n')}`, 'Scan ONVIF');
                    } else {
                        if (onvifScanStatus) onvifScanStatus.textContent = 'Aucune caméra trouvée';
                    }
                } catch (e) {
                    console.error('ONVIF scan failed:', e);
                    if (onvifScanStatus) onvifScanStatus.textContent = 'Erreur de scan';
                }
            }

            function openAddCameraForm() {
                if (!addCameraForm) return;
                addCameraForm.classList.remove('hidden');
                // Reset to video tab
                switchCamSourceTab('video');
                // populate videos
                if (newCamVideo) {
                    newCamVideo.innerHTML = '';
                    (availableVideosList || []).forEach((v) => {
                        const displayName = truncateFilename(v, 30);
                        newCamVideo.innerHTML += `<option value="${escapeHtml(v)}" title="${escapeHtml(v)}">${escapeHtml(displayName)}</option>`;
                    });
                }
                // Auto-detect webcams when opening form
                detectWebcams();
                newCamName?.focus();
            }
            function closeAddCameraForm() {
                addCameraForm?.classList.add('hidden');
                if (newCamName) newCamName.value = '';
                if (newCamHint) newCamHint.value = '';
                if (newCamRtspUrl) newCamRtspUrl.value = '';
                // Reset upload state - use form.reset() which properly clears file inputs
                if (uploadVideoForm) uploadVideoForm.reset();
                if (uploadVideoLabelText) uploadVideoLabelText.textContent = 'Choisir un fichier';
                if (uploadProgress) uploadProgress.textContent = '';
                if (onvifScanStatus) onvifScanStatus.textContent = '';
            }

            // Upload video handler function (extracted so we can re-attach after clone)
            async function handleVideoUpload(e) {
                const input = e.target;
                console.log('[Upload] Change event fired');
                const file = input.files?.[0];
                if (!file) {
                    console.log('[Upload] No file selected');
                    return;
                }

                console.log('[Upload] Starting upload for:', file.name);
                const formData = new FormData();
                formData.append('file', file);

                if (uploadVideoLabelText) uploadVideoLabelText.textContent = truncateFilename(file.name);
                if (uploadProgress) uploadProgress.textContent = 'Upload...';

                try {
                    const resp = await fetch('/api/videos/upload', {
                        method: 'POST',
                        body: formData
                    });
                    console.log('[Upload] Response status:', resp.status);
                    if (!resp.ok) {
                        const errText = await resp.text();
                        console.error('[Upload] Error response:', errText);
                        throw new Error(`Upload échoué (${resp.status})`);
                    }
                    const data = await resp.json();
                    console.log('[Upload] Success:', data);

                    if (uploadProgress) uploadProgress.textContent = 'OK !';

                    // Refresh video list
                    await loadVideos();

                    // Rebuild newCamVideo select with updated availableVideosList
                    if (newCamVideo && data.filename) {
                        newCamVideo.innerHTML = '';
                        (availableVideosList || []).forEach((v) => {
                            const displayName = truncateFilename(v, 30);
                            newCamVideo.innerHTML += `<option value="${escapeHtml(v)}" title="${escapeHtml(v)}">${escapeHtml(displayName)}</option>`;
                        });
                        // Select the uploaded video
                        newCamVideo.value = data.filename;
                    }

                    setTimeout(() => { if (uploadProgress) uploadProgress.textContent = ''; }, 2000);
                } catch (e) {
                    console.error('[Upload] Exception:', e);
                    if (uploadProgress) uploadProgress.textContent = `Erreur: ${e.message}`;
                }
            }

            // Upload video - auto upload on file select
            console.log('[Init] Attaching upload listener to:', uploadVideoInput);
            if (uploadVideoInput) {
                uploadVideoInput.addEventListener('change', handleVideoUpload);
            }

            addCameraBtn?.addEventListener('click', () => {
                if (currentView !== 'tracker') return;
                openAddCameraForm();
            });
            cancelCamBtn?.addEventListener('click', closeAddCameraForm);

            // Camera source tab switching
            camSourceTabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const sourceType = tab.dataset.source;
                    if (sourceType) switchCamSourceTab(sourceType);
                });
            });

            // Webcam detection button
            detectWebcamsBtn?.addEventListener('click', detectWebcams);

            // RTSP test button
            testRtspBtn?.addEventListener('click', testRtspConnection);

            // ONVIF scan button
            scanOnvifBtn?.addEventListener('click', scanOnvifCameras);

            saveCamBtn?.addEventListener('click', async () => {
                const name = (newCamName?.value || '').trim();
                const hint = (newCamHint?.value || '').trim();

                if (!name) {
                    uiAlert('Nom de caméra requis.', 'Caméras');
                    return;
                }

                try {
                    const cams = Array.isArray(currentSite?.cameras) ? [...currentSite.cameras] : [];
                    const desired = cameraIdFromName(name);
                    const id = makeUniqueCameraId(cams, desired);

                    let camData = { id, name, hint };

                    if (currentCamSourceType === 'video') {
                        const video = (newCamVideo?.value || '').trim();
                        if (!video) {
                            uiAlert('Sélectionnez une vidéo.', 'Caméras');
                            return;
                        }
                        camData.video = video;
                        camData.sourceType = 'video';
                    } else if (currentCamSourceType === 'webcam') {
                        const deviceId = newCamWebcam?.value;
                        if (deviceId === '' || deviceId === undefined) {
                            uiAlert('Sélectionnez une webcam.', 'Caméras');
                            return;
                        }
                        // Add camera to backend
                        const backendCamId = `webcam_${id}`;
                        await addBackendCamera(backendCamId, name, 'webcam', deviceId);
                        camData.backendCameraId = backendCamId;
                        camData.sourceType = 'webcam';
                    } else if (currentCamSourceType === 'rtsp') {
                        const rtspUrl = (newCamRtspUrl?.value || '').trim();
                        if (!rtspUrl) {
                            uiAlert('Entrez une URL RTSP.', 'Caméras');
                            return;
                        }
                        // Add camera to backend
                        const backendCamId = `rtsp_${id}`;
                        await addBackendCamera(backendCamId, name, 'rtsp', rtspUrl);
                        camData.backendCameraId = backendCamId;
                        camData.sourceType = 'rtsp';
                    }

                    cams.push(camData);
                    saveCurrentSiteCameras(cams);
                    loadSites();
                    closeAddCameraForm();
                    await loadVideos();
                    if (!currentVideo && !currentCameraId) selectCamera(id);
                    await loadZones();
                } catch (e) {
                    uiAlert(`Erreur ajout caméra: ${e?.message || e}`, 'Caméras');
                }
            });

            // Click delegation for site cards/rows
            sitesGrid?.addEventListener('click', async (e) => {
                const del = e.target?.closest?.('[data-delete-site]');
                if (del) {
                    const key = del.getAttribute('data-delete-site') || '';
                    const name = decodeURIComponent(key);
                    const ok = await uiConfirm(`Supprimer le site "${name}" ?`, 'Suppression');
                    if (ok) {
                        try { deleteSiteByName(name); loadSites(); } catch (err) { uiAlert(err?.message || String(err), 'Suppression'); }
                    }
                    return;
                }
                const el = e.target?.closest?.('[data-site]');
                if (!el) return;
                const key = el.getAttribute('data-site') || '';
                const name = decodeURIComponent(key);
                selectSiteByName(name);
            });
            zoneListSidebar?.addEventListener('click', (e) => {
                // Gestion des clics sur les sites
                const siteEl = e.target?.closest?.('[data-site]');
                if (siteEl) {
                    const key = siteEl.getAttribute('data-site') || '';
                    const name = decodeURIComponent(key);
                    selectSiteByName(name);
                    return;
                }
                
                // Gestion des clics sur les zones dans la sidebar
                const zoneEl = e.target?.closest?.('[data-select-zone]');
                if (zoneEl) {
                    e.preventDefault();
                    e.stopPropagation();
                    const zoneName = zoneEl.getAttribute('data-select-zone') || '';
                    if (zoneName) window.selectZone(zoneName);
                    return;
                }
                
                // Gestion des clics sur les dessins dans la sidebar
                const drawingEl = e.target?.closest?.('[data-select-drawing]');
                if (drawingEl) {
                    e.preventDefault();
                    e.stopPropagation();
                    const zoneName = drawingEl.getAttribute('data-select-drawing') || '';
                    const idx = Number(drawingEl.getAttribute('data-drawing-idx'));
                    if (zoneName && Number.isFinite(idx)) window.selectDrawing(zoneName, idx);
                    return;
                }
                
                // Gestion des clics sur les caméras dans la sidebar
                const cameraEl = e.target?.closest?.('[data-select-camera]');
                if (cameraEl) {
                    e.preventDefault();
                    e.stopPropagation();
                    const cameraId = cameraEl.getAttribute('data-select-camera') || '';
                    if (cameraId) selectCamera(cameraId);
                    return;
                }
            });
            
            // Gestionnaire pour les caméras dans le panneau caméras
            if (cameraGrid) {
                cameraGrid.addEventListener('click', (e) => {
                    const deleteBtn = e.target?.closest?.('[data-delete-camera]');
                    if (deleteBtn) {
                        e.preventDefault();
                        e.stopPropagation();
                        const cameraId = deleteBtn.getAttribute('data-delete-camera') || '';
                        if (cameraId) deleteCamera(cameraId);
                        return;
                    }

                    const selectEl = e.target?.closest?.('[data-select-camera]');
                    if (selectEl) {
                        e.preventDefault();
                        e.stopPropagation();
                        const cameraId = selectEl.getAttribute('data-select-camera') || '';
                        if (cameraId) selectCamera(cameraId);
                        return;
                    }
                });
            }

            // DEMO: always start on home; refresh resets sites to DEMO_SITES
            setView('home');

            function startLoadZonesLoop() {
                if (loadZonesLoopTimer) clearTimeout(loadZonesLoopTimer);
                const tick = async () => {
                    const isDetecting = !!(currentVideo && activeVideoStreams.has(currentVideo));
                    const nextMs = isDetecting ? PRESENCE_POLL_ACTIVE_MS : PRESENCE_POLL_IDLE_MS;
                    try { await loadZones(); } catch {}
                    loadZonesLoopTimer = setTimeout(tick, nextMs);
                };
                tick();
            }
            startLoadZonesLoop();
            setInterval(updateActiveStreams, 2000);
        }

        const DEFAULT_CAMERA_DEFS = [
            {
                id: 'entr1',
                name: 'Entrepôt / Logistique',
                hint: 'Déchargement & présence',
                video: 'entr1.mp4'
            },
            {
                id: 'convoyeur',
                name: 'Convoyeur / Ligne',
                hint: 'Tapis roulant & contrôle',
                video: 'video_01.mp4'
            }
        ];

        function getActiveCameras() {
            // En mode multi-site: un site peut être vide (0 caméra)
            if (currentSite && Array.isArray(currentSite.cameras)) return currentSite.cameras;
            return [];
        }

        function getCameraByVideo(videoName) {
            // Support both video files and camera sources (camera:xxx)
            if (videoName && videoName.startsWith('camera:')) {
                const backendId = videoName.replace('camera:', '');
                return getActiveCameras().find(c => c.backendCameraId === backendId) || null;
            }
            return getActiveCameras().find(c => c.video === videoName) || null;
        }

        function getCameraById(id) {
            return getActiveCameras().find(c => c.id === id) || null;
        }

        async function loadVideos() {
            const res = await fetch('/api/videos');
            const data = await res.json();
            const available = new Set(data.videos || []);
            const cams = getActiveCameras();
            availableVideosList = data.videos || [];

            // Garder la liste de vidéos pour l'interne (upload / fallback)
            const selected = videoSelect.value || currentVideo || '';
            videoSelect.innerHTML = '<option value="">-- Choisir une vidéo --</option>';
            (data.videos || []).forEach((v) => {
                videoSelect.innerHTML += `<option value="${v}">${v}</option>`;
            });

            // Add backend cameras (webcam/rtsp) to the select
            cams.forEach((cam) => {
                if (cam.sourceType === 'webcam' || cam.sourceType === 'rtsp') {
                    const sourceKey = `camera:${cam.backendCameraId}`;
                    videoSelect.innerHTML += `<option value="${sourceKey}">[${cam.sourceType.toUpperCase()}] ${cam.name}</option>`;
                    available.add(sourceKey); // Mark as available
                }
            });

            if (selected && available.has(selected)) videoSelect.value = selected;

            // UI: 2 caméras fixes (simple & fiable)
            cameraGrid.innerHTML = '';
            if (!cams || cams.length === 0) {
                cameraGrid.innerHTML = `
                    <div class="no-zones" style="grid-column: 1 / -1; text-align:left;">
                        Aucune caméra sur ce site. Cliquez sur <b>+</b> pour en ajouter une.
                    </div>
                `;
                return;
            }
            cams.forEach((cam) => {
                // Support both video files and backend cameras (webcam/rtsp)
                const isBackendCamera = cam.sourceType === 'webcam' || cam.sourceType === 'rtsp';
                const sourceKey = isBackendCamera ? `camera:${cam.backendCameraId}` : cam.video;

                const videoExists = isBackendCamera ? !!cam.backendCameraId : available.has(cam.video);
                const isActive = activeVideoStreams.has(sourceKey);
                const isCurrent = sourceKey === currentVideo;

                let statusText = 'Prête';
                if (!videoExists) {
                    statusText = 'Manquante';
                } else if (isActive) {
                    statusText = 'En ligne';
                }

                const sourceDisplayName = isBackendCamera
                    ? `${cam.sourceType.toUpperCase()}`
                    : truncateFilename(cam.video || '', 12);

                cameraGrid.innerHTML += `
                    <div class="camera-item ${isCurrent ? 'active' : ''}" data-camera="${cam.id}">
                        <div class="camera-item-header">
                            <div data-select-camera="${cam.id}" style="flex:1;cursor:pointer;">
                                <div class="camera-item-name">${cam.name}</div>
                                <div class="camera-item-status ${videoExists ? (isActive ? 'online' : 'offline') : 'offline'}">
                                    <span class="camera-status-dot" style="width:6px;height:6px;border-radius:50%;background:currentColor;"></span>
                                    <span class="camera-status-text">${statusText}</span>
                                </div>
                                <div style="margin-top: var(--space-2); font-size: var(--text-xs); color: var(--color-text-muted);">
                                    ${(cam.hint || '')} • <span style="font-family: 'Courier New', monospace;" title="${escapeHtml(sourceKey || '')}">${escapeHtml(sourceDisplayName)}</span>
                                </div>
                            </div>
                            <button class="camera-item-delete" data-delete-camera="${cam.id}" title="Supprimer cette caméra">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                `;
            });
        }

        function selectVideo(videoName) {
            videoSelect.value = videoName;
            videoSelect.dispatchEvent(new Event('change'));
        }

        function selectCamera(cameraId) {
            const cam = getCameraById(cameraId);
            if (!cam) return;
            currentCameraId = cameraId;

            // Handle different source types
            if (cam.sourceType === 'webcam' || cam.sourceType === 'rtsp') {
                // Use backend camera - treat it like a video with name "camera:xxx"
                if (cam.backendCameraId) {
                    selectVideo(`camera:${cam.backendCameraId}`);
                }
            } else if (cam.backendCameraId) {
                // Fallback: if backendCameraId is set, use it even without sourceType
                selectVideo(`camera:${cam.backendCameraId}`);
            } else if (cam.video) {
                // Default: video file
                selectVideo(cam.video);
            }
        }

        async function deleteCamera(cameraId) {
            const cam = getCameraById(cameraId);
            if (!cam) return;

            const confirmed = await uiConfirm(`Supprimer la caméra "${cam.name}" ?`, 'Suppression');
            if (!confirmed) return;

            // If it's a backend camera (webcam/rtsp), delete from backend too
            if (cam.backendCameraId) {
                try {
                    await fetch(`/api/cameras/${encodeURIComponent(cam.backendCameraId)}`, { method: 'DELETE' });
                    await loadBackendCameras();
                } catch (e) {
                    console.error('Failed to delete backend camera:', e);
                }
            }

            // Remove from currentSite.cameras
            if (currentSite && Array.isArray(currentSite.cameras)) {
                const idx = currentSite.cameras.findIndex(c => c.id === cameraId);
                if (idx >= 0) {
                    currentSite.cameras.splice(idx, 1);
                }
            }

            // If this was the current camera, clear selection
            if (currentCameraId === cameraId) {
                currentCameraId = null;
                currentVideo = null;
                videoSelect.value = '';
                placeholder.classList.remove('hidden');
                videoFrame.classList.add('hidden');
                videoStream.classList.add('hidden');
                drawCanvas.classList.add('hidden');
            }

            await loadVideos(); // This calls renderCameraGrid internally
            await loadZones();
        }

        async function openDrawZones() {
            if (!currentVideo) {
                uiAlert('Sélectionnez d\'abord une caméra.', 'Dessin');
                return;
            }

            // Open the editor overlay
            await editorOpen();
        }

        async function updateActiveStreams() {
            try {
                const prev = new Set(activeVideoStreams || []);
                const res = await fetch('/api/streams');
                const data = await res.json();

                activeVideoStreams = new Set(data.streams.map(s => s.video));

                // Diff start/stop pour garder les compteurs cohérents même si le stream a été lancé ailleurs
                for (const v of activeVideoStreams) {
                    if (!prev.has(v)) {
                        markVideoRunStart(v);
                        ensureZoneLive(v).lastTs = Date.now();
                    }
                }
                for (const v of prev) {
                    if (!activeVideoStreams.has(v)) {
                        markVideoRunStop(v);
                        if (zoneLiveTimersByVideo?.[v]) zoneLiveTimersByVideo[v].lastTs = 0;
                    }
                }

                if (data.streams.length === 0) {
                    activeStreamsDiv.innerHTML = '';
                    stopAllBtn.disabled = true;
                } else {
                    activeStreamsDiv.innerHTML = data.streams.map(s =>
                        `<span class="stream-badge ${s.video === currentVideo ? 'current' : ''}">
                            <span class="dot"></span>
                            ${s.video}
                        </span>`
                    ).join('');
                    stopAllBtn.disabled = false;
                }

                if (currentVideo) {
                    isCurrentVideoStreaming = activeVideoStreams.has(currentVideo);
                    setStartDetectionButtonUi(!!isCurrentVideoStreaming);
                }

                // Met à jour les cartes caméras sans reconstruire le DOM (évite le jitter)
                document.querySelectorAll('.camera-item[data-camera]').forEach((el) => {
                    const camId = el.getAttribute('data-camera');
                    const cam = getCameraById(camId);
                    if (!cam) return;

                    const active = activeVideoStreams.has(cam.video);
                    const isCurrent = cam.video === currentVideo;
                    el.classList.toggle('active', isCurrent);

                    const statusEl = el.querySelector('.camera-item-status');
                    if (!statusEl) return;

                    statusEl.classList.toggle('online', active);
                    statusEl.classList.toggle('offline', !active);
                    const textEl = statusEl.querySelector('.camera-status-text');
                    if (textEl) textEl.textContent = active ? 'En ligne' : 'Prête';
                });

                // steps KPI (site) + cache zones (site)
                if (currentView === 'tracker' && currentSite) {
                    refreshZonesCacheForSite(false).then(updateHeaderStepsKpis).catch(() => updateHeaderStepsKpis());
                } else {
                    updateHeaderStepsKpis();
                }
            } catch (e) {
                console.error('Error fetching streams:', e);
            }
        }

        async function loadZones() {
            if (loadZonesInFlight) return;
            loadZonesInFlight = true;
            try {
            if (currentView !== 'tracker') return;
            if (!currentVideo) {
                const cams = getActiveCameras();
                zonesGrid.innerHTML = '<div class="no-zones">Sélectionnez une vidéo</div>';
                zoneListSidebar.innerHTML = '<div style="color: var(--sidebar-text-subtle); font-size: var(--text-sm);">Sélectionnez une vidéo</div>';
                recapCameras.textContent = `${cams.length}`;
                recapCamerasSub.textContent = 'Caméras configurées';
            recapZones.textContent = '—';
                recapZonesSub.textContent = 'Sélectionnez une caméra';
            recapDrawings.textContent = '—';
            recapDrawingsSub.textContent = '—';
            recapActive.textContent = '—';
            recapActiveSub.textContent = '—';
                updateHeaderStepsKpis();
                return;
            }
            const isDetecting = activeVideoStreams.has(currentVideo);
            // Note: le dénominateur (pour % occupation/absence) ne doit avancer QUE quand la détection tourne.
            // (On garde le cumul par vidéo tant que la page reste ouverte.)
            // refresh cache zones pour le site (toutes caméras)
            await refreshZonesCacheForSite(false);

            // Définitions zones: ne pas re-fetch à chaque tick (sinon on sature quand on augmente la cadence)
            let zonesWithPolygons = zonesCacheByVideo[currentVideo] || {};
            const nowDefs = Date.now();
            const needDefs =
                !zonesDefsFetchedByVideo[currentVideo] ||
                (nowDefs - Number(zonesDefsFetchTsByVideo[currentVideo] || 0)) > ZONES_DEF_TTL_MS;
            if (needDefs) {
                const zonesRes = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}`);
                const zonesData = await zonesRes.json();
                zonesWithPolygons = zonesData.zones || {};
                zonesCacheByVideo[currentVideo] = zonesWithPolygons;
                zonesDefsFetchTsByVideo[currentVideo] = nowDefs;
                zonesDefsFetchedByVideo[currentVideo] = true;
            }

            const isLineZone = (zoneName) => {
                const polys = zonesWithPolygons?.[zoneName]?.polygons || [];
                for (let i = 0; i < polys.length; i++) {
                    if (getDrawType(currentVideo, zoneName, i) === 'line') return true;
                }
                return false;
            };

            // Présence:
            // - si détection active: on fetch et on met à jour le snapshot
            // - sinon: on gèle sur le dernier snapshot (ou 0 si jamais lancé)
            let zones = {};
            if (isDetecting) {
                try {
                    const presenceRes = await fetch(`/api/presence/${encodeURIComponent(currentVideo)}`);
                    const presenceData = await presenceRes.json();
                    zones = presenceData.zones || {};
                    lastPresenceByVideo[currentVideo] = zones;
                    presenceOkTsByVideo[currentVideo] = Date.now();
                    // Met à jour les compteurs locaux (occupation/absence) selon is_occupied
                    updateZoneLiveTimers(currentVideo, zones);
                    // Line zones: compte les passages + série
                    updateLinePass(currentVideo, zones, isLineZone);
                } catch (e) {
                    // Anti "état figé": si /presence échoue, ne pas conserver un ancien "Occupé"
                    zones = {};
                    presenceOkTsByVideo[currentVideo] = 0;
                }
            } else {
                zones = (videoHasRunByVideo[currentVideo] ? (lastPresenceByVideo[currentVideo] || {}) : {});
            }

            // Recap (camera + zones + drawings + active presence)
            const online = activeVideoStreams.size;
            recapCameras.textContent = `${online}/${getActiveCameras().length}`;
            recapCamerasSub.textContent = 'En ligne / configurées';

            const zoneCount = Object.keys(zonesWithPolygons).length;
            recapZones.textContent = `${zoneCount}`;
            recapZonesSub.textContent = 'Zones sur cette caméra';

            let drawingsCount = 0;
            for (const z of Object.values(zonesWithPolygons)) drawingsCount += (z.polygons || []).length;
            recapDrawings.textContent = `${drawingsCount}`;
            recapDrawingsSub.textContent = 'Dessins sur cette caméra';

            const lastOk = Number(presenceOkTsByVideo[currentVideo] || 0);
            const stale = isDetecting ? (!lastOk || ((Date.now() - lastOk) > PRESENCE_STALE_MS)) : false;
            const activeCount = (isDetecting && !stale) ? Object.values(zones).filter(z => z.is_occupied).length : 0;
            recapActive.textContent = `${activeCount}`;
            recapActiveSub.textContent = isDetecting ? (stale ? 'Sync…' : 'Zones occupées') : (videoHasRunByVideo[currentVideo] ? 'Détection en pause' : 'Lancez la détection');

            if (Object.keys(zonesWithPolygons).length === 0) {
                zonesGrid.innerHTML = '<div class="no-zones">Aucune zone définie pour cette vidéo</div>';
                updateDrawPanelZones(zonesWithPolygons);
                // IMPORTANT: même sans zones, la sidebar doit rester au niveau "site" (toutes les caméras),
                // et simplement sélectionner la caméra courante.
                renderAssetTree(zones || {}, zonesWithPolygons || {});
                return;
            }

            zonesGrid.innerHTML = '';
            updateDrawPanelZones(zonesWithPolygons);

            for (const name of Object.keys(zonesWithPolygons).sort()) {
                const info = zones[name] || { formatted_time: '00:00:00', is_occupied: false, total_time: 0 };
                const lastOk = Number(presenceOkTsByVideo[currentVideo] || 0);
                const stale = isDetecting ? (!lastOk || ((Date.now() - lastOk) > PRESENCE_STALE_MS)) : false;
                const uiOcc = isDetecting && !stale && !!info.is_occupied;
                const statusClass = isDetecting ? (stale ? 'empty' : (uiOcc ? 'occupied' : 'empty')) : 'empty';
                const statusLabel = isDetecting ? (stale ? 'Sync…' : (uiOcc ? 'Occupé' : 'Vide')) : (videoHasRunByVideo[currentVideo] ? 'Pause' : 'Prêt');
                const polys = (zonesWithPolygons[name]?.polygons || []);
                const drawings = polys.length;

                const previews = polys.slice(0, 2);
                const previewBoxes = previews.map((poly, idx) => {
                    const svg = buildPolyPreviewSvg(poly);
                    const isSel = selectedAsset && selectedAsset.zone === name && typeof selectedAsset.idx === 'number' && selectedAsset.idx === idx;
                    const zKey = encodeURIComponent(String(name));
                    return `
                        <div class="zone-preview-box ${isSel ? 'active' : ''}" data-zone="${zKey}" data-idx="${idx}" title="Sélectionner la forme #${idx + 1}">
                            ${svg}
                        </div>
                    `;
                }).join('');

                const isSelected = selectedAsset && selectedAsset.zone === name;

                // Compteurs locaux (fiables): occupation + absence
                const live = zoneLiveTimersByVideo?.[currentVideo]?.zones?.[name] || { occ: 0, abs: 0 };
                const occSec = (isDetecting || videoHasRunByVideo[currentVideo]) ? Number(live.occ || 0) : 0;
                const absSec = (isDetecting || videoHasRunByVideo[currentVideo]) ? Number(live.abs || 0) : 0;
                const denom = occSec + absSec;
                const occPct = denom > 0 ? Math.min(100, Math.max(0, (occSec / denom) * 100)) : 0;
                const absPct = denom > 0 ? Math.max(0, 100 - occPct) : 0;
                const secFmt = (s) => `${Math.max(0, Math.floor(Number(s) || 0))} s`;

                const isLine = isLineZone(name);
                const isSingleLineShape = isLine && drawings === 1 && (getDrawType(currentVideo, name, 0) === 'line');
                // UX: previews repliées par défaut pour les cartes "présence" (zones polygones).
                // (on laisse le comptage gérer son UI à part)
                const isPreviewsCollapsed = !isLine ? (presencePreviewsCollapsedByVideo?.[currentVideo]?.[name] ?? true) : false;
                const linePass = linePassByVideo?.[currentVideo]?.[name] || { count: 0, series: [] };
                const uptime = denom; // temps total où la détection tournait (par zone)
                const passageTime = occSec; // temps "actif" sur la ligne (proxy passage)

                zonesGrid.innerHTML += `
                    <div class="zone-card ${isSelected ? 'selected' : ''} ${(!isLine && isPreviewsCollapsed) ? 'is-previews-collapsed' : ''}" data-zone="${encodeURIComponent(String(name))}">
                        <div class="zone-card-header">
                            <div>
                                <div class="zone-name-pill">${name}</div>
                                ${!isLine ? `
                                    <button class="zone-forms-toggle" type="button" data-zone="${encodeURIComponent(String(name))}" aria-label="Afficher/Masquer les formes">
                                        <span>${drawings} forme(s)</span>
                                        <svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                            <path d="M7 10l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </button>
                                ` : `
                                    <div class="zone-forms-count">${drawings} forme(s)</div>
                                `}
                            </div>
                            <div class="zone-card-status ${statusClass}">${statusLabel}</div>
                        </div>
                        ${isSingleLineShape ? '' : `
                            <div class="zone-previews ${(!isLine && isPreviewsCollapsed) ? 'is-collapsed' : ''}">
                                ${previewBoxes}
                            </div>
                        `}
                        ${isLine ? `
                            ${isSingleLineShape ? `
                                <div class="line-kpi-row">
                                    <div class="line-kpi-left">
                                        <div class="line-kpi-num">${Number(linePass.count || 0)}</div>
                                        <div class="line-kpi-label">Passages</div>
                                    </div>
                                    <div class="line-kpi-right">
                                        <div class="line-kpi-mini"><span>Uptime</span><span>${formatHMS(uptime)}</span></div>
                                        <div class="line-kpi-mini"><span>Temps</span><span>${secFmt(passageTime)}</span></div>
                                    </div>
                                </div>
                            ` : `
                                <div class="line-metrics">
                                    <div class="line-metric"><span>Passages</span><span>${Number(linePass.count || 0)}</span></div>
                                    <div class="line-metric"><span>Uptime</span><span>${formatHMS(uptime)}</span></div>
                                    <div class="line-metric" style="grid-column:1 / -1;"><span>Temps passage</span><span>${secFmt(passageTime)}</span></div>
                                </div>
                            `}
                        ` : `
                            <div class="occ-bars">
                                <div>
                                    <div class="occ-head">
                    <span><strong>Occupation</strong> • <span class="pct">${occPct.toFixed(0)}%</span></span>
                                        <span class="secs">${secFmt(occSec)}</span>
                                    </div>
                                    <div class="occ-track"><div class="occ-fill" style="width:${occPct.toFixed(2)}%"></div></div>
                                </div>
                                <div>
                                    <div class="occ-head">
                    <span><strong>Absence</strong> • <span class="pct">${absPct.toFixed(0)}%</span></span>
                                        <span class="secs">${secFmt(absSec)}</span>
                                    </div>
                                    <div class="occ-track"><div class="occ-fill red" style="width:${absPct.toFixed(2)}%"></div></div>
                                </div>
                            </div>
                        `}
                        <div class="zone-card-actions">
                            <button class="btn btn-ghost btn-icon" data-reset-zone="${escapeHtml(name)}" title="Reset">
                                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                `;

                // Sidebar explorer: Source vidéo > Zones > Dessins
                const polyCount = (zonesWithPolygons[name]?.polygons || []).length;
                zonePolygonCounts[name] = polyCount;
            }

            // Render explorer tree
            renderAssetTree(zones, zonesWithPolygons);
            updateHeaderStepsKpis();
            } finally {
                loadZonesInFlight = false;
            }
        }

        function updateDrawPanelZones(zonesWithPolygons) {
            const prev = drawZoneSelect.value;
            drawZoneSelect.innerHTML = '<option value="">— Choisir —</option><option value="__new__">+ Nouvelle zone…</option>';
            Object.keys(zonesWithPolygons || {}).sort().forEach((z) => {
                drawZoneSelect.innerHTML += `<option value="${z}">${z}</option>`;
            });
            if (prev && [...drawZoneSelect.options].some(o => o.value === prev)) {
                drawZoneSelect.value = prev;
            }
            drawZoneNameGroup.classList.toggle('hidden', drawZoneSelect.value !== '__new__');
        }

        function getDrawType(video, zone, idx) {
            try {
                return localStorage.getItem(`drawmeta:${video}:${zone}:${idx}`) || 'include';
            } catch {
                return 'include';
            }
        }

        function setDrawType(video, zone, idx, type) {
            try {
                localStorage.setItem(`drawmeta:${video}:${zone}:${idx}`, type);
            } catch {}
        }

        function setLineMeta(video, zone, idx, meta) {
            try {
                localStorage.setItem(`linemeta:${video}:${zone}:${idx}`, JSON.stringify(meta || {}));
            } catch {}
        }

        function getLineMeta(video, zone, idx) {
            try {
                const raw = localStorage.getItem(`linemeta:${video}:${zone}:${idx}`);
                if (!raw) return null;
                return JSON.parse(raw);
            } catch {
                return null;
            }
        }

        function computeLineArrowFromMeta(meta, polyFallback) {
            // Retourne { mid:[x,y], end:[x,y] } en coords canvas
            let mid = null;
            let dir = null;
            if (meta?.p1 && meta?.p2 && Array.isArray(meta.p1) && Array.isArray(meta.p2)) {
                mid = [(meta.p1[0] + meta.p2[0]) / 2, (meta.p1[1] + meta.p2[1]) / 2];
                if (meta?.dir && Array.isArray(meta.dir)) dir = meta.dir;
                else {
                    const u = norm(sub(meta.p2, meta.p1));
                    dir = [-u[1], u[0]];
                }
            }
            if (!mid && polyFallback?.length) {
                // fallback grossier: centre du polygone + direction selon la plus longue arête
                let cx = 0, cy = 0;
                for (const p of polyFallback) { cx += p[0]; cy += p[1]; }
                cx /= polyFallback.length; cy /= polyFallback.length;
                mid = [cx, cy];
                let best = { d: 0, v: [1, 0] };
                for (let i = 0; i < polyFallback.length; i++) {
                    const a = polyFallback[i];
                    const b = polyFallback[(i + 1) % polyFallback.length];
                    const v = sub(b, a);
                    const d = v[0] * v[0] + v[1] * v[1];
                    if (d > best.d) best = { d, v };
                }
                const u = norm(best.v);
                dir = [-u[1], u[0]];
            }
            if (!mid) return null;
            const udir = norm(dir || [0, -1]);
            const end = add(mid, mul(udir, 60));
            return { mid, end, dir: udir };
        }

        function buildPolyPreviewSvg(poly) {
            if (!poly || poly.length < 3) return '';
            // Normalise points to a small viewBox with padding
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const p of poly) {
                minX = Math.min(minX, p[0]); minY = Math.min(minY, p[1]);
                maxX = Math.max(maxX, p[0]); maxY = Math.max(maxY, p[1]);
            }
            const w = Math.max(1, maxX - minX);
            const h = Math.max(1, maxY - minY);
            const pad = 6;
            const vw = 100, vh = 64;
            const sx = (vw - pad * 2) / w;
            const sy = (vh - pad * 2) / h;
            const s = Math.min(sx, sy);
            const pts = poly.map(p => {
                const x = (p[0] - minX) * s + pad;
                const y = (p[1] - minY) * s + pad;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(' ');
            return `
                <svg viewBox="0 0 ${vw} ${vh}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
                    <polygon points="${pts}" fill="rgba(34,197,94,0.18)" stroke="#22c55e" stroke-width="3" />
                </svg>
            `;
        }

        function colorsForType(type, isActive = false) {
            // palette: include=vert, line=bleu, exclude=orange/rouge
            const base = {
                include: { stroke: '#22c55e', fill: 'rgba(34,197,94,0.16)' },
                line: { stroke: '#10B0F9', fill: 'rgba(16,176,249,0.12)' },
                exclude: { stroke: '#F08321', fill: 'rgba(240,131,33,0.16)' }
            }[type] || { stroke: '#22c55e', fill: 'rgba(34,197,94,0.16)' };
            if (!isActive) return base;
            return { stroke: '#10B0F9', fill: 'rgba(16,176,249,0.18)' };
        }

        function renderAssetTree(presenceZones, zonesWithPolygons) {
            const cams = getActiveCameras();
            if (!cams || cams.length === 0) {
                zoneListSidebar.innerHTML = '<div style="color: var(--sidebar-text-subtle); font-size: var(--text-sm);">Aucune caméra</div>';
                return;
            }

            const prevScrollTop = zoneListSidebar.scrollTop || 0;

            const cameraRows = cams.map((cam) => {
                const isCurrent = cam.video === currentVideo;
                const camLabel = escapeHtml(cam.name || cam.video);
                const camMeta = escapeHtml(cam.id || 'Source');
                const videoKey = cam.video;

                const defs = isCurrent ? (zonesWithPolygons || {}) : (zonesCacheByVideo?.[videoKey] || {});
                const presence = isCurrent ? (presenceZones || {}) : (lastPresenceByVideo?.[videoKey] || {});

                const zonesHtml = `
                    <div class="tree-children">
                        ${Object.keys(defs || {}).length === 0 ? `
                            <div class="tree-row tree-leaf" style="cursor: default; opacity: 0.65;">
                                <div class="left">
                                    <span class="label">Aucune zone</span>
                                </div>
                            </div>
                        ` : Object.keys(defs || {}).sort().map((zoneName) => {
                            const info = presence?.[zoneName] || { formatted_time: '00:00:00', is_occupied: false };
                            const drawings = defs?.[zoneName]?.polygons || [];
                            const dotClass = info.is_occupied ? 'occupied' : '';
                            const isCollapsed = sidebarZonesCollapsedByVideo?.[videoKey]?.[zoneName] ?? true; // Par défaut replié
                            const hasDrawings = drawings.length > 0;

                            const zoneRow = isCurrent
                                ? `
                                    <div class="tree-row" data-select-zone="${escapeHtml(zoneName)}" style="cursor: pointer;">
                                        <div class="left">
                                            ${hasDrawings ? `
                                                <button class="tree-toggle-btn" onclick="event.stopPropagation(); toggleSidebarZone('${videoKey}', '${zoneName}')" type="button" aria-label="${isCollapsed ? 'Déplier' : 'Replier'}">
                                                    <svg class="tree-chevron ${isCollapsed ? 'collapsed' : 'expanded'}" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                                        <path d="M9 18l6-6-6-6" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            ` : '<span style="width: 18px;"></span>'}
                                            <span class="zone-dot ${dotClass}"></span>
                                            <span class="label">${escapeHtml(zoneName)}</span>
                                        </div>
                                    </div>
                                `
                                : `
                                    <div class="tree-row" style="cursor: default; opacity: 0.75;">
                                        <div class="left">
                                            ${hasDrawings ? `
                                                <button class="tree-toggle-btn" onclick="event.stopPropagation(); toggleSidebarZone('${videoKey}', '${zoneName}')" type="button" aria-label="${isCollapsed ? 'Déplier' : 'Replier'}">
                                                    <svg class="tree-chevron ${isCollapsed ? 'collapsed' : 'expanded'}" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                                        <path d="M9 18l6-6-6-6" stroke-linecap="round" stroke-linejoin="round"/>
                                                    </svg>
                                                </button>
                                            ` : '<span style="width: 18px;"></span>'}
                                            <span class="zone-dot ${dotClass}"></span>
                                            <span class="label">${escapeHtml(zoneName)}</span>
                                        </div>
                                    </div>
                                `;

                            const drawingsRows = drawings.map((_, idx) => {
                                return isCurrent
                                    ? `
                                        <div class="tree-row tree-leaf" data-select-drawing="${escapeHtml(zoneName)}" data-drawing-idx="${idx}" style="cursor: pointer;">
                                            <div class="left">
                                                <span class="label">Dessin ${idx + 1}</span>
                                            </div>
                                        </div>
                                    `
                                    : `
                                        <div class="tree-row tree-leaf" style="cursor: default; opacity: 0.6;">
                                            <div class="left">
                                                <span class="label">Dessin ${idx + 1}</span>
                                            </div>
                                        </div>
                                    `;
                            }).join('');

                            return `
                                ${zoneRow}
                                <div class="tree-children ${isCollapsed ? 'is-collapsed' : ''}">
                                    ${drawingsRows}
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;

                return `
                    <div class="tree-row ${isCurrent ? 'active' : ''}" data-select-camera="${escapeHtml(cam.id)}" style="cursor: pointer;">
                        <div class="left">
                            <img class="tree-icon-img" src="/static/assets_youn/SvIcons/video-camera-svgrepo-com.svg" alt="">
                            <span class="label">${camLabel}</span>
                        </div>
                        <span class="meta">${camMeta}</span>
                    </div>
                    ${zonesHtml}
                `;
            }).join('');

            const root = `
                <div class="draw-tree">
                    ${cameraRows}
                </div>
            `;
            zoneListSidebar.innerHTML = root;
            zoneListSidebar.scrollTop = prevScrollTop;
        }

        // Globaux cliquables depuis le HTML (style explorateur)
        window.selectZone = (zoneName) => {
            drawZoneSelect.value = zoneName;
            drawZoneNameGroup.classList.add('hidden');
            selectedAsset = { zone: zoneName };
            drawExistingZones();
            syncPresenceSelectionUI();
        };

        window.selectDrawing = (zoneName, idx) => {
            selectedAsset = { zone: zoneName, idx };
            drawExistingZones();
            syncPresenceSelectionUI();
        };

        window.toggleSidebarZone = (videoKey, zoneName) => {
            if (!sidebarZonesCollapsedByVideo[videoKey]) {
                sidebarZonesCollapsedByVideo[videoKey] = {};
            }
            const current = sidebarZonesCollapsedByVideo[videoKey][zoneName] ?? true;
            sidebarZonesCollapsedByVideo[videoKey][zoneName] = !current;
            // Stabilisation UI: ne déclenche PAS de fetch/rebuild async.
            // On re-render la sidebar immédiatement depuis les caches en mémoire.
            const curVideo = currentVideo;
            const presence = curVideo ? (lastPresenceByVideo[curVideo] || {}) : {};
            const zonesWithPolygons = curVideo ? (zonesCacheByVideo[curVideo] || {}) : {};
            renderAssetTree(presence, zonesWithPolygons);
        };

        function syncPresenceSelectionUI() {
            // Applique la surbrillance immédiatement (sans attendre le prochain loadZones à 1s)
            try {
                const zSel = selectedAsset?.zone || null;
                const idxSel = (selectedAsset && typeof selectedAsset.idx === 'number') ? selectedAsset.idx : null;
                document.querySelectorAll('.zone-card[data-zone]').forEach((el) => {
                    const z = decodeURIComponent(el.getAttribute('data-zone') || '');
                    el.classList.toggle('selected', !!zSel && z === zSel);
                });
                document.querySelectorAll('.zone-preview-box[data-zone][data-idx]').forEach((el) => {
                    const z = decodeURIComponent(el.getAttribute('data-zone') || '');
                    const idx = Number(el.getAttribute('data-idx'));
                    el.classList.toggle('active', !!zSel && z === zSel && idxSel !== null && idx === idxSel);
                });
            } catch {}
        }

        // Click sur les cartes de la section "Présences" => surbrillance / sélection (reuse des mêmes fonctions que la sidebar)
        zonesGrid?.addEventListener('click', (e) => {
            const t = e.target;
            if (!(t instanceof Element)) return;
            
            // Gestion des boutons reset (priorité haute)
            const resetBtn = t.closest('[data-reset-zone]');
            if (resetBtn) {
                e.preventDefault();
                e.stopPropagation();
                const zoneName = resetBtn.getAttribute('data-reset-zone') || '';
                if (zoneName) resetZoneTimer(zoneName);
                return;
            }
            
            // Gestion du toggle des previews
            const toggle = t.closest('.zone-forms-toggle');
            if (toggle) {
                e.preventDefault();
                e.stopPropagation();
                const z = decodeURIComponent(toggle.getAttribute('data-zone') || '');
                if (!z || !currentVideo) return;
                if (!presencePreviewsCollapsedByVideo[currentVideo]) presencePreviewsCollapsedByVideo[currentVideo] = {};
                // état par défaut = replié (true). Donc au premier clic on déplie.
                const prev = presencePreviewsCollapsedByVideo[currentVideo][z];
                const next = (prev == null) ? false : !prev;
                presencePreviewsCollapsedByVideo[currentVideo][z] = next;
                // Applique immédiatement au DOM (sans attendre un rerender)
                const card = toggle.closest('.zone-card[data-zone]');
                const previewsEl = card?.querySelector('.zone-previews');
                const collapsed = !!presencePreviewsCollapsedByVideo[currentVideo][z];
                card?.classList.toggle('is-previews-collapsed', collapsed);
                previewsEl?.classList.toggle('is-collapsed', collapsed);
                return;
            }
            
            // Ne pas déclencher si clic sur un bouton (autres boutons)
            if (t.closest('button')) {
                e.stopPropagation();
                return;
            }

            // Gestion des previews de dessins
            const preview = t.closest('.zone-preview-box');
            if (preview) {
                e.preventDefault();
                e.stopPropagation();
                const z = decodeURIComponent(preview.getAttribute('data-zone') || '');
                const idx = Number(preview.getAttribute('data-idx'));
                if (z && Number.isFinite(idx)) window.selectDrawing(z, idx);
                return;
            }

            // Gestion des clics sur les cartes de zones
            const card = t.closest('.zone-card[data-zone]');
            if (card) {
                e.preventDefault();
                e.stopPropagation();
                const z = decodeURIComponent(card.getAttribute('data-zone') || '');
                if (z) window.selectZone(z);
                return;
            }
        });

        function updateSteps() {
            if (currentView !== 'tracker') {
                Object.values(steps).forEach(s => s.classList.remove('active', 'done'));
                return;
            }
            Object.values(steps).forEach(s => s.classList.remove('active', 'done'));

            if (!currentVideo) {
                steps.step1.classList.add('active');
            } else if (isDrawing) {
                steps.step1.classList.add('done');
                steps.step2.classList.add('active');
            } else if (isCurrentVideoStreaming) {
                steps.step1.classList.add('done');
                steps.step2.classList.add('done');
                steps.step3.classList.add('active');
            } else {
                steps.step1.classList.add('done');
                steps.step2.classList.add('active');
            }
        }

        function updateStatus(status) {
            statusBadge.className = 'status-indicator ' + status;
            if (status === 'streaming') {
                statusText.textContent = 'Détection en cours';
            } else if (status === 'drawing') {
                statusText.textContent = 'Mode dessin';
            } else {
                statusText.textContent = 'Prêt';
            }
        }

        function syncCanvasSize() {
            const img = videoFrame.classList.contains('hidden') ? videoStream : videoFrame;
            if (!img.naturalWidth) return;

            const displayWidth = img.clientWidth;
            const displayHeight = img.clientHeight;

            drawCanvas.style.width = displayWidth + 'px';
            drawCanvas.style.height = displayHeight + 'px';
        }

        // Video selection
        videoSelect.addEventListener('change', async () => {
            if (!videoSelect.value) return;

            // Stabilisation: changer de vidéo annule tout mode dessin en cours
            if (isDrawing) {
                exitDrawingMode();
            }
            selectedAsset = null;
            drawPanel.classList.add('hidden');

            currentVideo = videoSelect.value;
            const cam = getCameraByVideo(currentVideo);
            currentCameraId = cam ? cam.id : null;
            currentVideoTitle.textContent = cam ? cam.name : currentVideo;

            // Update UI immediately to show selection
            placeholder.classList.add('hidden');
            startDetectionBtn.disabled = false;
            if (editZonesBtn) editZonesBtn.disabled = false;
            updateStatus('ready');

            // Refresh camera grid to show active state FIRST
            await loadVideos();

            // Then fetch video/camera info
            const infoUrl = `/api/videos/${encodeURIComponent(currentVideo)}/info`;
            try {
                const infoRes = await fetch(infoUrl);
                if (!infoRes.ok) {
                    console.error('[videoSelect change] API error:', await infoRes.text());
                    // Don't return - continue with default dimensions
                    videoWidth = 1280;
                    videoHeight = 720;
                } else {
                    const info = await infoRes.json();
                    videoWidth = info.width || 1280;
                    videoHeight = info.height || 720;
                }
            } catch (err) {
                console.error('[videoSelect change] fetch error:', err);
                // Use default dimensions
                videoWidth = 1280;
                videoHeight = 720;
            }

            drawCanvas.width = videoWidth;
            drawCanvas.height = videoHeight;

            isCurrentVideoStreaming = activeVideoStreams.has(currentVideo);

            if (isCurrentVideoStreaming) {
                // Already streaming - show the stream
                videoFrame.classList.add('hidden');
                videoStream.classList.remove('hidden');
                // Only set src if not already set to this stream (avoid duplicate connections)
                const expectedSrc = `/api/stream/${encodeURIComponent(currentVideo)}`;
                if (!videoStream.src.endsWith(expectedSrc)) {
                    videoStream.src = expectedSrc;
                }
                drawCanvas.classList.add('hidden');
                updateStatus('streaming');
            } else {
                // Not streaming - show a static frame (user must click "Lancer détection")
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
                videoFrame.classList.remove('hidden');
                videoStream.classList.add('hidden');
                if (videoStream.src) videoStream.src = '';  // Only clear if set
                drawCanvas.classList.remove('hidden');
            }

            const img = isCurrentVideoStreaming ? videoStream : videoFrame;
            img.onload = () => {
                syncCanvasSize();
                if (!isCurrentVideoStreaming) {
                    drawExistingZones();
                }
            };

            await updateActiveStreams();
            await loadZones();
            updateSteps();
        });

        // Video upload
        videoUpload.addEventListener('change', async () => {
            if (!videoUpload.files.length) return;

            const formData = new FormData();
            formData.append('file', videoUpload.files[0]);

            await fetch('/api/videos/upload', {
                method: 'POST',
                body: formData
            });

            await loadVideos();
            videoSelect.value = videoUpload.files[0].name;
            videoSelect.dispatchEvent(new Event('change'));
            videoUpload.value = '';
        });

        function updateFinishButtonState() {
            const ok = (drawMode === 'poly' && drawPoints.length >= 3) || (drawMode === 'line' && drawPoints.length >= 2);
            finishBtn.disabled = !ok;
        }

        function beginDrawing(mode, name) {
            drawMode = mode;
            activeDrawZoneName = name;
            isDrawing = true;
            drawPoints = [];

            drawHud.classList.remove('hidden');
            drawHudTitle.textContent = `Mode dessin — ${name}`;
            drawInstructions.innerHTML = `
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                ${mode === 'line'
                    ? 'Cliquez pour placer 2 points (ligne).'
                    : 'Cliquez pour placer les points (min 3) puis enregistrer.'}
            `;

            // Garantit que le canvas est visible pour capturer les clics
            drawCanvas.classList.remove('hidden');
            drawCanvas.classList.add('drawing');
            updateFinishButtonState();
            updateSteps();
            updateStatus('drawing');
        }

        function exitDrawingMode() {
            isDrawing = false;
            drawPoints = [];
            activeDrawZoneName = '';
            drawMode = 'poly';

            drawHud.classList.add('hidden');
            finishBtn.disabled = true;
            drawCanvas.classList.remove('drawing');
            updateSteps();
            updateStatus('ready');
        }

        // Panneau dessin (remplacé par l'éditeur "paint-like")
        // (le toggle du panneau legacy est désactivé)

        function updateStartDrawBtnState() {
            const zoneVal = drawZoneSelect.value;
            const isNewZone = zoneVal === '__new__';
            const newZoneName = drawZoneName.value.trim();

            // Disable if no zone selected, or if new zone but no name entered
            const canDraw = zoneVal && (!isNewZone || newZoneName);
            startDrawBtn.disabled = !canDraw;

            if (!canDraw) {
                startDrawBtn.title = 'Sélectionnez d\'abord une zone';
            } else {
                startDrawBtn.title = '';
            }
        }

        drawZoneSelect.addEventListener('change', () => {
            const v = drawZoneSelect.value;
            drawZoneNameGroup.classList.toggle('hidden', v !== '__new__');
            if (v === '__new__') drawZoneName.focus();
            updateStartDrawBtnState();
        });

        drawZoneName.addEventListener('input', () => {
            updateStartDrawBtnState();
        });

        // Initial state
        updateStartDrawBtnState();

        toolPolyBtn.addEventListener('click', () => setTool('poly'));
        toolLineBtn.addEventListener('click', () => setTool('line'));

        editSelectedBtn.addEventListener('click', () => {
            if (!selectedAsset || !selectedAsset.zone || typeof selectedAsset.idx !== 'number') {
                uiAlert('Sélectionnez un dessin dans la sidebar (Source > Zone > Dessin).', 'Edition');
                return;
            }
            if (isCurrentVideoStreaming) {
                uiAlert('Mettez la détection en pause avant d\'éditer.', 'Edition');
                return;
            }
            const poly = cachedZones?.[selectedAsset.zone]?.polygons?.[selectedAsset.idx];
            if (!poly) {
                uiAlert('Dessin introuvable.', 'Edition');
                return;
            }
            editPoints = clonePoints(poly);
            setEditMode(true);
            drawCanvas.classList.remove('hidden');
            drawCanvas.classList.add('drawing');
            redrawCanvas();
        });

        addPointBtn.addEventListener('click', () => {
            if (!editMode || !editPoints) return;
            // Ajout via prochain clic sur une arête (message)
            uiAlert('Maintenez Shift puis cliquez près d\'une arête pour insérer un point.', 'Edition');
            // Le clic canvas gère l'insertion si editMode et pas sur un point
        });

        deletePointBtn.addEventListener('click', () => {
            if (!editMode || !editPoints) return;
            if (editDragging && typeof editDragging.idx === 'number') {
                if (editPoints.length <= 3) {
                    uiAlert('Un polygone doit avoir au moins 3 points.', 'Edition');
                    return;
                }
                editPoints.splice(editDragging.idx, 1);
                editDragging = null;
                redrawCanvas();
                return;
            }
            uiAlert('Sélectionnez un point (cliquez dessus) puis supprimez-le.', 'Edition');
        });

        saveEditBtn.addEventListener('click', async () => {
            if (!editMode) return;
            await saveEditedPolygon();
        });

        startDrawBtn.addEventListener('click', () => {
            if (!currentVideo) {
                uiAlert('Sélectionnez d\'abord une vidéo.', 'Dessin');
                return;
            }
            if (isCurrentVideoStreaming) {
                uiAlert('Mettez la détection en pause avant de dessiner.', 'Dessin');
                return;
            }

            let zoneNameSelected = drawZoneSelect.value;
            if (!zoneNameSelected) {
                uiAlert('Choisissez une zone (ou créez-en une).', 'Dessin');
                return;
            }
            if (zoneNameSelected === '__new__') {
                zoneNameSelected = drawZoneName.value.trim();
                if (!zoneNameSelected) {
                    uiAlert('Entrez un nom de zone.', 'Dessin');
                    drawZoneName.focus();
                    return;
                }
            }
            beginDrawing(drawMode, zoneNameSelected);
        });

        stopDrawBtn.addEventListener('click', () => {
            exitDrawingMode();
            drawExistingZones();
        });

        drawCanvas.addEventListener('click', (e) => {
            if (!isDrawing) return;

            const rect = drawCanvas.getBoundingClientRect();
            const scaleX = drawCanvas.width / rect.width;
            const scaleY = drawCanvas.height / rect.height;

            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            if (drawMode === 'line' && drawPoints.length >= 2) return;

            drawPoints.push([x, y]);
            redrawCanvas();
            updateFinishButtonState();
        });

        // Edition points (drag + insertion)
        drawCanvas.addEventListener('pointerdown', (e) => {
            if (!editMode || !editPoints) return;
            e.preventDefault();
            const p = getCanvasPointFromEvent(e);
            // Hit zone plus large pour attraper le point sans créer à côté
            const idx = nearestVertexIndex(editPoints, p, HANDLE_RADIUS * 3.8);
            if (idx >= 0) {
                editDragging = { idx };
                drawCanvas.setPointerCapture(e.pointerId);
                redrawCanvas();
            } else {
                // insertion de point sur arête: volontaire (Shift) pour éviter les insertions accidentelles
                const inserted = e.shiftKey ? insertPointOnNearestEdge(editPoints, p) : false;
                if (inserted) {
                    redrawCanvas();
                }
            }
        });

        drawCanvas.addEventListener('pointermove', (e) => {
            if (!editMode || !editPoints) return;
            if (!editDragging) return;
            const p = getCanvasPointFromEvent(e);
            const idx = editDragging.idx;
            if (typeof idx === 'number' && editPoints[idx]) {
                editPoints[idx][0] = p[0];
                editPoints[idx][1] = p[1];
                redrawCanvas();
            }
        });

        drawCanvas.addEventListener('pointerup', (e) => {
            if (!editMode) return;
            if (editDragging) {
                editDragging = null;
                redrawCanvas();
            }
        });

        undoBtn.addEventListener('click', () => {
            if (drawPoints.length > 0) {
                drawPoints.pop();
                redrawCanvas();
                updateFinishButtonState();
            }
        });

        finishBtn.addEventListener('click', async () => {
            const isOk = (drawMode === 'poly' && drawPoints.length >= 3) || (drawMode === 'line' && drawPoints.length >= 2);
            if (!isOk) return;

            const zoneName = activeDrawZoneName;
            const polygons = drawMode === 'line'
                ? [lineToPolygon(drawPoints[0], drawPoints[1], 12)]
                : [drawPoints];

            const prevCount = zonePolygonCounts[zoneName] ?? 0;

            await fetch('/api/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: zoneName,
                    polygons,
                    video: currentVideo
                })
            });

            // Meta: mémorise le type du nouveau dessin côté navigateur (sans back)
            try {
                localStorage.setItem(
                    `drawmeta:${currentVideo}:${zoneName}:${prevCount}`,
                    drawMode === 'line' ? 'line' : 'include'
                );
            } catch {}
            // Ligne: mémorise aussi une direction par défaut pour afficher la flèche hors éditeur
            if (drawMode === 'line') {
                try {
                    const p1 = drawPoints[0];
                    const p2 = drawPoints[1];
                    const u = norm(sub(p2, p1));
                    const perp = [-u[1], u[0]];
                    setLineMeta(currentVideo, zoneName, prevCount, { p1, p2, dir: perp });
                } catch {}
            }

            // Recharge et prépare un autre dessin pour la même zone (append)
            await loadZones();
            drawExistingZones();
            drawPoints = [];
            redrawCanvas();
            updateFinishButtonState();

            // UX: on sélectionne automatiquement le dessin créé et on ouvre l'édition
            selectedAsset = { zone: zoneName, idx: prevCount };
            syncPresenceSelectionUI();
            exitDrawingMode();
            await drawExistingZones();
            // entre directement en mode édition pour ajuster sans message "sélectionnez un dessin"
            editSelectedBtn.click();
        });

        cancelBtn.addEventListener('click', () => {
            exitDrawingMode();
            drawExistingZones();
        });

        function lineToPolygon(p1, p2, thickness = 12) {
            const dx = p2[0] - p1[0];
            const dy = p2[1] - p1[1];
            const len = Math.hypot(dx, dy) || 1;
            const nx = -dy / len;
            const ny = dx / len;
            const half = thickness / 2;
            return [
                [p1[0] + nx * half, p1[1] + ny * half],
                [p1[0] - nx * half, p1[1] - ny * half],
                [p2[0] - nx * half, p2[1] - ny * half],
                [p2[0] + nx * half, p2[1] + ny * half]
            ];
        }

        function redrawCanvas() {
            ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
            drawExistingZonesSync();

            // Overlay édition: points + polygone
            if (editMode && editPoints && editPoints.length >= 3) {
                ctx.beginPath();
                ctx.moveTo(editPoints[0][0], editPoints[0][1]);
                for (let i = 1; i < editPoints.length; i++) ctx.lineTo(editPoints[i][0], editPoints[i][1]);
                ctx.closePath();
                ctx.fillStyle = 'rgba(16, 176, 249, 0.12)';
                ctx.fill();
                ctx.strokeStyle = '#10B0F9';
                ctx.lineWidth = 3;
                ctx.stroke();

                editPoints.forEach((p, i) => {
                    const isActive = editDragging && editDragging.idx === i;
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], HANDLE_RADIUS, 0, Math.PI * 2);
                    ctx.fillStyle = isActive ? '#F08321' : '#10B0F9';
                    ctx.fill();
                    ctx.strokeStyle = '#ffffff';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                });
            }

            if (drawPoints.length > 0) {
                if (drawMode === 'line') {
                    // Ligne: 2 points max
                    const p1 = drawPoints[0];
                    const p2 = drawPoints[1];

                    if (p2) {
                        const poly = lineToPolygon(p1, p2, 12);
                        ctx.beginPath();
                        ctx.moveTo(poly[0][0], poly[0][1]);
                        for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
                        ctx.closePath();
                        ctx.fillStyle = 'rgba(16, 176, 249, 0.18)';
                        ctx.fill();
                        ctx.strokeStyle = '#10B0F9';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        // Ligne centrale
                        ctx.beginPath();
                        ctx.moveTo(p1[0], p1[1]);
                        ctx.lineTo(p2[0], p2[1]);
                        ctx.strokeStyle = '#10B0F9';
                        ctx.lineWidth = 3;
                        ctx.stroke();
                    }

                    drawPoints.forEach((p) => {
                        ctx.beginPath();
                        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
                        ctx.fillStyle = '#10B0F9';
                        ctx.fill();
                        ctx.strokeStyle = '#fff';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    });
                } else {
                    // Zone: polygone
                    ctx.beginPath();
                    ctx.moveTo(drawPoints[0][0], drawPoints[0][1]);

                    for (let i = 1; i < drawPoints.length; i++) {
                        ctx.lineTo(drawPoints[i][0], drawPoints[i][1]);
                    }

                    if (drawPoints.length >= 3) {
                        ctx.closePath();
                        ctx.fillStyle = 'rgba(16, 176, 249, 0.25)';
                        ctx.fill();
                    }

                    ctx.strokeStyle = '#10B0F9';
                    ctx.lineWidth = 3;
                    ctx.stroke();

                    drawPoints.forEach((p, i) => {
                        ctx.beginPath();
                        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
                        ctx.fillStyle = i === 0 ? '#22c55e' : '#10B0F9';
                        ctx.fill();
                        ctx.strokeStyle = '#fff';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    });
                }
            }
        }

        let cachedZones = {};

        async function drawExistingZones(force = false) {
            if (!currentVideo) return;

            // cache-bust: après save/edit, certains navigateurs gardent parfois l'ancienne réponse
            const url = `/api/zones/${encodeURIComponent(currentVideo)}${force ? `?t=${Date.now()}` : ''}`;
            const res = await fetch(url, { cache: 'no-store' });
            const data = await res.json();
            cachedZones = data.zones;

            drawExistingZonesSync();
        }

        function drawExistingZonesSync() {
            ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

            for (const [name, zone] of Object.entries(cachedZones)) {
                const polygons = zone.polygons || [];
                for (let idx = 0; idx < polygons.length; idx++) {
                    const polygon = polygons[idx];
                    if (polygon.length < 3) continue;

                    const isSelectedDrawing =
                        selectedAsset &&
                        selectedAsset.zone === name &&
                        typeof selectedAsset.idx === 'number' &&
                        selectedAsset.idx === idx;

                    const isSelectedZone =
                        selectedAsset &&
                        selectedAsset.zone === name &&
                        typeof selectedAsset.idx !== 'number';

                    const type = getDrawType(currentVideo, name, idx);
                    const c = colorsForType(type, (isSelectedDrawing || isSelectedZone));
                    if (type === 'line' && polygon.length === 4) {
                        // Le backend stocke une ligne comme un quadrilatère fin: on la rend comme une vraie ligne + flèche
                        const a = [(polygon[0][0] + polygon[1][0]) / 2, (polygon[0][1] + polygon[1][1]) / 2];
                        const b = [(polygon[2][0] + polygon[3][0]) / 2, (polygon[2][1] + polygon[3][1]) / 2];
                        ctx.beginPath();
                        ctx.moveTo(a[0], a[1]);
                        ctx.lineTo(b[0], b[1]);
                        ctx.strokeStyle = c.stroke;
                        ctx.lineWidth = isSelectedDrawing ? 4 : (isSelectedZone ? 3 : 2);
                        ctx.setLineDash([]);
                        ctx.stroke();

                        // Flèche de sens (UI normale): visible, sans poignée
                        const meta = getLineMeta(currentVideo, name, idx);
                        const arrow = computeLineArrowFromMeta(meta, polygon);
                        if (arrow?.mid && arrow?.end) {
                            drawArrow(ctx, arrow.mid, arrow.end, c.stroke, { shaftWidth: 1.6, dashed: false, head: 22, wing: 13, outline: true, handle: false });
                        }
                    } else {
                        ctx.beginPath();
                        ctx.moveTo(polygon[0][0], polygon[0][1]);
                        for (let i = 1; i < polygon.length; i++) {
                            ctx.lineTo(polygon[i][0], polygon[i][1]);
                        }
                        ctx.closePath();

                        ctx.fillStyle = c.fill;
                        ctx.fill();
                        ctx.strokeStyle = c.stroke;
                        ctx.lineWidth = isSelectedDrawing ? 4 : (isSelectedZone ? 3 : 2);
                        ctx.stroke();
                    }

                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 14px Manrope, system-ui';
                    const label = isSelectedDrawing ? `${name} • ${type.toUpperCase()} ${idx + 1}` : name;
                    ctx.fillText(label, polygon[0][0] + 5, polygon[0][1] - 8);
                }
            }
        }

        // Detection (SLA: toggle fiable OFF <-> ON, anti double-clic)
        let detectionToggleInFlight = false;

        async function applyDetectionUiForCurrentVideo(isOn) {
            setStartDetectionButtonUi(!!isOn);
            if (!currentVideo) return;

            if (isOn) {
                videoFrame.classList.add('hidden');
                videoStream.classList.remove('hidden');
                drawCanvas.classList.add('hidden');
                // Only set src if not already set to this stream (avoid duplicate connections)
                const expectedSrc = `/api/stream/${encodeURIComponent(currentVideo)}`;
                if (!videoStream.src.endsWith(expectedSrc)) {
                    videoStream.src = expectedSrc;
                }
                updateStatus('streaming');
            } else {
                if (videoStream.src) videoStream.src = '';  // Only clear if set
                videoStream.classList.add('hidden');
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
                videoFrame.classList.remove('hidden');
                drawCanvas.classList.remove('hidden');
                updateStatus('ready');
                videoFrame.onload = () => {
                    syncCanvasSize();
                    drawExistingZones();
                };
            }
            updateSteps();
        }

        async function setDetectionForCurrentVideo(desiredOn) {
            if (!currentVideo) return;
            if (detectionToggleInFlight) return;
            detectionToggleInFlight = true;
            try {
                // lock UI pendant l'action (évite états incohérents)
                startDetectionBtn.disabled = true;
                stopAllBtn.disabled = true;

                // backend: appels explicites (plus fiable que dépendre du GET /api/stream/*)
                if (desiredOn) {
                    await fetch(`/api/stream/${encodeURIComponent(currentVideo)}/start`, { method: 'POST' });
                    presenceOkTsByVideo[currentVideo] = 0;
                    markVideoRunStart(currentVideo);
                    ensureZoneLive(currentVideo).lastTs = Date.now();
                } else {
                    await fetch(`/api/stream/${encodeURIComponent(currentVideo)}/stop`, { method: 'POST' });
                    presenceOkTsByVideo[currentVideo] = 0;
                    markVideoRunStop(currentVideo);
                    if (zoneLiveTimersByVideo?.[currentVideo]) zoneLiveTimersByVideo[currentVideo].lastTs = 0;
                }

                // source of truth: /api/streams
                await updateActiveStreams();
                isCurrentVideoStreaming = activeVideoStreams.has(currentVideo);

                await applyDetectionUiForCurrentVideo(isCurrentVideoStreaming);
            } catch (e) {
                console.error('Detection toggle failed:', e);
                uiAlert('Impossible de changer l\'état de détection. Réessaie.', 'Détection');
                // resync UI from backend state
                try {
                    await updateActiveStreams();
                    isCurrentVideoStreaming = currentVideo ? activeVideoStreams.has(currentVideo) : false;
                    await applyDetectionUiForCurrentVideo(isCurrentVideoStreaming);
                } catch {}
            } finally {
                // unlock UI
                startDetectionBtn.disabled = !currentVideo;
                // stopAll enabled only if at least 1 stream exists (updateActiveStreams gère aussi, mais on sécurise)
                stopAllBtn.disabled = !(activeVideoStreams && activeVideoStreams.size > 0);
                detectionToggleInFlight = false;
            }
        }

        startDetectionBtn.addEventListener('click', async () => {
            if (!currentVideo) return;
            const isOn = activeVideoStreams.has(currentVideo);
            await setDetectionForCurrentVideo(!isOn);
        });

        stopAllBtn.addEventListener('click', async () => {
            await fetch('/api/streams/stop', { method: 'POST' });

            isCurrentVideoStreaming = false;
            markAllRunsStop();
            // freeze all local ticks
            try { Object.values(zoneLiveTimersByVideo).forEach(v => { if (v) v.lastTs = 0; }); } catch {}
            try { Object.keys(presenceOkTsByVideo).forEach(v => presenceOkTsByVideo[v] = 0); } catch {}
            activeVideoStreams.clear();

            if (currentVideo) {
                videoStream.src = '';
                videoStream.classList.add('hidden');
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
                videoFrame.classList.remove('hidden');
                drawCanvas.classList.remove('hidden');

                videoFrame.onload = () => {
                    syncCanvasSize();
                    drawExistingZones();
                };
            }

            setStartDetectionButtonUi(false);

            updateSteps();
            updateStatus('ready');
            await updateActiveStreams();
        });

        async function resetZoneTimer(name) {
            const ok = await uiConfirm(`Remettre le timer de "${name}" à zéro ?`, 'Timers');
            if (!ok) return;
            await fetch(`/api/zones/reset/${encodeURIComponent(name)}`, { method: 'POST' });
            resetLocalTimersZone(currentVideo, name);
            await loadZones();
        }

        // Blur toggle
        const toggleBlurBtn = document.getElementById('toggleBlurBtn');

        function ensureRetroInner(btn) {
            if (!btn) return null;
            let inner = btn.querySelector('.retro-btn-inner');
            if (!inner) {
                inner = document.createElement('span');
                inner.className = 'retro-btn-inner';
                btn.innerHTML = '';
                btn.appendChild(inner);
            }
            return inner;
        }

        function setStartDetectionButtonUi(isOn) {
            const inner = ensureRetroInner(startDetectionBtn);
            if (!inner) return;
            if (isOn) {
                inner.innerHTML = `
                    <img class="retro-icon-img" src="/static/assets_youn/SvIcons/pause-svgrepo-com.svg" alt="">
                    Pause détection
                `;
                startDetectionBtn.classList.add('is-on');
            } else {
                inner.innerHTML = `
                    <img class="retro-icon-img" src="/static/assets_youn/SvIcons/play-svgrepo-com.svg" alt="">
                    Lancer la détection
                `;
                startDetectionBtn.classList.remove('is-on');
            }
        }

        async function updateBlurButton() {
            const res = await fetch('/api/blur');
            const data = await res.json();
            const inner = ensureRetroInner(toggleBlurBtn);
            if (!inner) return;
            if (data.enabled) {
                inner.innerHTML = `
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                    </svg>
                    Floutage: ON
                `;
                toggleBlurBtn.classList.add('is-on');
            } else {
                inner.innerHTML = `
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"/>
                    </svg>
                    Floutage: OFF
                `;
                toggleBlurBtn.classList.remove('is-on');
            }
        }

        toggleBlurBtn.addEventListener('click', async () => {
            await fetch('/api/blur/toggle', { method: 'POST' });
            await updateBlurButton();
        });

        updateBlurButton();

        async function deleteZone(name) {
            const ok = await uiConfirm(`Supprimer la zone "${name}" de cette vidéo ?`, 'Suppression');
            if (!ok) return;
            await fetch(`/api/zones/${encodeURIComponent(currentVideo)}/${encodeURIComponent(name)}`, { method: 'DELETE' });

            // Invalider caches + purger états locaux pour recalculer les KPI (conformité interzone) sur les zones restantes
            try {
                // force refetch définitions zones
                zonesDefsFetchedByVideo[currentVideo] = false;
                zonesDefsFetchTsByVideo[currentVideo] = 0;
                delete zonesCacheByVideo[currentVideo];
                zonesCacheRefreshTs = 0;

                // purge timers/présence locaux pour la zone supprimée (évite pollution des moyennes)
                const v = zoneLiveTimersByVideo?.[currentVideo];
                if (v?.zones && name in v.zones) delete v.zones[name];
                const lp = linePassByVideo?.[currentVideo];
                if (lp && name in lp) delete lp[name];
                const pr = lastPresenceByVideo?.[currentVideo];
                if (pr && name in pr) delete pr[name];
                if (selectedAsset?.zone === name) selectedAsset = null;
            } catch {}

            await loadZones();
            drawExistingZones();
        }

        document.getElementById('deleteAllZonesBtn').addEventListener('click', async () => {
            const ok = await uiConfirm('Reset all : remettre à zéro tous les compteurs et taux (occup./absence + comptage) ?', 'Reset all');
            if (!ok) return;
            await fetch('/api/zones/reset', { method: 'POST' });
            resetLocalTimersAll();
            await loadZones();
            drawExistingZones();
        });

        window.addEventListener('resize', syncCanvasSize);