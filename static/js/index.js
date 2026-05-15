// State
        let currentVideo = null;
        let currentCameraId = null;
        let currentView = 'home'; // 'home' | 'tracker'
        // DEMO mode: sites live only in memory (refresh => reset)
        const DEMO_SITES = [
            {
                name: 'Entrepôt Central',
                location: 'Lyon',
                cameras: [
                    { id: 'cam1', name: 'Entrepôt', hint: 'Déchargement & présence', video: 'entr1.mp4' },
                    { id: 'cam2', name: 'Comptage A', hint: 'Ligne de comptage', video: 'w1.mp4' },
                    { id: 'cam6', name: 'Comptage B', hint: 'Ligne de comptage', video: 'w2.mp4' }
                ]
            },
            {
                name: 'Entrée',
                location: 'Lyon',
                cameras: [
                    { id: 'cam5', name: 'Accueil', hint: 'Contrôle d\'accès', video: 'video_04.mp4' }
                ]
            },
            {
                name: 'Gallerie Voltaire',
                location: 'Paris',
                cameras: [
                    { id: 'cam3', name: 'Hall Principal', hint: 'Surveillance flux visiteurs', video: 'Mall1.mp4' },
                    { id: 'cam4', name: 'Galerie Est', hint: 'Comptage & présence', video: 'Mall2.mp4' }
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
        // Counting module state (backed by server)
        const countingStateByVideo = {};   // { [videoName]: { configured, zone_name, direction, enabled, count, reversed } }
        const presencePreviewsCollapsedByVideo = {}; // { [videoName]: { [zoneName]: boolean } }
        const sidebarZonesCollapsedByVideo = {}; // { [videoName]: { [zoneName]: boolean } } pour replier les zones dans la sidebar
        const zonesDefsFetchTsByVideo = {}; // { [videoName]: epochMs } pour throttle /api/zones/{video}
        const zonesDefsFetchedByVideo = {}; // { [videoName]: boolean } pour distinguer "0 zones" vs "pas encore fetch"
        const presenceOkTsByVideo = {};     // { [videoName]: epochMs } dernier /api/presence OK (anti-stale)
        let loadZonesInFlight = false;
        let loadZonesLoopTimer = null;
        const PRESENCE_POLL_ACTIVE_MS = 1200;  // 1.2s polling when detecting
        const PRESENCE_POLL_IDLE_MS = 5000;   // 5s polling when idle
        const ZONES_DEF_TTL_MS = 2500;
        const PRESENCE_STALE_MS = 3500; // Adjusted for slower polling rate

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
            // Also reset counting if this zone is the counting ROI
            try {
                const cs = countingStateByVideo?.[video];
                if (cs?.zone_name === zoneName && cs.enabled) {
                    fetch(`/api/counting/${encodeURIComponent(video)}/reset`, { method: 'POST' });
                }
            } catch {}
        }

        async function fetchCountingState(video) {
            if (!video) return null;
            try {
                const res = await fetch(`/api/counting/${encodeURIComponent(video)}`);
                const data = await res.json();
                countingStateByVideo[video] = data;
                return data;
            } catch { return null; }
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
        const videoLabelText = document.getElementById('videoLabelText');
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
        const trackerBackBtn = document.getElementById('trackerBackBtn');
        const trackerBreadcrumb = document.getElementById('trackerBreadcrumb');

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

            // Build cameras array from backend cameras
            const syncedCameras = [];
            for (const [backendId, camData] of Object.entries(backendCameras)) {
                const existing = camerasSite?.cameras?.find(c => c.backendCameraId === backendId);
                if (existing) {
                    syncedCameras.push(existing);
                } else {
                    const camType = camData.type;
                    syncedCameras.push({
                        id: backendId,
                        name: camData.name || backendId,
                        hint: camType === 'webcam' ? 'Webcam' : 'RTSP',
                        sourceType: camType,
                        backendCameraId: backendId
                    });
                }
            }

            // Only create/keep the "Caméras" site if there are actual backend cameras
            if (syncedCameras.length === 0) {
                // Remove stale empty site if it exists
                const idx = sitesCache.findIndex(s => s.name === CAMERAS_SITE_NAME);
                if (idx !== -1) sitesCache.splice(idx, 1);
                return;
            }

            if (!camerasSite) {
                camerasSite = { name: CAMERAS_SITE_NAME, location: 'Local', cameras: [] };
                sitesCache.push(camerasSite);
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
            toolCountLineBtn.classList.toggle('active', tool === 'countingROI');
            toolIncludeBtn.classList.toggle('active', tool === 'include');
            toolExcludeBtn.classList.toggle('active', tool === 'exclude');
            editorState.lineDirEnd = null;
            editorGuide.textContent =
                tool === 'select'
                    ? "Sélection: cliquez un point (hit zone large) puis glissez pour déplacer. Shift + clic près d'une arête = ajouter un point."
                    : tool === 'countingROI'
                        ? "ROI Comptage: dessinez un polygone (3+ points) délimitant la zone du convoyeur. La ligne de comptage sera à 75%. Puis Sauver."
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
                : `<div style="color: rgba(255,255,255,0.30); font-size: 12px;">Aucune zone</div>`;
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

            /* Auto-select first zone if one exists, so polygons show immediately */
            if (editorState.zone) {
                editorSelectZone(editorState.zone);
            } else {
                editorRenderZoneList();
                editorUpdateToolbarEnabled();
            }

            editorOverlay.classList.remove('hidden');
            editorState.open = true;
            // sync canvas display size after image load
            editorFrame.onload = () => {
                const r = editorFrame.getBoundingClientRect();
                editorCanvas.style.width = r.width + 'px';
                editorCanvas.style.height = r.height + 'px';
                /* Re-render with the selected zone's polygons now that canvas is sized */
                if (editorState.zone) editorSelectZone(editorState.zone);
                else editorRender();
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

                    editorCtx.beginPath();
                    editorCtx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) editorCtx.lineTo(poly[i][0], poly[i][1]);
                    editorCtx.closePath();
                    editorCtx.fillStyle = c.fill;
                    editorCtx.fill();
                    editorCtx.strokeStyle = c.stroke;
                    editorCtx.lineWidth = isActive ? 4 : (isGhost ? 1.6 : 2);
                    editorCtx.stroke();
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
                    const t = editorState.tool === 'countingROI' ? 'countingROI' : (editorState.tool === 'exclude' ? 'exclude' : 'include');
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
                        editorCtx.fillStyle = i === 0 ? '#1d5bff' : '#22c55e';
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
                        editorCtx.fillStyle = '#1d5bff';
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

            let best = null; // { idx, kind, vIdx?, point?, d2? }

            // 1/2/3) Proximité (sommets, arêtes)
            for (let idx = polys.length - 1; idx >= 0; idx--) {
                const poly = polys[idx];
                if (!poly || poly.length < 3) continue;

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

            if (editorState.tool === 'include' || editorState.tool === 'exclude' || editorState.tool === 'countingROI') {
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
                if (editorState.tool === 'countingROI') {
                    // Auto-create a "ROI Comptage" zone for counting
                    const autoName = 'ROI Comptage';
                    if (!editorState.zones[autoName]) {
                        editorState.zones[autoName] = { polygons: [] };
                    }
                    editorState.zone = autoName;
                    editorRenderZoneList();
                } else {
                    const keys = Object.keys(editorState.zones || {}).sort();
                    if (keys.length >= 1) {
                        editorSelectZone(keys[0]);
                    } else {
                        uiAlert('Créez d\'abord une zone à droite (bouton +), puis cliquez sur \"Sauvegarder\".', 'Sauvegarde');
                        try { editorNewZoneName?.focus(); } catch {}
                        return;
                    }
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
            if (editorState.tool === 'countingROI' && editorState.points.length >= 3) {
                const poly = clonePoints(editorState.points);
                editorState.points = [];
                editorRender();
                // Save as include zone AND auto-configure as counting ROI
                await editorPostNewPolygon('countingROI', poly);
                // Auto-configure counting for this zone (direction auto-computed from polygon)
                const zoneName = editorState.zone;
                if (zoneName && currentVideo) {
                    await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/config`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ zone_name: zoneName })
                    });
                }
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
        toolCountLineBtn.addEventListener('click', () => {
            editorSetTool('countingROI');
            editorState.points = [];
            editorRender();
        });
        toolIncludeBtn.addEventListener('click', () => { editorSetTool('include'); editorState.points = []; editorRender(); });
        toolExcludeBtn.addEventListener('click', () => { editorSetTool('exclude'); editorState.points = []; editorRender(); });
        toolUndoBtn.addEventListener('click', () => {
            // Annulation "intelligente" façon paint:
            // - si on trace, on retire le dernier point
            // - sinon on annule le dernier snapshot
            if ((editorState.tool === 'include' || editorState.tool === 'exclude' || editorState.tool === 'countingROI') && editorState.points.length) {
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

        function drawArrow(ctx, from, to, color = '#1d5bff', opts = {}) {
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
            /* Invalidate sidebar HTML caches so next render always updates DOM */
            _lastSitesSidebarHTML = '';
            _lastAssetTreeHTML = '';

            // Hide all views
            homeView.classList.add('hidden');
            trackerView.classList.add('hidden');
            const analyticsView = document.getElementById('analyticsView');
            const logsView = document.getElementById('logsView');
            if (analyticsView) analyticsView.classList.add('hidden');
            if (logsView) logsView.classList.add('hidden');

            // Deactivate all nav
            navSites.classList.remove('active');
            navTracker.classList.remove('active');
            const navAnalytics = document.getElementById('navAnalytics');
            const navLogs = document.getElementById('navLogs');
            if (navAnalytics) navAnalytics.classList.remove('active');
            if (navLogs) navLogs.classList.remove('active');

            if (stepsEl) stepsEl.style.display = (view === 'tracker') ? '' : 'none';

            switch (view) {
                case 'home':
                    homeView.classList.remove('hidden');
                    navSites.classList.add('active');
                    pageTitleEl.textContent = 'Sites';
                    pageSubtitleEl.textContent = 'Choisissez un site ou créez-en un pour démarrer la configuration';
                    renderSitesSidebar();
                    break;
                case 'tracker':
                    trackerView.classList.remove('hidden');
                    navTracker.classList.add('active');
                    const siteLabel = currentSite?.name ? ` — ${currentSite.name}` : '';
                    pageTitleEl.textContent = `Zone Presence Tracker${siteLabel}`;
                    pageSubtitleEl.textContent = 'Surveillance et analyse du temps de présence';
                    updateTrackerBreadcrumb();
                    renderSidebarForTracker();
                    break;
                case 'analytics':
                    if (analyticsView) analyticsView.classList.remove('hidden');
                    if (navAnalytics) navAnalytics.classList.add('active');
                    pageTitleEl.textContent = 'Analytics';
                    pageSubtitleEl.textContent = 'Données et indicateurs de performance';
                    initAnalyticsDashboard();
                    break;
                case 'logs':
                    if (logsView) logsView.classList.remove('hidden');
                    if (navLogs) navLogs.classList.add('active');
                    pageTitleEl.textContent = 'Log / Historique';
                    pageSubtitleEl.textContent = 'Journal d\'audit et historique des événements';
                    loadPerfMetrics();
                    loadLogs();
                    startLogAutoRefresh();
                    break;
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

        async function loadSites() {
            await refreshZonesCacheForSites();
            renderSitesHome();
            renderSitesSidebar();
            /* Keep location LOV in sync */
            _syncLocationLov();
        }

        function _syncLocationLov() {
            const sel = document.getElementById('newSiteLocation');
            if (!sel) return;
            const locs = [...new Set((sitesCache || []).map(s => s.location).filter(Boolean))].sort();
            const prev = sel.value;
            let html = '<option value="" disabled>Lieu…</option>';
            for (const l of locs) html += `<option value="${l}">${l}</option>`;
            html += '<option value="__new__">+ Nouveau lieu…</option>';
            sel.innerHTML = html;
            if (prev && locs.includes(prev)) sel.value = prev; else sel.selectedIndex = 0;
            if (typeof LovDropdown !== 'undefined') LovDropdown.refresh();
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

        let _sitesViewMode = 'grid'; // 'grid' | 'list'

        function countSiteForms(site) {
            let count = 0;
            for (const cam of (site.cameras || [])) {
                const video = cam.video;
                if (!video) continue;
                const videoZones = zones_by_video_cache?.[video];
                if (videoZones) {
                    for (const zName of Object.keys(videoZones)) {
                        const z = videoZones[zName];
                        count += (z.polygons || []).length;
                    }
                }
            }
            return count;
        }

        // Lightweight zone cache for counting forms
        let zones_by_video_cache = {};
        async function refreshZonesCacheForSites() {
            try {
                // Fetch zones for all videos used by sites
                const videos = new Set();
                for (const s of (sitesCache || [])) {
                    for (const c of (s.cameras || [])) {
                        if (c.video) videos.add(c.video);
                    }
                }
                for (const v of videos) {
                    try {
                        const res = await fetch(`/api/zones/${encodeURIComponent(v)}`);
                        if (res.ok) {
                            const data = await res.json();
                            zones_by_video_cache[v] = data.zones || {};
                        }
                    } catch {}
                }
            } catch {}
        }

        function renderSitesHome() {
            const container = document.getElementById('sitesContainer');
            if (!container) return;

            if (!sitesCache || sitesCache.length === 0) {
                container.innerHTML = '<div class="no-zones">Aucun site. Créez-en un pour commencer.</div>';
                return;
            }

            if (_sitesViewMode === 'grid') {
                renderSitesGrid(container);
            } else {
                renderSitesList(container);
            }
        }

        /* hierarchy icon helpers */
        const _hierIcon = (src, cls) => `<img src="/static/assets_youn/SvIcons/${src}" class="hier-icon${cls ? ' ' + cls : ''}" alt="">`;
        const ICON_LIEU  = _hierIcon('Lieux.svg');
        const ICON_SITE  = _hierIcon('Site.svg');
        const ICON_CAM   = _hierIcon('camera.svg');
        const ICON_ZONE  = _hierIcon('zone.svg');

        function renderSitesGrid(container) {
            /* Group by location first */
            const groups = {};
            for (const s of sitesCache) {
                const loc = s.location || 'Sans lieu';
                if (!groups[loc]) groups[loc] = [];
                groups[loc].push(s);
            }

            let html = '';
            for (const [loc, sites] of Object.entries(groups)) {
                const totalCams = sites.reduce((n, s) => n + (s.cameras || []).length, 0);
                const totalForms = sites.reduce((n, s) => n + countSiteForms(s), 0);

                const _lgCol = getLocColor(loc);
                html += `<div class="loc-group">
                    <div class="loc-group__header" style="--card-loc-color:${_lgCol};">
                        <span class="sb-loc__sq" style="background:${_lgCol};width:10px;height:10px;border-radius:1px;flex:0 0 10px;"></span>
                        <span class="loc-group__name">${escapeHtml(loc)}</span>
                        <span class="loc-group__badges">
                            <span class="loc-group__badge">${ICON_SITE}<strong>${sites.length}</strong> site${sites.length > 1 ? 's' : ''}</span>
                            <span class="loc-group__badge">${ICON_CAM}<strong>${totalCams}</strong> cam${totalCams > 1 ? 's' : ''}</span>
                            <span class="loc-group__badge">${ICON_ZONE}<strong>${totalForms}</strong> zone${totalForms > 1 ? 's' : ''}</span>
                        </span>
                    </div>
                    <div class="loc-group__cards">`;

                for (const s of sites) {
                    const cams = s.cameras || [];
                    const siteName = escapeHtml(s.name);
                    const siteKey = encodeURIComponent(String(s.name ?? ''));
                    const forms = countSiteForms(s);
                    const _locCol = getLocColor(loc);
                    html += `
                        <div class="site-card-v2" data-site="${siteKey}" style="--card-loc-color:${_locCol};">
                            <div class="site-card-v2__top">
                                <div>
                                    <div class="site-card-v2__name">${ICON_SITE} ${siteName}</div>
                                    <div class="site-card-v2__location">${ICON_LIEU} ${escapeHtml(loc)}</div>
                                </div>
                                <div class="site-menu-wrap">
                                    <button class="site-menu-btn" type="button" title="Options">
                                        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="2.5" r="1.5"/><circle cx="8" cy="8" r="1.5"/><circle cx="8" cy="13.5" r="1.5"/></svg>
                                    </button>
                                    <div class="site-menu-panel">
                                        <button class="site-menu-item" data-site-action="open" data-site-key="${siteKey}" type="button">
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/></svg>
                                            Ouvrir
                                        </button>
                                        <button class="site-menu-item" data-site-action="rename" data-site-key="${siteKey}" type="button">
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
                                            Renommer
                                        </button>
                                        <button class="site-menu-item" data-site-action="move" data-site-key="${siteKey}" type="button">
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
                                            Changer de lieu
                                        </button>
                                        <button class="site-menu-item site-menu-item--danger" data-site-action="delete" data-site-key="${siteKey}" type="button">
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>
                                            Supprimer
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="site-card-v2__stats">
                                <span class="site-card-v2__stat">${ICON_CAM}<strong>${cams.length}</strong> Cam${cams.length > 1 ? 's' : ''}</span>
                                <span class="site-card-v2__stat">${ICON_ZONE}<strong>${forms}</strong> Zone${forms > 1 ? 's' : ''}</span>
                            </div>
                        </div>`;
                }
                html += '</div></div>';
            }
            container.innerHTML = html;
        }

        /** Site status: 'active' (detection running), 'idle' (no stream) */
        function getSiteStatus(site) {
            const cams = site.cameras || [];
            if (cams.length === 0) return 'idle';
            const hasActive = cams.some(c => {
                const vid = c.video || c.backendCameraId || '';
                return vid && activeVideoStreams.has(vid);
            });
            return hasActive ? 'active' : 'idle';
        }

        const STATUS_COLORS = { active: '#22c55e', pause: '#F08321', idle: '#6E7180' };

        function renderSitesList(container) {
            /* Group sites by location */
            const groups = {};
            for (const s of sitesCache) {
                const loc = s.location || 'Sans lieu';
                if (!groups[loc]) groups[loc] = [];
                groups[loc].push(s);
            }

            let html = '';
            for (const [loc, sites] of Object.entries(groups)) {
                const totalCams = sites.reduce((n, s) => n + (s.cameras || []).length, 0);
                const totalForms = sites.reduce((n, s) => n + countSiteForms(s), 0);

                const _listLocCol = getLocColor(loc);
                html += `<div class="sites-list-group">
                    <div class="sites-list-loc">
                        <span class="sb-loc__sq" style="background:${_listLocCol};width:8px;height:8px;border-radius:1px;flex:0 0 8px;"></span>
                        <span>${escapeHtml(loc)}</span>
                        <span class="sites-list-loc__count">${sites.length} site${sites.length > 1 ? 's' : ''}</span>
                        <span class="sites-list-loc__totals">
                            <span class="site-list-row__stat">${ICON_CAM} <strong>${totalCams}</strong> cams</span>
                            <span class="site-list-row__stat">${ICON_ZONE} <strong>${totalForms}</strong> zones</span>
                        </span>
                    </div>`;
                for (const s of sites) {
                    const cams = s.cameras || [];
                    const siteKey = encodeURIComponent(String(s.name ?? ''));
                    const forms = countSiteForms(s);
                    const status = getSiteStatus(s);
                    const statusColor = STATUS_COLORS[status] || STATUS_COLORS.idle;
                    const _lCol = getLocColor(loc);
                    html += `
                        <div class="site-list-row" data-site="${siteKey}" style="--card-loc-color:${_lCol};">
                            <span class="site-status-sq" style="background:${statusColor};" title="${status === 'active' ? 'Détection active' : status === 'pause' ? 'En pause' : 'Inactif'}"></span>
                            <div class="site-list-row__name">${escapeHtml(s.name)}</div>
                            <div class="site-list-row__stats">
                                <span class="site-list-row__stat"><strong>${cams.length}</strong> cam${cams.length > 1 ? 's' : ''}</span>
                                <span class="site-list-row__stat"><strong>${forms}</strong> zone${forms > 1 ? 's' : ''}</span>
                            </div>
                            <div class="site-menu-wrap">
                                <button class="site-menu-btn" type="button" title="Options">
                                    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="2.5" r="1.5"/><circle cx="8" cy="8" r="1.5"/><circle cx="8" cy="13.5" r="1.5"/></svg>
                                </button>
                                <div class="site-menu-panel">
                                    <button class="site-menu-item" data-site-action="open" data-site-key="${siteKey}" type="button">
                                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/></svg>
                                        Ouvrir
                                    </button>
                                    <button class="site-menu-item" data-site-action="rename" data-site-key="${siteKey}" type="button">
                                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
                                        Renommer
                                    </button>
                                    <button class="site-menu-item" data-site-action="move" data-site-key="${siteKey}" type="button">
                                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
                                        Changer de lieu
                                    </button>
                                    <button class="site-menu-item site-menu-item--danger" data-site-action="delete" data-site-key="${siteKey}" type="button">
                                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>
                                        Supprimer
                                    </button>
                                </div>
                            </div>
                        </div>`;
                }
                html += '</div>';
            }
            container.innerHTML = html;
        }

        /* ---- Sidebar: unified Russian-doll tree ---- */
        const _sidebarLabel = () => document.getElementById('sidebarSectionLabel');

        /* Palette of location colors */
        const LOC_PALETTE = ['#C8602A','#8B6F4E','#5B7065','#9A4B3F','#A68A64','#6B5B4E','#C49A6C'];
        const _locColorMap = {};
        let _locColorIdx = 0;
        function getLocColor(loc) {
            if (!_locColorMap[loc]) {
                _locColorMap[loc] = LOC_PALETTE[_locColorIdx % LOC_PALETTE.length];
                _locColorIdx++;
            }
            return _locColorMap[loc];
        }

        /* Zone colors (rotate a softer palette) */
        const ZONE_PALETTE = ['#60a5fa','#f97316','#4ade80','#c084fc','#fb7185','#2dd4bf','#facc15','#818cf8'];

        /* ---- Tracker breadcrumb helper ---- */
        function updateTrackerBreadcrumb() {
            if (!trackerBreadcrumb) return;
            const sep = '<span class="bc-sep">/</span>';
            const parts = [];
            if (currentSite) {
                const loc = currentSite.location || '';
                if (loc) {
                    parts.push(`<span class="bc-part bc-clickable" data-bc-action="home" title="Retour aux sites">${ICON_LIEU} ${escapeHtml(loc)}</span>`);
                }
                parts.push(`<span class="bc-part bc-clickable" data-bc-action="site" data-bc-site="${escapeHtml(currentSite.name)}" title="${escapeHtml(currentSite.name)}">${ICON_SITE} ${escapeHtml(currentSite.name)}</span>`);
            }
            if (currentCameraId) {
                const cam = getCameraById(currentCameraId);
                if (cam) {
                    parts.push(`<span class="bc-part bc-current">${ICON_CAM} ${escapeHtml(cam.name || cam.id)}</span>`);
                }
            }
            trackerBreadcrumb.innerHTML = parts.join(sep);

            /* Click delegation on breadcrumb parts */
            trackerBreadcrumb.onclick = (e) => {
                const el = e.target?.closest?.('[data-bc-action]');
                if (!el) return;
                const action = el.dataset.bcAction;
                if (action === 'home') {
                    setView('home');
                } else if (action === 'site') {
                    /* Already on the site — could reload or stay */
                    const name = el.dataset.bcSite;
                    if (name) selectSiteByName(name);
                }
            };
        }

        /* Anti-flicker for sites sidebar */
        let _lastSitesSidebarHTML = '';

        function renderSitesSidebar() {
            if (!zoneListSidebar) return;
            const lbl = _sidebarLabel();
            if (lbl) lbl.textContent = 'Explorer';

            if (!sitesCache || sitesCache.length === 0) {
                const empty = '<div class="sb-empty">Aucun site</div>';
                if (_lastSitesSidebarHTML !== empty) {
                    _lastSitesSidebarHTML = empty;
                    zoneListSidebar.innerHTML = empty;
                }
                return;
            }

            /* Group by location */
            const groups = {};
            for (const s of sitesCache) {
                const loc = s.location || 'Sans lieu';
                if (!groups[loc]) groups[loc] = [];
                groups[loc].push(s);
            }

            const isTracker = currentView === 'tracker';
            let html = '';

            for (const [loc, sites] of Object.entries(groups)) {
                const locColor = getLocColor(loc);

                html += `<div class="sb-group">
                    <div class="sb-loc">
                        <span class="sb-loc__sq" style="background:${locColor};"></span>
                        <span>${escapeHtml(loc)}</span>
                    </div>`;

                for (const s of sites) {
                    const siteKey = encodeURIComponent(String(s.name ?? ''));
                    const status = getSiteStatus(s);
                    const sColor = STATUS_COLORS[status] || STATUS_COLORS.idle;
                    const cams = s.cameras || [];
                    const isSiteOpen = isTracker && currentSite?.name === s.name;

                    html += `
                        <div class="sb-item sb-item--site${isSiteOpen ? ' sb-item--active' : ''}" data-site="${siteKey}" style="--sb-loc-color:${locColor};">
                            <span class="site-status-sq" style="background:${sColor};"></span>
                            ${ICON_SITE}
                            <span class="sb-item__name">${escapeHtml(s.name)}</span>
                            <span class="sb-item__meta">${cams.length}</span>
                        </div>`;

                    /* If this site is open in tracker, show cameras + zones */
                    if (isSiteOpen) {
                        for (const cam of cams) {
                            const vid = cam.video || cam.backendCameraId || '';
                            const isStreaming = vid && activeVideoStreams.has(vid);
                            const camColor = isStreaming ? STATUS_COLORS.active : STATUS_COLORS.idle;
                            const isCamActive = cam.id === currentCameraId;

                            const camZones = zones_by_video_cache?.[vid] || zonesCacheByVideo?.[vid] || {};
                            const zoneNames = Object.keys(camZones);

                            html += `
                                <div class="sb-item sb-item--cam${isCamActive ? ' sb-item--active' : ''}" data-cam-id="${cam.id}" style="--sb-loc-color:${locColor};">
                                    <span class="site-status-sq" style="background:${camColor};"></span>
                                    ${ICON_CAM}
                                    <span class="sb-item__name">${escapeHtml(cam.name || cam.id)}</span>
                                    <span class="sb-item__meta">${zoneNames.length}</span>
                                </div>`;

                            /* Zones under this camera (always shown when site is open) */
                            let zi = 0;
                            for (const zn of zoneNames) {
                                const zColor = ZONE_PALETTE[zi % ZONE_PALETTE.length];
                                zi++;
                                html += `
                                    <div class="sb-item sb-item--zone${isCamActive ? '' : ' sb-item--dim'}" data-zone-name="${escapeHtml(zn)}" style="--sb-loc-color:${locColor};">
                                        <span class="sb-zone-sq" style="background:${zColor};"></span>
                                        ${ICON_ZONE}
                                        <span class="sb-item__name">${escapeHtml(zn)}</span>
                                    </div>`;
                            }
                        }
                    }
                }
                html += '</div>';
            }

            /* Anti-flicker: skip DOM update if nothing changed */
            if (html !== _lastSitesSidebarHTML) {
                _lastSitesSidebarHTML = html;
                zoneListSidebar.innerHTML = html;
            }

            /* Unified click delegation */
            zoneListSidebar.onclick = (e) => {
                /* Zone click */
                const zoneEl = e.target?.closest?.('[data-zone-name]');
                if (zoneEl) {
                    /* Could scroll to zone in editor in the future */
                    return;
                }
                /* Camera click */
                const camEl = e.target?.closest?.('[data-cam-id]');
                if (camEl) {
                    const camId = camEl.getAttribute('data-cam-id');
                    if (camId) selectCamera(camId);
                    return;
                }
                /* Site click */
                const siteEl = e.target?.closest?.('[data-site]');
                if (siteEl) {
                    const name = decodeURIComponent(siteEl.getAttribute('data-site') || '');
                    selectSiteByName(name);
                    return;
                }
            };
        }

        function renderSidebarForHome() { renderSitesSidebar(); }
        function renderSidebarForTracker() { renderSitesSidebar(); }

        function createSite(name, location) {
            const n = String(name || '').trim();
            if (!n) throw new Error('Nom de site requis');
            if ((sitesCache || []).some(s => s.name === n)) throw new Error('Site déjà existant');
            const site = { name: n, cameras: [] };
            const loc = String(location || '').trim();
            if (loc) site.location = loc;
            sitesCache.push(site);
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

        // ==================== Performance Monitor Chart ====================
        let _perfData = null;       // cached metrics response
        let _perfHidden = new Set(["YOLO Inference", "FPS", "Active Detections"]); // hidden by default

        async function loadPerfMetrics() {
            try {
                const res = await fetch('/api/metrics?points=120');
                _perfData = await res.json();
                renderPerfLegends();
                drawPerfChart();
            } catch (e) {
                console.warn('Failed to load metrics:', e);
            }
        }

        function renderPerfLegends() {
            const container = document.getElementById('perfLegends');
            if (!container || !_perfData) return;

            container.innerHTML = '';
            for (const [name, series] of Object.entries(_perfData)) {
                const vals = series.data.map(d => d.v);
                const min = Math.round(Math.min(...vals));
                const max = Math.round(Math.max(...vals));
                const unit = series.unit || '';
                const isOff = _perfHidden.has(name);

                const el = document.createElement('div');
                el.className = 'perf-legend' + (isOff ? ' is-off' : '');
                el.style.color = series.color;
                el.innerHTML = `
                    <span class="perf-legend-check"></span>
                    <span class="perf-legend-label">${name} [${min}${unit ? ' ' + unit : ''} – ${max}${unit ? ' ' + unit : ''}]</span>
                `;
                el.addEventListener('click', () => {
                    if (_perfHidden.has(name)) _perfHidden.delete(name);
                    else _perfHidden.add(name);
                    renderPerfLegends();
                    drawPerfChart();
                });
                container.appendChild(el);
            }
        }

        function hexToRgba(hex, a) {
            const r = parseInt(hex.slice(1,3),16);
            const g = parseInt(hex.slice(3,5),16);
            const b = parseInt(hex.slice(5,7),16);
            return `rgba(${r},${g},${b},${a})`;
        }

        function drawPerfChart() {
            const canvas = document.getElementById('perfCanvas');
            if (!canvas || !_perfData) return;

            const wrap = canvas.parentElement;
            const dpr = window.devicePixelRatio || 1;
            const w = wrap.clientWidth;
            const h = wrap.clientHeight;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);

            // Background
            ctx.fillStyle = '#0A0E13';
            ctx.fillRect(0, 0, w, h);

            // Horizontal grid
            const gridLines = 4;
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.lineWidth = 1;
            for (let i = 1; i < gridLines; i++) {
                const y = Math.round((h / gridLines) * i) + 0.5;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            // Vertical time grid
            const sampleLen = Object.values(_perfData)[0]?.data?.length || 200;
            const gridStep = Math.max(1, Math.floor(sampleLen / 8));
            ctx.strokeStyle = 'rgba(255,255,255,0.03)';
            for (let i = gridStep; i < sampleLen; i += gridStep) {
                const x = Math.round((i / (sampleLen - 1)) * w) + 0.5;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }

            // Draw each series as STEP chart (staircase / cranté)
            const pad = 4;
            const drawH = h - pad * 2;

            for (const [name, series] of Object.entries(_perfData)) {
                if (_perfHidden.has(name)) continue;
                const data = series.data;
                if (!data || data.length < 2) continue;

                const sMin = series.min ?? 0;
                const sMax = series.max ?? 100;
                const range = sMax - sMin || 1;

                const toY = (v) => h - pad - ((v - sMin) / range) * drawH;

                // Build step path
                ctx.beginPath();
                let prevY = toY(data[0].v);
                ctx.moveTo(0, prevY);

                for (let i = 1; i < data.length; i++) {
                    const x = (i / (data.length - 1)) * w;
                    const y = toY(data[i].v);
                    // Horizontal line to new x at old y (step), then vertical jump
                    ctx.lineTo(x, prevY);
                    ctx.lineTo(x, y);
                    prevY = y;
                }
                // Extend to right edge
                ctx.lineTo(w, prevY);

                // Stroke the step line
                ctx.strokeStyle = series.color;
                ctx.lineWidth = 1.5;
                ctx.lineJoin = 'miter';
                ctx.stroke();

                // Fill under the step curve
                ctx.lineTo(w, h);
                ctx.lineTo(0, h);
                ctx.closePath();
                ctx.fillStyle = hexToRgba(series.color, 0.05);
                ctx.fill();
            }
        }

        // Resize handler for chart
        window.addEventListener('resize', () => {
            if (currentView === 'logs' && _perfData) drawPerfChart();
        });

        // ==================== Log / Historique ====================
        let _logAutoRefreshTimer = null;
        let _logCurrentFilter = '';
        let _logLastHash = '';  // avoid DOM thrashing on unchanged data

        function formatLogTimestamp(isoStr) {
            try {
                const d = new Date(isoStr);
                const pad = (n) => String(n).padStart(2, '0');
                return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
            } catch { return isoStr; }
        }

        async function loadLogs() {
            const consoleInner = document.getElementById('logConsoleInner');
            const logCount = document.getElementById('logCount');
            if (!consoleInner) return;
            try {
                const url = _logCurrentFilter
                    ? `/api/logs?limit=500&category=${encodeURIComponent(_logCurrentFilter)}`
                    : '/api/logs?limit=500';
                const res = await fetch(url);
                const data = await res.json();
                const logs = data.logs || [];

                // Quick hash to skip DOM updates when nothing changed
                const hash = `${data.total}:${logs.length}:${logs[0]?.ts || ''}`;
                if (hash === _logLastHash) return;
                _logLastHash = hash;

                if (logCount) logCount.textContent = `${data.total || logs.length} événement(s)`;

                if (logs.length === 0) {
                    consoleInner.innerHTML = '<div class="log-empty">Aucun événement enregistré</div>';
                    return;
                }

                let html = '';
                logs.forEach(entry => {
                    const ts = formatLogTimestamp(entry.ts);
                    const level = entry.level || 'info';
                    const cat = entry.category || 'system';
                    const action = entry.action || '';
                    const detail = entry.detail || '';
                    html += `<div class="log-entry">
                        <span class="log-ts">${ts}</span>
                        <span class="log-level log-level--${level}"></span>
                        <span class="log-cat log-cat--${cat}">${cat}</span>
                        <span class="log-action">${action}</span>
                        <span class="log-detail">${detail}</span>
                    </div>`;
                });
                consoleInner.innerHTML = html;
            } catch (e) {
                consoleInner.innerHTML = '<div class="log-empty">Erreur de chargement des logs</div>';
            }
        }

        function startLogAutoRefresh() {
            stopLogAutoRefresh();
            _logAutoRefreshTimer = setInterval(() => {
                if (currentView === 'logs') loadLogs();
            }, 3000);
        }

        function stopLogAutoRefresh() {
            if (_logAutoRefreshTimer) {
                clearInterval(_logAutoRefreshTimer);
                _logAutoRefreshTimer = null;
            }
        }

        // Log filter buttons
        document.getElementById('logFilters')?.addEventListener('click', (e) => {
            const btn = e.target.closest('.log-filter-btn');
            if (!btn) return;
            document.querySelectorAll('.log-filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _logCurrentFilter = btn.dataset.cat || '';
            _logLastHash = '';  // force refresh on filter change
            loadLogs();
        });

        document.getElementById('refreshLogsBtn')?.addEventListener('click', () => loadLogs());

        document.getElementById('clearLogsBtn')?.addEventListener('click', async () => {
            if (!confirm('Effacer tout le journal d\'audit ?')) return;
            await fetch('/api/logs', { method: 'DELETE' });
            loadLogs();
        });

        // Initialize
        init();

        async function init() {
            // Important: récupérer d'abord les streams, puis construire l'UI (évite les resets)
            await updateActiveStreams();
            await loadBackendCameras();
            loadSites();

            // Wrap all selects with custom LovDropdown
            if (window.LovDropdown) {
                LovDropdown.wrapAll('select');
            }

            // Nav
            navSites?.addEventListener('click', () => setView('home'));
            trackerBackBtn?.addEventListener('click', () => setView('home'));
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
            document.getElementById('navAnalytics')?.addEventListener('click', () => setView('analytics'));
            document.getElementById('navLogs')?.addEventListener('click', () => {
                setView('logs');
            });

            // View toggle (grid / list)
            document.getElementById('sitesViewGrid')?.addEventListener('click', () => {
                _sitesViewMode = 'grid';
                document.getElementById('sitesViewGrid')?.classList.add('active');
                document.getElementById('sitesViewList')?.classList.remove('active');
                renderSitesHome();
            });
            document.getElementById('sitesViewList')?.addEventListener('click', () => {
                _sitesViewMode = 'list';
                document.getElementById('sitesViewList')?.classList.add('active');
                document.getElementById('sitesViewGrid')?.classList.remove('active');
                renderSitesHome();
            });

            // Create site — location LOV
            const newSiteLocationSelect = document.getElementById('newSiteLocation');

            /* Handle "Nouveau lieu" choice */
            newSiteLocationSelect?.addEventListener('change', () => {
                if (newSiteLocationSelect.value === '__new__') {
                    const newLoc = prompt('Nom du nouveau lieu :');
                    if (newLoc && newLoc.trim()) {
                        const val = newLoc.trim();
                        /* Add temp option */
                        const opt = document.createElement('option');
                        opt.value = val; opt.textContent = val;
                        newSiteLocationSelect.insertBefore(opt, newSiteLocationSelect.lastElementChild);
                        newSiteLocationSelect.value = val;
                    } else {
                        newSiteLocationSelect.selectedIndex = 0;
                    }
                    if (typeof LovDropdown !== 'undefined') LovDropdown.refresh();
                }
            });

            async function handleCreateSite() {
                const name = (newSiteNameInput?.value || '').trim();
                if (!name) return;
                const location = (newSiteLocationSelect?.value || '').trim();
                if (!location || location === '__new__') {
                    uiAlert('Sélectionnez un lieu.', 'Sites');
                    return;
                }
                try {
                    createSite(name, location);
                    newSiteNameInput.value = '';
                    newSiteLocationSelect.selectedIndex = 0;
                    if (typeof LovDropdown !== 'undefined') LovDropdown.refresh();
                    await loadSites();
                } catch (e) {
                    uiAlert(`Erreur création site: ${e?.message || e}`, 'Sites');
                }
            }
            createSiteBtn?.addEventListener('click', handleCreateSite);
            newSiteNameInput?.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') handleCreateSite();
            });

            /* Init location LOV */
            _syncLocationLov();

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
                // Toggle: if already open, close it
                if (addCameraForm && !addCameraForm.classList.contains('hidden')) {
                    closeAddCameraForm();
                } else {
                    openAddCameraForm();
                }
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

            /* ---- Site context menu (three-dot) ---- */
            function closeSiteMenus() {
                document.querySelectorAll('.site-menu-wrap.is-open').forEach(w => w.classList.remove('is-open'));
            }
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.site-menu-wrap')) closeSiteMenus();
            });

            // Click delegation for site cards/rows
            const sitesContainer = document.getElementById('sitesContainer');
            sitesContainer?.addEventListener('click', async (e) => {
                /* Three-dot toggle */
                const menuBtn = e.target?.closest?.('.site-menu-btn');
                if (menuBtn) {
                    e.stopPropagation();
                    const wrap = menuBtn.closest('.site-menu-wrap');
                    const wasOpen = wrap.classList.contains('is-open');
                    closeSiteMenus();
                    if (!wasOpen) wrap.classList.add('is-open');
                    return;
                }

                /* Menu action */
                const actionBtn = e.target?.closest?.('[data-site-action]');
                if (actionBtn) {
                    e.stopPropagation();
                    const action = actionBtn.getAttribute('data-site-action');
                    const key = actionBtn.getAttribute('data-site-key') || '';
                    const name = decodeURIComponent(key);
                    closeSiteMenus();

                    if (action === 'delete') {
                        const ok = await uiConfirm(`Supprimer le site "${name}" ?`, 'Suppression');
                        if (ok) {
                            try { deleteSiteByName(name); loadSites(); } catch (err) { uiAlert(err?.message || String(err), 'Suppression'); }
                        }
                    } else if (action === 'open') {
                        selectSiteByName(name);
                    } else if (action === 'rename') {
                        const newName = prompt(`Renommer "${name}" en :`, name);
                        if (newName && newName.trim() && newName.trim() !== name) {
                            const site = sitesCache.find(s => s.name === name);
                            if (site) { site.name = newName.trim(); loadSites(); }
                        }
                    } else if (action === 'move') {
                        const site = sitesCache.find(s => s.name === name);
                        if (!site) return;
                        const locs = [...new Set((sitesCache || []).map(s => s.location).filter(Boolean))].sort();
                        const choices = locs.join(', ');
                        const newLoc = prompt(`Déplacer "${name}" vers quel lieu ?\nLieux existants : ${choices}\n(ou entrez un nouveau lieu)`, site.location || '');
                        if (newLoc && newLoc.trim()) {
                            site.location = newLoc.trim();
                            loadSites();
                        }
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

            /* ── Smooth 1s ticker: interpolates local timers + patches DOM in-place ── */
            setInterval(() => {
                if (currentView !== 'tracker' || !currentVideo) return;
                if (!activeVideoStreams.has(currentVideo)) return;
                const vid = currentVideo;
                const v = zoneLiveTimersByVideo?.[vid];
                if (!v || !v.zones) return;

                /* Advance local timers by 1s based on last known occupancy */
                const now = Date.now();
                const last = Number(v.lastTs || 0);
                if (!last) return;
                const dt = Math.max(0, (now - last) / 1000);
                if (dt > 5) return; // stale, wait for real poll
                v.lastTs = now;
                for (const name of Object.keys(v.zones)) {
                    const z = v.zones[name];
                    /* Use the last known occupancy state from presenceOkTsByVideo */
                    const lastPres = lastPresenceByVideo?.[vid]?.[name];
                    const isOcc = !!(lastPres?.is_occupied);
                    if (isOcc) z.occ += dt;
                    else z.abs += dt;
                }

                /* Patch DOM in-place: find zone-cards and update numbers + bar widths */
                const cards = document.querySelectorAll('.zone-card[data-zone]');
                for (const card of cards) {
                    const zName = decodeURIComponent(card.dataset.zone || '');
                    const z = v.zones[zName];
                    if (!z) continue;
                    const occ = z.occ || 0;
                    const abs = z.abs || 0;
                    const total = occ + abs;
                    if (total <= 0) continue;
                    const occPct = Math.min(100, (occ / total) * 100);
                    const absPct = Math.max(0, 100 - occPct);
                    const secFmt = (s) => `${Math.max(0, Math.floor(Number(s) || 0))} s`;

                    /* Update percentage texts */
                    const pcts = card.querySelectorAll('.occ-head .pct');
                    if (pcts[0]) pcts[0].textContent = `${occPct.toFixed(0)}%`;
                    if (pcts[1]) pcts[1].textContent = `${absPct.toFixed(0)}%`;

                    /* Update second labels */
                    const secs = card.querySelectorAll('.occ-head .secs');
                    if (secs[0]) secs[0].textContent = secFmt(occ);
                    if (secs[1]) secs[1].textContent = secFmt(abs);

                    /* Update bar widths (CSS transition handles smoothing) */
                    const fills = card.querySelectorAll('.occ-fill');
                    if (fills[0]) fills[0].style.width = `${occPct.toFixed(2)}%`;
                    if (fills[1]) fills[1].style.width = `${absPct.toFixed(2)}%`;
                }
            }, 1000);
        }

        const DEFAULT_CAMERA_DEFS = [
            {
                id: 'entr1',
                name: 'Entrepôt / Logistique',
                hint: 'Déchargement & présence',
                video: 'entr1.mp4'
            },
            {
                id: 'w1',
                name: 'Comptage A',
                hint: 'Ligne de comptage',
                video: 'w1.mp4'
            },
            {
                id: 'w2',
                name: 'Comptage B',
                hint: 'Ligne de comptage',
                video: 'w2.mp4'
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

            /* Refresh sidebar to highlight active cam */
            if (currentView === 'tracker') {
                updateTrackerBreadcrumb();
                renderSidebarForTracker();
            }

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
                    // Fetch counting state from backend
                    await fetchCountingState(currentVideo);
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

                // UX: previews repliées par défaut pour les cartes "présence" (zones polygones).
                const isPreviewsCollapsed = presencePreviewsCollapsedByVideo?.[currentVideo]?.[name] ?? true;

                // Check if this zone is the counting ROI
                const cs = countingStateByVideo?.[currentVideo] || {};
                const zoneSettings = cs.zone_settings || {};
                const zoneMode = zoneSettings[name]?.mode || 'simple';
                const isCountingZone = cs.zone_name === name && cs.enabled;
                const countVal = isCountingZone ? (cs.count || 0) : null;

                zonesGrid.innerHTML += `
                    <div class="zone-card ${isSelected ? 'selected' : ''} ${isPreviewsCollapsed ? 'is-previews-collapsed' : ''}" data-zone="${encodeURIComponent(String(name))}">
                        <div class="zone-card-header">
                            <div>
                                <div class="zone-name-pill">${name}${isCountingZone ? ' <span style="color:var(--color-accent);font-size:0.75em;">&#x25B6; Comptage</span>' : ''}</div>
                                <button class="zone-forms-toggle" type="button" data-zone="${encodeURIComponent(String(name))}" aria-label="Afficher/Masquer les formes">
                                    <span>${drawings} forme(s)</span>
                                    <svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                        <path d="M7 10l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    </svg>
                                </button>
                            </div>
                            <div class="zone-card-status ${statusClass}">${statusLabel}</div>
                        </div>
                        <div class="zone-previews ${isPreviewsCollapsed ? 'is-collapsed' : ''}">
                            ${previewBoxes}
                        </div>
                        ${isCountingZone ? `
                            <div class="line-kpi-row">
                                <div class="line-kpi-left">
                                    <div class="line-kpi-num" style="color:${zoneMode === 'simple' ? '#ff9600' : 'var(--color-accent)'};">${countVal}</div>
                                    <div class="line-kpi-label">${zoneMode === 'simple' ? 'Simple' : 'Complexe'}</div>
                                </div>
                                <div class="line-kpi-right">
                                    <div class="line-kpi-mini"><span>Direction</span><span>${cs.angle != null ? Math.round(cs.angle) + '°' : '—'}</span></div>
                                </div>
                            </div>
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
                                    <div class="occ-track"><div class="occ-fill blue" style="width:${absPct.toFixed(2)}%"></div></div>
                                </div>
                            </div>
                        `}
                        <div class="zone-card-actions">
                            <button class="ctrl-btn ctrl-btn--ghost ctrl-btn--sm" data-reset-zone="${escapeHtml(name)}" title="Reset" style="width:auto; padding:5px 8px;">
                                <svg class="ctrl-btn__svg" width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

            // Update counting UI
            updateCountingZoneOptions();
            const cs = countingStateByVideo?.[currentVideo];
            if (cs && countingDisplay) {
                if (cs.enabled) { countingDisplay.style.display = 'block'; countingValue.textContent = cs.count || 0; }
            }
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
            // palette: include=vert, line=bleu, exclude=orange/rouge, countingROI=cyan
            const base = {
                include: { stroke: '#22c55e', fill: 'rgba(34,197,94,0.16)' },
                line: { stroke: '#10B0F9', fill: 'rgba(16,176,249,0.12)' },
                exclude: { stroke: '#F08321', fill: 'rgba(240,131,33,0.16)' }
            }[type] || { stroke: '#22c55e', fill: 'rgba(34,197,94,0.16)' };
            if (!isActive) return base;
            return { stroke: '#1d5bff', fill: 'rgba(29,91,255,0.18)' };
        }

        /* Anti-flicker: only update sidebar DOM if content actually changed */
        let _lastAssetTreeHTML = '';

        function renderAssetTree(presenceZones, zonesWithPolygons) {
            const cams = getActiveCameras();
            if (!cams || cams.length === 0) {
                const empty = '<div class="sb-empty">Aucune caméra</div>';
                if (_lastAssetTreeHTML !== empty) {
                    _lastAssetTreeHTML = empty;
                    zoneListSidebar.innerHTML = empty;
                }
                return;
            }

            const prevScrollTop = zoneListSidebar.scrollTop || 0;

            /* Determine location color for current site */
            const loc = currentSite?.location || 'Sans lieu';
            const locColor = getLocColor(loc);

            /* Site header row */
            let html = `<div class="sb-group">
                <div class="sb-loc" style="margin-bottom:2px;">
                    <span class="sb-loc__sq" style="background:${locColor};"></span>
                    ${ICON_LIEU}
                    <span>${escapeHtml(loc)}</span>
                </div>
                <div class="sb-item sb-item--site sb-item--active" style="--sb-loc-color:${locColor};">
                    <span class="site-status-sq" style="background:${STATUS_COLORS.active};"></span>
                    ${ICON_SITE}
                    <span class="sb-item__name">${escapeHtml(currentSite?.name || '')}</span>
                    <span class="sb-item__meta">${cams.length}</span>
                </div>`;

            cams.forEach((cam, camIdx) => {
                const isCurrent = cam.video === currentVideo;
                const camLabel = escapeHtml(cam.name || cam.video);
                const videoKey = cam.video;
                const isBackend = cam.sourceType === 'webcam' || cam.sourceType === 'rtsp';
                const sourceKey = isBackend ? `camera:${cam.backendCameraId}` : cam.video;
                const isStreaming = sourceKey && activeVideoStreams.has(sourceKey);
                const camColor = isStreaming ? STATUS_COLORS.active : STATUS_COLORS.idle;

                const defs = isCurrent ? (zonesWithPolygons || {}) : (zonesCacheByVideo?.[videoKey] || {});
                const presence = isCurrent ? (presenceZones || {}) : (lastPresenceByVideo?.[videoKey] || {});
                const zoneNames = Object.keys(defs || {}).sort();
                const dimClass = !isCurrent ? ' sb-item--dim' : '';

                html += `
                    <div class="sb-item sb-item--cam${isCurrent ? ' sb-item--active' : ''}" data-select-camera="${escapeHtml(cam.id)}" style="--sb-loc-color:${locColor}; cursor:pointer;">
                        <span class="site-status-sq" style="background:${camColor};"></span>
                        ${ICON_CAM}
                        <span class="sb-item__name">${camLabel}</span>
                        <span class="sb-item__meta">${zoneNames.length}</span>
                    </div>`;

                /* Zones under ALL cameras (dim non-selected ones) */
                let zi = 0;
                for (const zoneName of zoneNames) {
                    const info = presence?.[zoneName] || { formatted_time: '00:00:00', is_occupied: false };
                    const drawings = defs?.[zoneName]?.polygons || [];
                    const zColor = ZONE_PALETTE[zi % ZONE_PALETTE.length];
                    const dotCls = info.is_occupied ? ' occupied' : '';
                    const isCollapsed = sidebarZonesCollapsedByVideo?.[videoKey]?.[zoneName] ?? true;
                    const hasDrawings = drawings.length > 0;
                    zi++;

                    html += `
                        <div class="sb-item sb-item--zone${dimClass}" data-select-zone="${escapeHtml(zoneName)}"${!isCurrent ? ` data-select-camera="${escapeHtml(cam.id)}"` : ''} style="--sb-loc-color:${locColor}; cursor:pointer;">
                            ${hasDrawings && isCurrent ? `
                                <button class="tree-toggle-btn" onclick="event.stopPropagation(); toggleSidebarZone('${videoKey}', '${zoneName}')" type="button">
                                    <svg class="tree-chevron ${isCollapsed ? 'collapsed' : 'expanded'}" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                        <path d="M9 18l6-6-6-6" stroke-linecap="round" stroke-linejoin="round"/>
                                    </svg>
                                </button>
                            ` : ''}
                            <span class="sb-zone-sq${dotCls}" style="background:${zColor};"></span>
                            ${ICON_ZONE}
                            <span class="sb-item__name">${escapeHtml(zoneName)}</span>
                            <span class="sb-item__time">${info.formatted_time}</span>
                        </div>`;

                    /* Drawings sub-items (only for current camera) */
                    if (isCurrent && hasDrawings && !isCollapsed) {
                        drawings.forEach((_, idx) => {
                            html += `
                                <div class="sb-item sb-item--drawing" data-select-drawing="${escapeHtml(zoneName)}" data-drawing-idx="${idx}" style="--sb-loc-color:${locColor}; cursor:pointer;">
                                    <span class="sb-item__name">Dessin ${idx + 1}</span>
                                </div>`;
                        });
                    }
                }
                if (zoneNames.length === 0) {
                    html += `<div class="sb-item sb-item--zone${dimClass}" style="opacity:0.35; cursor:default;">
                        <span class="sb-item__name">Aucune zone</span>
                    </div>`;
                }
            });

            html += '</div>';

            /* Anti-flicker: skip DOM update if nothing changed */
            if (html !== _lastAssetTreeHTML) {
                _lastAssetTreeHTML = html;
                zoneListSidebar.innerHTML = html;
            }
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

            /* With object-fit: contain, compute the actual rendered image size */
            const containerW = img.clientWidth;
            const containerH = img.clientHeight;
            const natW = img.naturalWidth;
            const natH = img.naturalHeight;
            const scale = Math.min(containerW / natW, containerH / natH);
            const renderedW = natW * scale;
            const renderedH = natH * scale;

            drawCanvas.style.width = renderedW + 'px';
            drawCanvas.style.height = renderedH + 'px';
        }

        // Video selection
        videoSelect.addEventListener('change', async () => {
            if (!videoSelect.value) return;

            // IMPORTANT: Stop any existing stream IMMEDIATELY to free bandwidth
            // This prevents accumulating streams when switching videos
            if (videoStream.src) {
                videoStream.src = '';
            }

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
            if (videoLabelText) videoLabelText.textContent = cam ? cam.name : currentVideo;

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
                // Set the new stream source (old one was already cleared above)
                videoStream.src = `/api/stream/${encodeURIComponent(currentVideo)}`;
                drawCanvas.classList.add('hidden');
                updateStatus('streaming');
                // MJPEG doesn't fire onload reliably — sync after short delay
                setTimeout(syncCanvasSize, 300);
                setTimeout(syncCanvasSize, 800);
            } else {
                // Not streaming - show a static frame (user must click "Lancer détection")
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
                videoFrame.classList.remove('hidden');
                videoStream.classList.add('hidden');
                // Stream src already cleared at start of handler
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
            syncCountingUI();
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
                ctx.strokeStyle = '#1d5bff';
                ctx.lineWidth = 3;
                ctx.stroke();

                editPoints.forEach((p, i) => {
                    const isActive = editDragging && editDragging.idx === i;
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], HANDLE_RADIUS, 0, Math.PI * 2);
                    ctx.fillStyle = isActive ? '#F08321' : '#1d5bff';
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
                        ctx.strokeStyle = '#1d5bff';
                        ctx.lineWidth = 2;
                        ctx.stroke();

                        // Ligne centrale
                        ctx.beginPath();
                        ctx.moveTo(p1[0], p1[1]);
                        ctx.lineTo(p2[0], p2[1]);
                        ctx.strokeStyle = '#1d5bff';
                        ctx.lineWidth = 3;
                        ctx.stroke();
                    }

                    drawPoints.forEach((p) => {
                        ctx.beginPath();
                        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
                        ctx.fillStyle = '#1d5bff';
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

                    ctx.strokeStyle = '#1d5bff';
                    ctx.lineWidth = 3;
                    ctx.stroke();

                    drawPoints.forEach((p, i) => {
                        ctx.beginPath();
                        ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
                        ctx.fillStyle = i === 0 ? '#22c55e' : '#1d5bff';
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
                // Always clear first, then set new source to avoid stale connections
                videoStream.src = '';
                videoStream.src = `/api/stream/${encodeURIComponent(currentVideo)}`;
                updateStatus('streaming');
            } else {
                // Stop stream immediately
                videoStream.src = '';
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

        function setStartDetectionButtonUi(isOn) {
            if (!startDetectionBtn) return;
            if (isOn) {
                startDetectionBtn.innerHTML = `
                    <svg class="ctrl-btn__svg" width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
                    </svg>
                    <span>Pause détection</span>
                `;
                startDetectionBtn.classList.add('is-on');
            } else {
                startDetectionBtn.innerHTML = `
                    <img class="ctrl-btn__icon" src="/static/assets_youn/SvIcons/play-svgrepo-com.svg" alt="">
                    <span>Lancer la détection</span>
                `;
                startDetectionBtn.classList.remove('is-on');
            }
        }

        async function updateBlurButton() {
            if (!toggleBlurBtn) return;
            const res = await fetch('/api/blur');
            const data = await res.json();
            if (data.enabled) {
                toggleBlurBtn.innerHTML = `
                    <svg class="ctrl-btn__svg" width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                    </svg>
                    <span>Floutage: ON</span>
                `;
                toggleBlurBtn.classList.add('is-on', 'btn-blur-on');
            } else {
                toggleBlurBtn.innerHTML = `
                    <svg class="ctrl-btn__svg" width="15" height="15" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"/>
                    </svg>
                    <span>Floutage: OFF</span>
                `;
                toggleBlurBtn.classList.remove('is-on', 'btn-blur-on');
            }
        }

        toggleBlurBtn.addEventListener('click', async () => {
            await fetch('/api/blur/toggle', { method: 'POST' });
            await updateBlurButton();
        });

        updateBlurButton();

        // ==================== Counting Module UI ====================
        const countingZoneSelect = document.getElementById('countingZoneSelect');
        const countingModeSelect = document.getElementById('countingModeSelect');
        const countingToggleBtn = document.getElementById('countingToggleBtn');
        const countingFlipBtn = document.getElementById('countingFlipBtn');
        const countingResetBtn = document.getElementById('countingResetBtn');
        const countingDisplay = document.getElementById('countingDisplay');
        const countingModeLabel = document.getElementById('countingModeLabel');
        const countingValue = document.getElementById('countingValue');
        const countingAngleLabel = document.getElementById('countingAngleLabel');
        const simpleParamsPanel = document.getElementById('simpleParamsPanel');
        const simpleThresholdSlider = document.getElementById('simpleThresholdSlider');
        const simpleThresholdValue = document.getElementById('simpleThresholdValue');
        const simpleCooldownSlider = document.getElementById('simpleCooldownSlider');
        const simpleCooldownValue = document.getElementById('simpleCooldownValue');

        function updateCountingZoneOptions() {
            if (!countingZoneSelect) return;
            const prev = countingZoneSelect.value;
            countingZoneSelect.innerHTML = '<option value="">— Aucune —</option>';
            const zones = zonesCacheByVideo[currentVideo] || {};
            for (const name of Object.keys(zones).sort()) {
                countingZoneSelect.innerHTML += `<option value="${name}">${name}</option>`;
            }
            if (prev && [...countingZoneSelect.options].some(o => o.value === prev)) {
                countingZoneSelect.value = prev;
            }
        }

        let _simpleParamsLoaded = false;
        async function syncCountingUI() {
            if (!currentVideo) {
                if (countingToggleBtn) countingToggleBtn.disabled = true;
                if (countingFlipBtn) countingFlipBtn.disabled = true;
                if (countingResetBtn) countingResetBtn.disabled = true;
                if (countingModeSelect) countingModeSelect.disabled = true;
                if (countingDisplay) countingDisplay.style.display = 'none';
                if (countingAngleLabel) countingAngleLabel.textContent = 'auto';
                if (simpleParamsPanel) simpleParamsPanel.style.display = 'none';
                return;
            }
            const data = await fetchCountingState(currentVideo);
            if (!data) return;

            updateCountingZoneOptions();

            if (data.configured && data.zone_name) {
                countingZoneSelect.value = data.zone_name;
            }

            // Restore mode selector from server zone_settings for the active zone
            const mode = data.mode || 'simple';
            if (countingModeSelect) {
                countingModeSelect.value = mode;
            }

            // Show auto-computed angle
            if (countingAngleLabel) {
                countingAngleLabel.textContent = data.angle != null ? `${Math.round(data.angle)}°` : 'auto';
            }

            const configured = !!data.configured;
            countingToggleBtn.disabled = !configured;
            if (countingFlipBtn) countingFlipBtn.disabled = !configured;
            if (countingResetBtn) countingResetBtn.disabled = !configured;
            if (countingModeSelect) countingModeSelect.disabled = !data.zone_name;
            const toggleSpan = countingToggleBtn.querySelector('span');
            if (data.enabled) {
                countingToggleBtn.classList.add('is-on');
                if (toggleSpan) toggleSpan.textContent = 'Pause Comptage';
            } else {
                countingToggleBtn.classList.remove('is-on');
                if (toggleSpan) toggleSpan.textContent = 'Activer Comptage';
            }

            // Single counter display (adapts to mode)
            if (countingDisplay) {
                countingDisplay.style.display = data.configured ? 'block' : 'none';
                if (countingModeLabel) {
                    countingModeLabel.textContent = mode === 'simple' ? 'Simple (Gradient)' : 'Complexe (MOG2)';
                }
                countingDisplay.style.color = mode === 'simple' ? '#ff9600' : 'var(--color-accent)';
                if (countingValue) countingValue.textContent = data.count || 0;
            }

            // Simple params panel — only when mode is simple
            if (simpleParamsPanel) {
                simpleParamsPanel.style.display = mode === 'simple' ? 'block' : 'none';
            }

            // Load slider values from server once
            if (!_simpleParamsLoaded && mode === 'simple') {
                _simpleParamsLoaded = true;
                try {
                    const params = await fetch('/api/counting/params').then(r => r.json());
                    if (simpleThresholdSlider) {
                        simpleThresholdSlider.value = params.simple_gradient_threshold || 30;
                        if (simpleThresholdValue) simpleThresholdValue.textContent = simpleThresholdSlider.value;
                    }
                    if (simpleCooldownSlider) {
                        simpleCooldownSlider.value = params.simple_cooldown_frames || 10;
                        if (simpleCooldownValue) simpleCooldownValue.textContent = simpleCooldownSlider.value;
                    }
                } catch {}
            }
        }

        // Mode select: save mode to server for this zone, then sync UI
        if (countingModeSelect) {
            countingModeSelect.addEventListener('change', async () => {
                const zoneName = countingZoneSelect ? countingZoneSelect.value : '';
                if (!currentVideo || !zoneName) return;
                const mode = countingModeSelect.value;
                await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ zone_name: zoneName, mode })
                });
                await syncCountingUI();
            });
        }

        // Zone select: switch active zone (preserves per-zone mode & flip from server)
        if (countingZoneSelect) {
            countingZoneSelect.addEventListener('change', async () => {
                const zoneName = countingZoneSelect.value;
                if (!currentVideo || !zoneName) {
                    if (countingToggleBtn) countingToggleBtn.disabled = true;
                    if (countingFlipBtn) countingFlipBtn.disabled = true;
                    if (countingResetBtn) countingResetBtn.disabled = true;
                    if (countingModeSelect) countingModeSelect.disabled = true;
                    if (countingDisplay) countingDisplay.style.display = 'none';
                    return;
                }
                // Check if this zone already has settings — if so, use its saved mode
                const data = await fetchCountingState(currentVideo);
                const zoneSettings = data?.zone_settings || {};
                const existingMode = zoneSettings[zoneName]?.mode || (countingModeSelect ? countingModeSelect.value : 'simple');
                await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ zone_name: zoneName, mode: existingMode })
                });
                countingToggleBtn.disabled = false;
                await syncCountingUI();
            });
        }

        // Toggle button: enable/disable counting (dispatches to correct mode on server)
        if (countingToggleBtn) {
            countingToggleBtn.addEventListener('click', async () => {
                if (!currentVideo) return;
                await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/toggle`, { method: 'POST' });
                await syncCountingUI();
                await loadZones();
            });
        }

        // Flip button: rotates direction by 90°
        if (countingFlipBtn) {
            countingFlipBtn.addEventListener('click', async () => {
                if (!currentVideo) return;
                await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/flip`, { method: 'POST' });
                await syncCountingUI();
            });
        }

        // Reset button: reset counting (dispatches to correct mode on server)
        if (countingResetBtn) {
            countingResetBtn.addEventListener('click', async () => {
                if (!currentVideo) return;
                await fetch(`/api/counting/${encodeURIComponent(currentVideo)}/reset`, { method: 'POST' });
                await syncCountingUI();
            });
        }

        // Simple params sliders
        if (simpleThresholdSlider) {
            simpleThresholdSlider.addEventListener('input', () => {
                if (simpleThresholdValue) simpleThresholdValue.textContent = simpleThresholdSlider.value;
            });
            simpleThresholdSlider.addEventListener('change', async () => {
                await fetch('/api/counting/params', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ simple_gradient_threshold: parseInt(simpleThresholdSlider.value) })
                });
            });
        }
        if (simpleCooldownSlider) {
            simpleCooldownSlider.addEventListener('input', () => {
                if (simpleCooldownValue) simpleCooldownValue.textContent = simpleCooldownSlider.value;
            });
            simpleCooldownSlider.addEventListener('change', async () => {
                await fetch('/api/counting/params', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ simple_cooldown_frames: parseInt(simpleCooldownSlider.value) })
                });
            });
        }

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
                // If deleted zone was counting ROI, stop counting
                const cs = countingStateByVideo?.[currentVideo];
                if (cs?.zone_name === name && cs.enabled) {
                    try {
                        fetch(`/api/counting/${encodeURIComponent(currentVideo)}/toggle`, { method: 'POST' });
                    } catch {}
                }
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

        /* ====== ANALYTICS DASHBOARD ====== */
        let _analyticsInited = false;

        /* -- Heatmap fixed data (Arcy: presence par créneau) -- */
        const _hmData = {
            days: ['Lun','Mar','Mer','Jeu','Ven'],
            hours: ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00'],
            grid: [
                [10,15,20,25,30,22,18,12],
                [5,10,18,30,35,28,20,15],
                [8,12,22,28,32,25,19,10],
                [12,18,25,32,38,30,22,16],
                [15,20,28,35,40,33,25,18]
            ]
        };
        const _hmColors = ['#FFD440','#F8A340','#E84045','#BB015A','#8A0042'];

        const _hmMetrics = [
            { icon: 'diamond', label: 'Temps moyen de présence', value: '6h', dir: 'up' },
            { icon: 'circle',  label: 'Temps de réaction alerte', value: '4h', dir: 'up' },
            { icon: 'triangle',label: "Taux d'escalade zones", value: '10%', dir: 'down' },
        ];

        /* -- SVG icons -- */
        const _iDiamond = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M9.93 1.25c-.6.01-1.2.24-1.65.7L1.92 8.41a2.34 2.34 0 00.03 3.3l6.47 6.36a2.34 2.34 0 003.3-.03l6.36-6.47a2.34 2.34 0 00-.03-3.3L11.59 1.92a2.34 2.34 0 00-1.66-.67zM10 5.41a.63.63 0 01.63.63v5a.63.63 0 01-1.25 0v-5A.63.63 0 0110 5.41zm0 7.5a.83.83 0 110 1.67.83.83 0 010-1.67z" fill="#E84045"/></svg>`;
        const _iCircle  = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 1.67a8.33 8.33 0 100 16.67A8.33 8.33 0 0010 1.67zm0 1.25a7.08 7.08 0 110 14.17A7.08 7.08 0 0110 2.92zM10 5.83a.63.63 0 00-.63.63v4.17a.63.63 0 001.25 0V6.46A.63.63 0 0010 5.83zm0 6.67a.83.83 0 100 1.67.83.83 0 000-1.67z" fill="#E84045"/></svg>`;
        const _iTriangle= `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M10 2.11c-.65 0-1.3.32-1.65.95L1.9 14.71c-.68 1.23.24 2.79 1.65 2.79h12.9c1.4 0 2.33-1.56 1.65-2.79L11.65 3.05c-.35-.63-1-.95-1.65-.95zM10 6.66a.63.63 0 01.63.63v4.17a.63.63 0 01-1.25 0V7.29A.63.63 0 0110 6.66zm0 6.67a.83.83 0 110 1.67.83.83 0 010-1.67z" fill="#E84045"/></svg>`;
        const _tUp  = `<svg width="16" height="16" viewBox="0 0 20 21" fill="none"><path d="M5.5 9.1L10 4.67M10 4.67L14.5 9.1M10 4.67V16.33" stroke="#F08083" stroke-width="2" stroke-linecap="square"/></svg>`;
        const _tDown= `<svg width="16" height="16" viewBox="0 0 20 21" fill="none"><path d="M14.5 11.9L10 16.33M10 16.33L5.5 11.9M10 16.33V4.67" stroke="#40E5D1" stroke-width="2" stroke-linecap="square"/></svg>`;
        const _iconMap = { diamond: _iDiamond, circle: _iCircle, triangle: _iTriangle };

        /* -- Canvas helpers -- */
        function _hiDPI(canvas, w, h) {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = w * dpr;
            canvas.height = h * dpr;
            canvas.style.width = w + 'px';
            canvas.style.height = h + 'px';
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);
            return ctx;
        }

        /* -- Shared tooltip -- */
        let _tooltip = null;
        function _getTooltip() {
            if (!_tooltip) {
                _tooltip = document.createElement('div');
                _tooltip.className = 'aw-tooltip';
                document.body.appendChild(_tooltip);
            }
            return _tooltip;
        }
        function _showTooltip(e, html) {
            const t = _getTooltip();
            t.innerHTML = html;
            t.classList.add('is-visible');
            const pad = 12;
            const rect = t.getBoundingClientRect();
            let x = e.clientX + pad;
            let y = e.clientY - rect.height - pad;
            if (x + rect.width > window.innerWidth) x = e.clientX - rect.width - pad;
            if (y < 0) y = e.clientY + pad;
            t.style.left = x + 'px';
            t.style.top = y + 'px';
        }
        function _hideTooltip() {
            if (_tooltip) _tooltip.classList.remove('is-visible');
        }

        /* -- Canvas mouse coords helper (accounts for hi-DPI) -- */
        function _canvasCoords(canvas, e) {
            const r = canvas.getBoundingClientRect();
            return { x: e.clientX - r.left, y: e.clientY - r.top };
        }

        /* -- Attach hover to heatmap canvas -- */
        function _attachHeatmapHover(canvas) {
            const d = _hmData;
            canvas.style.cursor = 'crosshair';
            canvas.addEventListener('mousemove', e => {
                const W = 330, H = 220;
                const pad = { l: 40, t: 6, r: 6, b: 30 };
                const gw = W - pad.l - pad.r;
                const gh = H - pad.t - pad.b;
                const cw = gw / d.hours.length;
                const ch = gh / d.days.length;
                const { x, y } = _canvasCoords(canvas, e);
                const col = Math.floor((x - pad.l) / cw);
                const row = Math.floor((y - pad.t) / ch);
                if (col >= 0 && col < d.hours.length && row >= 0 && row < d.days.length) {
                    const val = d.grid[row][col];
                    const maxV = Math.max(...d.grid.flat());
                    const ratio = Math.min(val / maxV, 1);
                    const ci = Math.min(Math.floor(ratio * _hmColors.length), _hmColors.length - 1);
                    const color = _hmColors[ci];
                    _showTooltip(e, `
                        <div class="aw-tooltip__label">${d.days[row]} - ${d.hours[col]}</div>
                        <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${color}"></span>${val} présences</div>
                    `);
                } else {
                    _hideTooltip();
                }
            });
            canvas.addEventListener('mouseleave', _hideTooltip);
        }

        /* -- Attach hover to diverging bar canvas -- */
        function _attachDivergingHover(canvas) {
            const d = _dvData;
            canvas.style.cursor = 'crosshair';
            canvas.addEventListener('mousemove', e => {
                const W = 330, H = 200;
                const pad = { l: 40, t: 10, r: 16, b: 26 };
                const gw = W - pad.l - pad.r;
                const gh = H - pad.t - pad.b;
                const maxPos = Math.max(...d.resolved);
                const maxNeg = Math.max(...d.outstanding);
                const maxVal = maxPos + maxNeg;
                const zeroY = pad.t + (maxPos / maxVal) * gh;
                const barW = Math.min(32, (gw / d.months.length) * 0.55);
                const groupW = gw / d.months.length;
                const { x, y } = _canvasCoords(canvas, e);
                let found = false;
                for (let i = 0; i < d.months.length; i++) {
                    const cx = pad.l + i * groupW + groupW / 2;
                    const bx = cx - barW / 2;
                    const posH = (d.resolved[i] / maxVal) * gh;
                    const negH = (d.outstanding[i] / maxVal) * gh;
                    // Check positive bar
                    if (x >= bx && x <= bx + barW && y >= zeroY - posH && y <= zeroY) {
                        _showTooltip(e, `
                            <div class="aw-tooltip__label">${d.months[i]} - Traitées</div>
                            <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${_dvColors[0]}"></span>${d.resolved[i]}</div>
                        `);
                        found = true; break;
                    }
                    // Check negative bar
                    if (x >= bx && x <= bx + barW && y >= zeroY && y <= zeroY + negH) {
                        _showTooltip(e, `
                            <div class="aw-tooltip__label">${d.months[i]} - En cours</div>
                            <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${_dvColors[1]}"></span>${d.outstanding[i]}</div>
                        `);
                        found = true; break;
                    }
                }
                if (!found) _hideTooltip();
            });
            canvas.addEventListener('mouseleave', _hideTooltip);
        }

        /* -- Attach hover to horizontal bar canvas -- */
        function _attachHorizontalBarHover(canvas) {
            const d = _hbData;
            canvas.style.cursor = 'crosshair';
            canvas.addEventListener('mousemove', e => {
                const parentW = canvas.parentElement?.clientWidth || 740;
                const W = Math.max(340, parentW - 24), H = 200;
                const pad = { l: 110, t: 6, r: 16, b: 6 };
                const gw = W - pad.l - pad.r;
                const rowH = (H - pad.t - pad.b) / d.cats.length;
                const barH = Math.min(16, rowH * 0.6);
                const maxVal = Math.max(...d.values);
                const { x, y } = _canvasCoords(canvas, e);
                let found = false;
                for (let i = 0; i < d.cats.length; i++) {
                    const by = pad.t + i * rowH + (rowH - barH) / 2;
                    const bw = (d.values[i] / maxVal) * gw;
                    if (x >= pad.l && x <= pad.l + bw && y >= by && y <= by + barH) {
                        const color = _hbColors[i % _hbColors.length];
                        const pct = Math.round((d.values[i] / d.values.reduce((a,b) => a + b, 0)) * 100);
                        _showTooltip(e, `
                            <div class="aw-tooltip__label">${d.cats[i]}</div>
                            <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${color}"></span>${d.values[i]} détections (${pct}%)</div>
                        `);
                        found = true; break;
                    }
                }
                if (!found) _hideTooltip();
            });
            canvas.addEventListener('mouseleave', _hideTooltip);
        }

        /* -- Attach hover to stats bar mini chart -- */
        function _attachStatsBarHover(card) {
            card.querySelectorAll('.aw-stats-bar-col').forEach((col, i) => {
                const b = _sbData.bars[i];
                if (!b) return;
                col.style.cursor = 'pointer';
                col.addEventListener('mouseenter', e => {
                    const isLast = i === _sbData.bars.length - 1;
                    const color = isLast ? _sbData.highlightColor : 'rgba(29,91,255,0.7)';
                    _showTooltip(e, `
                        <div class="aw-tooltip__label">${b.name}</div>
                        <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${color}"></span>${b.value}%</div>
                    `);
                });
                col.addEventListener('mousemove', e => {
                    const isLast = i === _sbData.bars.length - 1;
                    const color = isLast ? _sbData.highlightColor : 'rgba(29,91,255,0.7)';
                    _showTooltip(e, `
                        <div class="aw-tooltip__label">${b.name}</div>
                        <div class="aw-tooltip__value"><span class="aw-tooltip__color" style="background:${color}"></span>${b.value}%</div>
                    `);
                });
                col.addEventListener('mouseleave', _hideTooltip);
            });
        }

        function _drawHeatmap(canvas) {
            const d = _hmData;
            const W = 330, H = 220;
            const ctx = _hiDPI(canvas, W, H);
            const pad = { l: 40, t: 6, r: 6, b: 30 };
            const gw = W - pad.l - pad.r;
            const gh = H - pad.t - pad.b;
            const cw = gw / d.hours.length;
            const ch = gh / d.days.length;
            const maxV = Math.max(...d.grid.flat());

            ctx.clearRect(0, 0, W, H);

            // Y-axis labels
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            d.days.forEach((day, i) => {
                ctx.fillStyle = '#9A9AAF';
                ctx.fillText(day, pad.l - 8, pad.t + i * ch + ch / 2);
            });

            // X-axis labels
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            d.hours.forEach((hr, j) => {
                ctx.save();
                ctx.translate(pad.l + j * cw + cw / 2, H - pad.b + 6);
                ctx.rotate(-55 * Math.PI / 180);
                ctx.fillStyle = '#9A9AAF';
                ctx.textAlign = 'right';
                ctx.fillText(hr, 0, 0);
                ctx.restore();
            });

            // Cells
            const gap = 2;
            d.grid.forEach((row, i) => {
                row.forEach((val, j) => {
                    const ratio = Math.min(val / maxV, 1);
                    const ci = Math.min(Math.floor(ratio * _hmColors.length), _hmColors.length - 1);
                    ctx.fillStyle = _hmColors[ci];
                    // Subtle glow on hottest cells only
                    if (ci >= 3) {
                        ctx.shadowColor = _hmColors[ci];
                        ctx.shadowBlur = 2;
                    }
                    ctx.beginPath();
                    ctx.roundRect(
                        pad.l + j * cw + gap,
                        pad.t + i * ch + gap,
                        cw - gap * 2,
                        ch - gap * 2,
                        3
                    );
                    ctx.fill();
                    ctx.shadowBlur = 0;
                });
            });
        }

        /* -- Card "..." menu helper -- */
        function _addCardMenu(card) {
            card.style.position = 'relative';
            const wrap = document.createElement('div');
            wrap.className = 'aw-card__menu-wrap';
            wrap.innerHTML = `
                <button class="aw-card__menu-btn" type="button" title="Options">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="2.5" r="1.5"/><circle cx="8" cy="8" r="1.5"/><circle cx="8" cy="13.5" r="1.5"/></svg>
                </button>
                <div class="aw-card__menu-panel">
                    <button class="aw-card__menu-item" data-action="configure" type="button">
                        <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12.22 2h-.44a2 2 0 00-2 2v.18a2 2 0 01-1 1.73l-.43.25a2 2 0 01-2 0l-.15-.08a2 2 0 00-2.73.73l-.22.38a2 2 0 00.73 2.73l.15.1a2 2 0 011 1.72v.51a2 2 0 01-1 1.74l-.15.09a2 2 0 00-.73 2.73l.22.38a2 2 0 002.73.73l.15-.08a2 2 0 012 0l.43.25a2 2 0 011 1.73V20a2 2 0 002 2h.44a2 2 0 002-2v-.18a2 2 0 011-1.73l.43-.25a2 2 0 012 0l.15.08a2 2 0 002.73-.73l.22-.39a2 2 0 00-.73-2.73l-.15-.08a2 2 0 01-1-1.74v-.5a2 2 0 011-1.74l.15-.09a2 2 0 00.73-2.73l-.22-.38a2 2 0 00-2.73-.73l-.15.08a2 2 0 01-2 0l-.43-.25a2 2 0 01-1-1.73V4a2 2 0 00-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                        Configurer
                    </button>
                    <button class="aw-card__menu-item aw-card__menu-item--danger" data-action="delete" type="button">
                        <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
                        Supprimer
                    </button>
                </div>`;
            card.appendChild(wrap);
        }

        /* -- Build widget card HTML -- */
        function _buildHeatmapCard() {
            const card = document.createElement('div');
            card.className = 'aw-card';
            _addCardMenu(card);

            // Title
            const title = document.createElement('div');
            title.className = 'aw-card__title';
            title.textContent = 'Rapport de présence';
            card.appendChild(title);

            // Chart area (canvas + legend bar)
            const chartWrap = document.createElement('div');
            chartWrap.className = 'aw-card__chart';

            const canvas = document.createElement('canvas');
            canvas.className = 'aw-card__canvas';
            chartWrap.appendChild(canvas);

            // Gradient legend bar
            const legendCol = document.createElement('div');
            legendCol.className = 'aw-card__legend';
            const gradBar = document.createElement('div');
            gradBar.className = 'aw-card__legend-bar';
            gradBar.style.background = `linear-gradient(to bottom, ${_hmColors[_hmColors.length-1]}, ${_hmColors[2]}, ${_hmColors[0]})`;
            gradBar.style.height = '140px';
            legendCol.appendChild(gradBar);
            const labelsCol = document.createElement('div');
            labelsCol.className = 'aw-card__legend-labels';
            labelsCol.innerHTML = '<span>40</span><span>20</span><span>0</span>';
            legendCol.appendChild(labelsCol);
            chartWrap.appendChild(legendCol);

            card.appendChild(chartWrap);

            // Metrics
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'aw-card__metrics';
            _hmMetrics.forEach(m => {
                const icon = _iconMap[m.icon] || _iCircle;
                const trendSVG = m.dir === 'up' ? _tUp : _tDown;
                const trendClass = m.dir === 'up' ? 'aw-metric__trend--up' : 'aw-metric__trend--down';
                const row = document.createElement('div');
                row.className = 'aw-metric';
                row.innerHTML = `
                    <div class="aw-metric__left">${icon}<span>${m.label}</span></div>
                    <div class="aw-metric__right">
                        <span class="aw-metric__value">${m.value}</span>
                        <span class="aw-metric__trend ${trendClass}">${trendSVG}</span>
                    </div>`;
                metricsDiv.appendChild(row);
            });
            card.appendChild(metricsDiv);

            return { card, canvas };
        }

        /* -- Diverging bar chart data (Arcy: alertes traitées vs en cours) -- */
        const _dvData = {
            months: ['Jan','Fév','Mar','Avr','Mai'],
            resolved: [50, 75, 60, 80, 40],
            outstanding: [20, 30, 15, 25, 10],
        };
        const _dvColors = ['#F7BFC1', '#E84045'];
        const _dvMetrics = [
            { icon: 'diamond', label: 'Délai moyen de traitement', value: '6h', dir: 'up' },
            { icon: 'circle',  label: 'Temps de réponse alerte', value: '4h', dir: 'up' },
            { icon: 'triangle',label: 'Taux de résolution', value: '10%', dir: 'down' },
        ];

        function _drawDivergingBar(canvas) {
            const d = _dvData;
            const W = 330, H = 200;
            const ctx = _hiDPI(canvas, W, H);
            const pad = { l: 40, t: 10, r: 16, b: 26 };
            const gw = W - pad.l - pad.r;
            const gh = H - pad.t - pad.b;
            const maxPos = Math.max(...d.resolved);
            const maxNeg = Math.max(...d.outstanding);
            const maxVal = maxPos + maxNeg;
            const zeroY = pad.t + (maxPos / maxVal) * gh;
            const barW = Math.min(32, (gw / d.months.length) * 0.55);
            const groupW = gw / d.months.length;

            ctx.clearRect(0, 0, W, H);

            // Gridlines
            for (let i = 0; i <= 4; i++) {
                const y = pad.t + (gh * i / 4);
                ctx.strokeStyle = 'rgba(126,126,143,0.25)';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(pad.l, y);
                ctx.lineTo(W - pad.r, y);
                ctx.stroke();
            }

            // Zero line
            ctx.strokeStyle = 'rgba(126,126,143,0.45)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pad.l, zeroY);
            ctx.lineTo(W - pad.r, zeroY);
            ctx.stroke();

            // Y-axis labels
            ctx.font = '10px sans-serif';
            ctx.fillStyle = '#9A9AAF';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(maxPos), pad.l - 8, pad.t);
            ctx.fillText('0', pad.l - 8, zeroY);
            ctx.fillText('-' + maxNeg, pad.l - 8, pad.t + gh);

            // Bars
            d.months.forEach((month, i) => {
                const cx = pad.l + i * groupW + groupW / 2;
                const bx = cx - barW / 2;

                // Positive bar (resolved)
                const posH = (d.resolved[i] / maxVal) * gh;
                ctx.fillStyle = _dvColors[0];
                ctx.beginPath();
                ctx.roundRect(bx, zeroY - posH, barW, posH, [4, 4, 0, 0]);
                ctx.fill();

                // Negative bar (outstanding)
                const negH = (d.outstanding[i] / maxVal) * gh;
                ctx.fillStyle = _dvColors[1];
                ctx.beginPath();
                ctx.roundRect(bx, zeroY, barW, negH, [0, 0, 4, 4]);
                ctx.fill();

                // X-axis label
                ctx.fillStyle = '#9A9AAF';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.shadowBlur = 0;
                ctx.fillText(month, cx, H - pad.b + 6);
            });
        }

        function _buildDivergingCard() {
            const card = document.createElement('div');
            card.className = 'aw-card';
            _addCardMenu(card);

            const title = document.createElement('div');
            title.className = 'aw-card__title';
            title.textContent = 'Suivi des alertes';
            card.appendChild(title);

            // Legend
            const legendWrap = document.createElement('div');
            legendWrap.style.cssText = 'display:flex;gap:16px;padding:0 28px 12px;';
            legendWrap.innerHTML = `
                <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#9A9AAF;">
                    <span style="width:12px;height:12px;border-radius:2px;background:${_dvColors[0]};"></span>Traitées
                </span>
                <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#9A9AAF;">
                    <span style="width:12px;height:12px;border-radius:2px;background:${_dvColors[1]};"></span>En cours
                </span>`;
            card.appendChild(legendWrap);

            const chartWrap = document.createElement('div');
            chartWrap.className = 'aw-card__chart';
            chartWrap.style.padding = '0 12px';
            const canvas = document.createElement('canvas');
            canvas.className = 'aw-card__canvas';
            chartWrap.appendChild(canvas);
            card.appendChild(chartWrap);

            // Metrics
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'aw-card__metrics';
            _dvMetrics.forEach(m => {
                const icon = _iconMap[m.icon] || _iCircle;
                const trendSVG = m.dir === 'up' ? _tUp : _tDown;
                const trendClass = m.dir === 'up' ? 'aw-metric__trend--up' : 'aw-metric__trend--down';
                const row = document.createElement('div');
                row.className = 'aw-metric';
                row.innerHTML = `
                    <div class="aw-metric__left">${icon}<span>${m.label}</span></div>
                    <div class="aw-metric__right">
                        <span class="aw-metric__value">${m.value}</span>
                        <span class="aw-metric__trend ${trendClass}">${trendSVG}</span>
                    </div>`;
                metricsDiv.appendChild(row);
            });
            card.appendChild(metricsDiv);

            return { card, canvas };
        }

        /* -- Drag & drop removed -- */

        /* -- Horizontal bar chart data (Arcy: répartition détections par type) -- */
        const _hbData = {
            cats: ['Présence humaine','Intrusion','Zone vide','Mouvement suspect','Comptage','Accès non autorisé'],
            values: [120, 90, 80, 70, 100, 50],
        };
        const _hbColors = ['#9152EE','#40D3F4','#40E5D1','#4C86FF','#DAC5F9','#F08083'];
        const _hbStats = [
            { label: 'Détections critiques', value: 321, pct: 12, dir: 'up', compare: 'vs 293 semaine dernière' },
            { label: 'Total détections', value: 1120, pct: 4, dir: 'down', compare: 'vs 1 060 semaine dernière' },
        ];
        const _hbMetrics = [
            { icon: 'diamond', label: 'Délai moyen de détection', value: '6h', dir: 'up' },
            { icon: 'circle',  label: 'Temps de réponse incident', value: '4h', dir: 'up' },
            { icon: 'triangle',label: 'Taux d\'escalade détections', value: '10%', dir: 'down' },
        ];

        function _drawHorizontalBar(canvas) {
            const d = _hbData;
            const parentW = canvas.parentElement?.clientWidth || 740;
            const W = Math.max(340, parentW - 24), H = 200;
            const ctx = _hiDPI(canvas, W, H);
            const pad = { l: 110, t: 6, r: 16, b: 6 };
            const gw = W - pad.l - pad.r;
            const rowH = (H - pad.t - pad.b) / d.cats.length;
            const barH = Math.min(16, rowH * 0.6);
            const maxVal = Math.max(...d.values);

            ctx.clearRect(0, 0, W, H);

            // Gridlines
            for (let i = 0; i <= 4; i++) {
                const x = pad.l + (gw * i / 4);
                ctx.strokeStyle = 'rgba(126,126,143,0.18)';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(x, pad.t);
                ctx.lineTo(x, H - pad.b);
                ctx.stroke();
            }

            d.cats.forEach((cat, i) => {
                const y = pad.t + i * rowH + (rowH - barH) / 2;
                const bw = (d.values[i] / maxVal) * gw;
                const color = _hbColors[i % _hbColors.length];

                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.roundRect(pad.l, y, bw, barH, [0, 4, 4, 0]);
                ctx.fill();

                // Label
                ctx.fillStyle = '#9A9AAF';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                const label = cat.length > 18 ? cat.slice(0, 17) + '…' : cat;
                ctx.fillText(label, pad.l - 8, y + barH / 2);
            });
        }

        function _buildHorizontalBarCard() {
            const card = document.createElement('div');
            card.className = 'aw-card';
            card.setAttribute('data-aw-type', 'hbar');
            card.style.width = '780px';
            _addCardMenu(card);

            // Header with title + period label
            const header = document.createElement('div');
            header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:28px 52px 20px 28px;';
            const title = document.createElement('div');
            title.className = 'aw-card__title';
            title.textContent = 'Rapport de détections';
            title.style.padding = '0';
            header.appendChild(title);
            const periodLabel = document.createElement('span');
            periodLabel.style.cssText = 'font-size:12px;color:#9A9AAF;padding:5px 12px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:6px;';
            periodLabel.textContent = '7 derniers jours';
            header.appendChild(periodLabel);
            card.appendChild(header);

            // Chart
            const chartWrap = document.createElement('div');
            chartWrap.className = 'aw-card__chart';
            chartWrap.style.padding = '0 12px';
            const canvas = document.createElement('canvas');
            canvas.className = 'aw-card__canvas';
            chartWrap.appendChild(canvas);
            card.appendChild(chartWrap);

            // Bottom row: stats left + metrics right
            const bottomRow = document.createElement('div');
            bottomRow.style.cssText = 'display:flex;gap:24px;padding:16px 28px 24px;';

            // Stats counters (left side)
            const statsWrap = document.createElement('div');
            statsWrap.style.cssText = 'display:flex;gap:20px;flex:1;';
            _hbStats.forEach(s => {
                const badgeColor = s.dir === 'up' ? 'rgba(232,64,69,0.40)' : 'rgba(64,229,209,0.40)';
                const textColor = s.dir === 'up' ? '#F08083' : '#40E5D1';
                const arrow = s.dir === 'up' ? _tUp : _tDown;
                const col = document.createElement('div');
                col.style.cssText = 'flex:1;';
                col.innerHTML = `
                    <div style="font-size:13px;color:rgba(255,255,255,0.50);margin-bottom:6px;">${s.label}</div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-family:'Courier New',monospace;font-size:30px;font-weight:700;color:#fff;">${s.value.toLocaleString('fr-FR')}</span>
                        <span style="display:inline-flex;align-items:center;gap:2px;padding:2px 8px;border-radius:20px;background:${badgeColor};color:${textColor};font-size:11px;font-weight:600;">${arrow} ${s.pct}%</span>
                    </div>
                    <div style="font-size:11px;color:rgba(255,255,255,0.25);margin-top:3px;">${s.compare}</div>`;
                statsWrap.appendChild(col);
            });
            bottomRow.appendChild(statsWrap);

            // Metrics (right side)
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'aw-card__metrics';
            metricsDiv.style.cssText = 'flex:1;padding:0;';
            _hbMetrics.forEach(m => {
                const icon = _iconMap[m.icon] || _iCircle;
                const trendSVG = m.dir === 'up' ? _tUp : _tDown;
                const trendClass = m.dir === 'up' ? 'aw-metric__trend--up' : 'aw-metric__trend--down';
                const row = document.createElement('div');
                row.className = 'aw-metric';
                row.innerHTML = `
                    <div class="aw-metric__left">${icon}<span>${m.label}</span></div>
                    <div class="aw-metric__right">
                        <span class="aw-metric__value">${m.value}</span>
                        <span class="aw-metric__trend ${trendClass}">${trendSVG}</span>
                    </div>`;
                metricsDiv.appendChild(row);
            });
            bottomRow.appendChild(metricsDiv);
            card.appendChild(bottomRow);

            return { card, canvas };
        }

        /* -- Widget 4: Stats mini-bar card (Arcy: détections par site) -- */
        const _sbData = {
            title: 'Détections mensuelles',
            value: 2847,
            description: '+12.5% vs mois dernier',
            bars: [
                { name: 'Jan', value: 40 },
                { name: 'Fév', value: 55 },
                { name: 'Mar', value: 45 },
                { name: 'Avr', value: 70 },
                { name: 'Mai', value: 60 },
                { name: 'Juin', value: 85 },
            ],
            defaultColor: 'rgba(29,91,255,0.25)',
            highlightColor: '#E84045',
        };

        function _buildStatsBarCard() {
            const card = document.createElement('div');
            card.className = 'aw-card';
            card.style.width = '320px';
            _addCardMenu(card);

            // Header
            const header = document.createElement('div');
            header.style.cssText = 'padding:28px 28px 8px;';
            header.innerHTML = `<div style="font-size:13px;color:#9A9AAF;margin-bottom:10px;">${_sbData.title}</div>
                <div style="font-size:32px;font-weight:800;color:#fff;font-family:'Courier New',monospace;">${_sbData.value.toLocaleString('fr-FR')}</div>
                <div style="font-size:12px;color:#9A9AAF;margin-top:4px;">${_sbData.description}</div>`;
            card.appendChild(header);

            // Mini bar chart
            const barsWrap = document.createElement('div');
            barsWrap.className = 'aw-stats-bars';
            barsWrap.style.marginTop = '16px';
            _sbData.bars.forEach((b, i) => {
                const col = document.createElement('div');
                col.className = 'aw-stats-bar-col';
                const bar = document.createElement('div');
                bar.className = 'aw-stats-bar';
                const isLast = i === _sbData.bars.length - 1;
                bar.style.background = isLast ? _sbData.highlightColor : _sbData.defaultColor;
                bar.style.height = '0%';
                bar.setAttribute('data-target-h', b.value + '%');
                col.appendChild(bar);
                const lbl = document.createElement('div');
                lbl.className = 'aw-stats-bar-label';
                lbl.textContent = b.name;
                col.appendChild(lbl);
                barsWrap.appendChild(col);
            });
            card.appendChild(barsWrap);

            // Bottom padding
            const spacer = document.createElement('div');
            spacer.style.height = '24px';
            card.appendChild(spacer);

            return { card };
        }

        function _animateStatsBars(card) {
            card.querySelectorAll('.aw-stats-bar').forEach(bar => {
                const h = bar.getAttribute('data-target-h');
                if (h) requestAnimationFrame(() => { bar.style.height = h; });
            });
        }

        /* -- Widget 5: Circular progress card (Arcy: objectif couverture) -- */
        const _cpData = {
            title: 'Couverture zones',
            description: 'Zones surveillées vs objectif',
            current: 18,
            goal: 24,
            color: '#1d5bff',
        };

        function _buildCircularProgressCard() {
            const card = document.createElement('div');
            card.className = 'aw-card';
            card.style.width = '320px';
            _addCardMenu(card);

            const pct = Math.round((_cpData.current / _cpData.goal) * 100);
            const r = 72, circ = 2 * Math.PI * r;
            const offset = circ * (1 - pct / 100);

            // Header
            const header = document.createElement('div');
            header.style.cssText = 'padding:28px 28px 0;text-align:center;';
            header.innerHTML = `<div style="font-size:18px;font-weight:700;color:#fff;">${_cpData.title}</div>
                <div style="font-size:12px;color:#9A9AAF;margin-top:4px;">${_cpData.description}</div>`;
            card.appendChild(header);

            // Circle
            const circleWrap = document.createElement('div');
            circleWrap.className = 'aw-circle-wrap';
            circleWrap.style.margin = '20px auto 24px';
            circleWrap.innerHTML = `
                <svg viewBox="0 0 200 200">
                    <g transform="rotate(-90,100,100)">
                        <circle cx="100" cy="100" r="${r}" fill="transparent" stroke="rgba(255,255,255,0.06)" stroke-width="14"/>
                        <circle class="aw-progress-arc" cx="100" cy="100" r="${r}" fill="transparent"
                            stroke="${_cpData.color}" stroke-width="14" stroke-linecap="round"
                            stroke-dasharray="${circ}" stroke-dashoffset="${circ}"
                            data-target-offset="${offset}"
                            style="transition: stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);"/>
                    </g>
                </svg>
                <div class="aw-circle-center">
                    <span class="aw-circle-pct">${pct}%</span>
                    <span class="aw-circle-sub">${_cpData.current} / ${_cpData.goal} zones</span>
                </div>`;
            card.appendChild(circleWrap);

            return { card };
        }

        function _animateCircle(card) {
            const arc = card.querySelector('.aw-progress-arc');
            if (arc) {
                const target = arc.getAttribute('data-target-offset');
                requestAnimationFrame(() => { arc.style.strokeDashoffset = target; });
            }
        }

        /* -- Card menu interactions -- */
        function _initCardMenus(grid) {
            // Toggle "..." menu
            grid.addEventListener('click', (e) => {
                const btn = e.target.closest('.aw-card__menu-btn');
                if (btn) {
                    e.stopPropagation();
                    const wrap = btn.closest('.aw-card__menu-wrap');
                    const wasOpen = wrap.classList.contains('is-open');
                    grid.querySelectorAll('.aw-card__menu-wrap.is-open').forEach(w => w.classList.remove('is-open'));
                    if (!wasOpen) wrap.classList.add('is-open');
                    return;
                }
                const item = e.target.closest('.aw-card__menu-item');
                if (item) {
                    e.stopPropagation();
                    const action = item.getAttribute('data-action');
                    const card = item.closest('.aw-card');
                    grid.querySelectorAll('.aw-card__menu-wrap.is-open').forEach(w => w.classList.remove('is-open'));
                    if (action === 'delete' && card) {
                        card.style.transition = 'opacity 0.25s, transform 0.25s';
                        card.style.opacity = '0';
                        card.style.transform = 'scale(0.95)';
                        setTimeout(() => card.remove(), 260);
                    }
                    // 'configure' → nothing for now
                    return;
                }
                // Click outside menus → close
                grid.querySelectorAll('.aw-card__menu-wrap.is-open').forEach(w => w.classList.remove('is-open'));
            });
            document.addEventListener('click', () => {
                grid.querySelectorAll('.aw-card__menu-wrap.is-open').forEach(w => w.classList.remove('is-open'));
            });
        }

        function _redrawAllAwCards(grid) {
            grid.querySelectorAll('.aw-card').forEach(card => {
                const canvas = card.querySelector('.aw-card__canvas');
                if (!canvas) return;
                const title = card.querySelector('.aw-card__title')?.textContent || '';
                if (title.includes('présence')) _drawHeatmap(canvas);
                else if (title.includes('alertes')) _drawDivergingBar(canvas);
                else if (title.includes('détections')) _drawHorizontalBar(canvas);
            });
        }

        /* -- Init -- */
        function initAnalyticsDashboard() {
            const grid = document.getElementById('analyticsGrid');
            if (!grid) return;

            if (_analyticsInited) {
                // Redraw canvases on revisit
                requestAnimationFrame(() => _redrawAllAwCards(grid));
                return;
            }
            _analyticsInited = true;

            // Build cards
            grid.innerHTML = '';
            const hm = _buildHeatmapCard();
            grid.appendChild(hm.card);
            const dv = _buildDivergingCard();
            grid.appendChild(dv.card);
            const hb = _buildHorizontalBarCard();
            grid.appendChild(hb.card);
            const sb = _buildStatsBarCard();
            grid.appendChild(sb.card);
            const cp = _buildCircularProgressCard();
            grid.appendChild(cp.card);

            // drag-and-drop removed
            _initCardMenus(grid);
            _initFab(grid);

            // Draw after layout + attach hover tooltips
            requestAnimationFrame(() => {
                _redrawAllAwCards(grid);
                _animateStatsBars(sb.card);
                _animateCircle(cp.card);
                // Hover tooltips
                _attachHeatmapHover(hm.canvas);
                _attachDivergingHover(dv.canvas);
                _attachHorizontalBarHover(hb.canvas);
                _attachStatsBarHover(sb.card);
            });
        }

        /* -- Floating Action Button "+" -- */
        const _fabWidgets = [
            { label: 'Heatmap présence', build: () => _buildHeatmapCard() },
            { label: 'Suivi alertes', build: () => _buildDivergingCard() },
            { label: 'Rapport détections', build: () => _buildHorizontalBarCard() },
            { label: 'Stats mensuelles', build: () => _buildStatsBarCard() },
            { label: 'Couverture zones', build: () => _buildCircularProgressCard() },
        ];

        function _initFab(grid) {
            // Create FAB container inside analyticsView
            const view = document.getElementById('analyticsView');
            if (!view || view.querySelector('.aw-fab')) return;

            const fab = document.createElement('div');
            fab.className = 'aw-fab';
            fab.innerHTML = `
                <button class="aw-fab__btn" type="button" title="Ajouter un widget">
                    <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                </button>
                <div class="aw-fab__menu">
                    ${_fabWidgets.map((w, i) => `
                        <button class="aw-fab__option" data-fab-idx="${i}" type="button">
                            ${w.label}
                        </button>`).join('')}
                </div>`;
            view.appendChild(fab);

            // Toggle
            fab.querySelector('.aw-fab__btn').addEventListener('click', (e) => {
                e.stopPropagation();
                fab.classList.toggle('is-open');
            });

            // Option click → add widget
            fab.querySelectorAll('.aw-fab__option').forEach(opt => {
                opt.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const idx = parseInt(opt.getAttribute('data-fab-idx'), 10);
                    const wDef = _fabWidgets[idx];
                    if (!wDef) return;
                    const result = wDef.build();
                    grid.appendChild(result.card);
                    // Draw canvas / animate / attach hover
                    requestAnimationFrame(() => {
                        if (result.canvas) {
                            const title = result.card.querySelector('.aw-card__title')?.textContent || '';
                            if (title.includes('présence')) { _drawHeatmap(result.canvas); _attachHeatmapHover(result.canvas); }
                            else if (title.includes('alertes')) { _drawDivergingBar(result.canvas); _attachDivergingHover(result.canvas); }
                            else if (title.includes('détections')) { _drawHorizontalBar(result.canvas); _attachHorizontalBarHover(result.canvas); }
                        }
                        _animateStatsBars(result.card);
                        _attachStatsBarHover(result.card);
                        _animateCircle(result.card);
                    });
                    fab.classList.remove('is-open');
                    // Scroll to new card
                    result.card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                });
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (!fab.contains(e.target)) fab.classList.remove('is-open');
            });
        }

        window.addEventListener('resize', () => {
            syncCanvasSize();
            if (_analyticsInited && !document.getElementById('analyticsView')?.classList.contains('hidden')) {
                const g = document.getElementById('analyticsGrid');
                if (g) _redrawAllAwCards(g);
            }
        });