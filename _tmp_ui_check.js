
        // State
        let currentVideo = null;
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
        const drawPanel = document.getElementById('drawPanel');
        const drawZoneSelect = document.getElementById('drawZoneSelect');
        const drawZoneNameGroup = document.getElementById('drawZoneNameGroup');
        const drawZoneName = document.getElementById('drawZoneName');
        const toolPolyBtn = document.getElementById('toolPolyBtn');
        const toolLineBtn = document.getElementById('toolLineBtn');
        const startDrawBtn = document.getElementById('startDrawBtn');
        const stopDrawBtn = document.getElementById('stopDrawBtn');
        const drawHud = document.getElementById('drawHud');
        const drawHudTitle = document.getElementById('drawHudTitle');
        const undoBtn = document.getElementById('undoBtn');
        const finishBtn = document.getElementById('finishBtn');
        const cancelBtn = document.getElementById('cancelBtn');
        const drawInstructions = document.getElementById('drawInstructions');
        const startDetectionBtn = document.getElementById('startDetectionBtn');
        const stopDetectionBtn = document.getElementById('stopDetectionBtn');
        const stopAllBtn = document.getElementById('stopAllBtn');
        const resetTimersBtn = document.getElementById('resetTimersBtn');
        const zonesGrid = document.getElementById('zonesGrid');
        const statusBadge = document.getElementById('statusBadge');
        const statusText = document.getElementById('statusText');
        const activeStreamsDiv = document.getElementById('activeStreams');
        const cameraGrid = document.getElementById('cameraGrid');
        const currentVideoTitle = document.getElementById('currentVideoTitle');
        const zoneListSidebar = document.getElementById('zoneListSidebar');

        function setTool(tool) {
            drawMode = tool;
            toolPolyBtn.classList.toggle('active', tool === 'poly');
            toolLineBtn.classList.toggle('active', tool === 'line');
            updateFinishButtonState();
        }
        const steps = {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3')
        };

        // Initialize
        init();

        async function init() {
            // Important: rÃ©cupÃ©rer d'abord les streams, puis construire l'UI (Ã©vite les â€œresetsâ€)
            await updateActiveStreams();
            await loadVideos();
            await loadZones();
            updateSteps();

            setInterval(loadZones, 1000);
            setInterval(updateActiveStreams, 2000);
        }

        async function loadVideos() {
            const res = await fetch('/api/videos');
            const data = await res.json();

            const selected = videoSelect.value || currentVideo || '';

            videoSelect.innerHTML = '<option value="">-- Choisir une vidÃ©o --</option>';
            cameraGrid.innerHTML = '';

            data.videos.forEach((v, i) => {
                videoSelect.innerHTML += `<option value="${v}">${v}</option>`;

                const isActive = activeVideoStreams.has(v);
                const isCurrent = v === currentVideo;

                cameraGrid.innerHTML += `
                    <div class="camera-item ${isCurrent ? 'active' : ''}" data-video="${v}" onclick="selectVideo('${v}')">
                        <div class="camera-item-name">${v.split('.')[0]}</div>
                        <div class="camera-item-status ${isActive ? 'online' : 'offline'}">
                            <span class="camera-status-dot" style="width:6px;height:6px;border-radius:50%;background:currentColor;"></span>
                            <span class="camera-status-text">${isActive ? 'En ligne' : 'Hors ligne'}</span>
                        </div>
                    </div>
                `;
            });

            // Restore selection if possible (stabilise la dropdown)
            if (selected && data.videos.includes(selected)) {
                videoSelect.value = selected;
            }
        }

        function selectVideo(videoName) {
            videoSelect.value = videoName;
            videoSelect.dispatchEvent(new Event('change'));
        }

        async function updateActiveStreams() {
            try {
                const res = await fetch('/api/streams');
                const data = await res.json();

                activeVideoStreams = new Set(data.streams.map(s => s.video));

                if (data.streams.length === 0) {
                    activeStreamsDiv.innerHTML = '';
                    stopAllBtn.classList.add('hidden');
                } else {
                    activeStreamsDiv.innerHTML = data.streams.map(s =>
                        `<span class="stream-badge ${s.video === currentVideo ? 'current' : ''}">
                            <span class="dot"></span>
                            ${s.video}
                        </span>`
                    ).join('');
                    stopAllBtn.classList.remove('hidden');
                }

                if (currentVideo) {
                    isCurrentVideoStreaming = activeVideoStreams.has(currentVideo);
                    if (isCurrentVideoStreaming) {
                        startDetectionBtn.classList.add('hidden');
                        stopDetectionBtn.classList.remove('hidden');
                    } else {
                        startDetectionBtn.classList.remove('hidden');
                        stopDetectionBtn.classList.add('hidden');
                    }
                }

                // Met Ã  jour les cartes camÃ©ras sans reconstruire le DOM (Ã©vite le jitter)
                document.querySelectorAll('.camera-item[data-video]').forEach((el) => {
                    const v = el.getAttribute('data-video');
                    const active = activeVideoStreams.has(v);
                    const isCurrent = v === currentVideo;

                    el.classList.toggle('active', isCurrent);

                    const statusEl = el.querySelector('.camera-item-status');
                    if (!statusEl) return;

                    statusEl.classList.toggle('online', active);
                    statusEl.classList.toggle('offline', !active);
                    const textEl = statusEl.querySelector('.camera-status-text');
                    if (textEl) textEl.textContent = active ? 'En ligne' : 'Hors ligne';
                });
            } catch (e) {
                console.error('Error fetching streams:', e);
            }
        }

        async function loadZones() {
            if (!currentVideo) {
                zonesGrid.innerHTML = '<div class="no-zones">SÃ©lectionnez une vidÃ©o</div>';
                zoneListSidebar.innerHTML = '<div style="color: var(--sidebar-text-subtle); font-size: var(--text-sm);">SÃ©lectionnez une vidÃ©o</div>';
                return;
            }

            const [presenceRes, zonesRes] = await Promise.all([
                fetch(`/api/presence/${encodeURIComponent(currentVideo)}`),
                fetch(`/api/zones/${encodeURIComponent(currentVideo)}`)
            ]);
            const presenceData = await presenceRes.json();
            const zonesData = await zonesRes.json();

            const zones = presenceData.zones || {};
            const zonesWithPolygons = zonesData.zones || {};

            if (Object.keys(zones).length === 0) {
                zonesGrid.innerHTML = '<div class="no-zones">Aucune zone dÃ©finie pour cette vidÃ©o</div>';
                zoneListSidebar.innerHTML = '<div style="color: var(--sidebar-text-subtle); font-size: var(--text-sm);">Aucune zone</div>';
                updateDrawPanelZones(zonesWithPolygons);
                return;
            }

            zonesGrid.innerHTML = '';
            zoneListSidebar.innerHTML = '';
            updateDrawPanelZones(zonesWithPolygons);

            for (const [name, info] of Object.entries(zones)) {
                const statusClass = info.is_occupied ? 'occupied' : 'empty';
                const statusLabel = info.is_occupied ? 'OccupÃ©' : 'Vide';

                zonesGrid.innerHTML += `
                    <div class="zone-card">
                        <div class="zone-card-header">
                            <div class="zone-card-name">${name}</div>
                            <div class="zone-card-status ${statusClass}">${statusLabel}</div>
                        </div>
                        <div class="zone-card-time">${info.formatted_time}</div>
                        <div class="zone-card-actions">
                            <button class="btn btn-ghost btn-icon" onclick="resetZoneTimer('${name}')" title="Reset">
                                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                                </svg>
                            </button>
                            <button class="btn btn-ghost btn-icon" onclick="deleteZone('${name}')" title="Supprimer">
                                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                `;

                // Sidebar explorer: Source vidÃ©o > Zones > Dessins
                const polyCount = (zonesWithPolygons[name]?.polygons || []).length;
                zonePolygonCounts[name] = polyCount;
            }

            // Render explorer tree
            renderAssetTree(zones, zonesWithPolygons);
        }

        function updateDrawPanelZones(zonesWithPolygons) {
            const prev = drawZoneSelect.value;
            drawZoneSelect.innerHTML = '<option value="">â€” Choisir â€”</option><option value="__new__">+ Nouvelle zoneâ€¦</option>';
            Object.keys(zonesWithPolygons || {}).sort().forEach((z) => {
                drawZoneSelect.innerHTML += `<option value="${z}">${z}</option>`;
            });
            if (prev && [...drawZoneSelect.options].some(o => o.value === prev)) {
                drawZoneSelect.value = prev;
            }
            drawZoneNameGroup.classList.toggle('hidden', drawZoneSelect.value !== '__new__');
        }

        function iconForDrawing(video, zone, idx) {
            try {
                const t = localStorage.getItem(`drawmeta:${video}:${zone}:${idx}`);
                if (t === 'line') return 'ï¼'; // ligne
            } catch {}
            return 'â¬ '; // polygone
        }

        function renderAssetTree(presenceZones, zonesWithPolygons) {
            const root = `
                <div class="draw-tree">
                    <div class="tree-row" onclick="selectVideo('${currentVideo}')">
                        <div class="left">
                            <span class="label">ðŸ“¹ ${currentVideo}</span>
                        </div>
                        <span class="meta">Source</span>
                    </div>
                    <div class="tree-children">
                        ${Object.keys(zonesWithPolygons || {}).sort().map((zoneName) => {
                            const info = presenceZones[zoneName] || { formatted_time: '00:00:00', is_occupied: false };
                            const drawings = zonesWithPolygons[zoneName]?.polygons || [];
                            const dotClass = info.is_occupied ? 'occupied' : '';
                            return `
                                <div class="tree-row" onclick="selectZone('${zoneName}')">
                                    <div class="left">
                                        <span class="zone-dot ${dotClass}"></span>
                                        <span class="label">${zoneName}</span>
                                    </div>
                                    <span class="meta">${info.formatted_time}</span>
                                </div>
                                <div class="tree-children">
                                    ${drawings.map((_, idx) => `
                                        <div class="tree-row tree-leaf" onclick="selectDrawing('${zoneName}', ${idx})">
                                            <div class="left">
                                                <span class="label">${iconForDrawing(currentVideo, zoneName, idx)} Dessin ${idx + 1}</span>
                                            </div>
                                            <span class="meta">#${idx + 1}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
            zoneListSidebar.innerHTML = root;
        }

        // Globaux cliquables depuis le HTML (style explorateur)
        window.selectZone = (zoneName) => {
            drawZoneSelect.value = zoneName;
            drawZoneNameGroup.classList.add('hidden');
            selectedAsset = { zone: zoneName };
            drawExistingZones();
        };

        window.selectDrawing = (zoneName, idx) => {
            selectedAsset = { zone: zoneName, idx };
            drawExistingZones();
        };

        function updateSteps() {
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
                statusText.textContent = 'DÃ©tection en cours';
            } else if (status === 'drawing') {
                statusText.textContent = 'Mode dessin';
            } else {
                statusText.textContent = 'PrÃªt';
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

            currentVideo = videoSelect.value;
            currentVideoTitle.textContent = currentVideo;

            const infoRes = await fetch(`/api/videos/${encodeURIComponent(currentVideo)}/info`);
            const info = await infoRes.json();
            videoWidth = info.width;
            videoHeight = info.height;

            drawCanvas.width = videoWidth;
            drawCanvas.height = videoHeight;

            isCurrentVideoStreaming = activeVideoStreams.has(currentVideo);

            if (isCurrentVideoStreaming) {
                videoFrame.classList.add('hidden');
                videoStream.classList.remove('hidden');
                videoStream.src = `/api/stream/${encodeURIComponent(currentVideo)}`;
                drawCanvas.classList.add('hidden');
                updateStatus('streaming');
            } else {
                videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
                videoFrame.classList.remove('hidden');
                videoStream.classList.add('hidden');
                videoStream.src = '';
                drawCanvas.classList.remove('hidden');
                updateStatus('ready');
            }

            placeholder.classList.add('hidden');
            startDetectionBtn.disabled = false;

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
            drawHudTitle.textContent = `Mode dessin â€” ${name}`;
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

        // Panneau dessin (sans modal)
        toggleDrawPanelBtn.addEventListener('click', () => {
            drawPanel.classList.toggle('hidden');
        });

        drawZoneSelect.addEventListener('change', () => {
            const v = drawZoneSelect.value;
            drawZoneNameGroup.classList.toggle('hidden', v !== '__new__');
            if (v === '__new__') drawZoneName.focus();
        });

        toolPolyBtn.addEventListener('click', () => setTool('poly'));
        toolLineBtn.addEventListener('click', () => setTool('line'));

        startDrawBtn.addEventListener('click', () => {
            if (!currentVideo) {
                alert('SÃ©lectionnez d\'abord une vidÃ©o');
                return;
            }
            if (isCurrentVideoStreaming) {
                alert('ArrÃªtez la dÃ©tection avant de dessiner');
                return;
            }

            let zoneNameSelected = drawZoneSelect.value;
            if (!zoneNameSelected) {
                alert('Choisissez une zone (ou crÃ©ez-en une)');
                return;
            }
            if (zoneNameSelected === '__new__') {
                zoneNameSelected = drawZoneName.value.trim();
                if (!zoneNameSelected) {
                    alert('Entrez un nom de zone');
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

            // Meta: mÃ©morise le type du nouveau dessin cÃ´tÃ© navigateur (sans back)
            try {
                localStorage.setItem(`drawmeta:${currentVideo}:${zoneName}:${prevCount}`, drawMode);
            } catch {}

            // Recharge et prÃ©pare un autre dessin pour la mÃªme zone (append)
            await loadZones();
            drawExistingZones();
            drawPoints = [];
            redrawCanvas();
            updateFinishButtonState();
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

        async function drawExistingZones() {
            if (!currentVideo) return;

            const res = await fetch(`/api/zones/${encodeURIComponent(currentVideo)}`);
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

                    ctx.beginPath();
                    ctx.moveTo(polygon[0][0], polygon[0][1]);
                    for (let i = 1; i < polygon.length; i++) {
                        ctx.lineTo(polygon[i][0], polygon[i][1]);
                    }
                    ctx.closePath();

                    ctx.fillStyle = isSelectedDrawing || isSelectedZone
                        ? 'rgba(16, 176, 249, 0.18)'
                        : 'rgba(240, 131, 33, 0.16)';
                    ctx.fill();
                    ctx.strokeStyle = isSelectedDrawing || isSelectedZone ? '#10B0F9' : '#F08321';
                    ctx.lineWidth = isSelectedDrawing ? 4 : (isSelectedZone ? 3 : 2);
                    ctx.stroke();

                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 14px Manrope, system-ui';
                    const label = isSelectedDrawing ? `${name} â€¢ Dessin ${idx + 1}` : name;
                    ctx.fillText(label, polygon[0][0] + 5, polygon[0][1] - 8);
                }
            }
        }

        // Detection
        startDetectionBtn.addEventListener('click', async () => {
            if (!currentVideo) return;

            isCurrentVideoStreaming = true;
            activeVideoStreams.add(currentVideo);

            videoFrame.classList.add('hidden');
            videoStream.classList.remove('hidden');
            drawCanvas.classList.add('hidden');

            videoStream.src = `/api/stream/${encodeURIComponent(currentVideo)}`;

            startDetectionBtn.classList.add('hidden');
            stopDetectionBtn.classList.remove('hidden');

            updateSteps();
            updateStatus('streaming');
            await updateActiveStreams();
        });

        stopDetectionBtn.addEventListener('click', async () => {
            if (!currentVideo) return;

            await fetch(`/api/stream/${encodeURIComponent(currentVideo)}/stop`, { method: 'POST' });

            isCurrentVideoStreaming = false;
            activeVideoStreams.delete(currentVideo);

            videoStream.src = '';
            videoStream.classList.add('hidden');
            videoFrame.src = `/api/videos/${encodeURIComponent(currentVideo)}/frame?t=${Date.now()}`;
            videoFrame.classList.remove('hidden');
            drawCanvas.classList.remove('hidden');

            startDetectionBtn.classList.remove('hidden');
            stopDetectionBtn.classList.add('hidden');

            updateSteps();
            updateStatus('ready');
            await updateActiveStreams();

            videoFrame.onload = () => {
                syncCanvasSize();
                drawExistingZones();
            };
        });

        stopAllBtn.addEventListener('click', async () => {
            await fetch('/api/streams/stop', { method: 'POST' });

            isCurrentVideoStreaming = false;
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

            startDetectionBtn.classList.remove('hidden');
            stopDetectionBtn.classList.add('hidden');

            updateSteps();
            updateStatus('ready');
            await updateActiveStreams();
        });

        resetTimersBtn.addEventListener('click', async () => {
            if (!confirm('Remettre tous les timers Ã  zÃ©ro ?')) return;
            await fetch('/api/zones/reset', { method: 'POST' });
            await loadZones();
        });

        async function resetZoneTimer(name) {
            if (!confirm(`Remettre le timer de "${name}" Ã  zÃ©ro ?`)) return;
            await fetch(`/api/zones/reset/${encodeURIComponent(name)}`, { method: 'POST' });
            await loadZones();
        }

        // Blur toggle
        const toggleBlurBtn = document.getElementById('toggleBlurBtn');

        async function updateBlurButton() {
            const res = await fetch('/api/blur');
            const data = await res.json();
            if (data.enabled) {
                toggleBlurBtn.innerHTML = `
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                    </svg>
                    Floutage: ON
                `;
                toggleBlurBtn.classList.add('btn-blur-on');
            } else {
                toggleBlurBtn.innerHTML = `
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"/>
                    </svg>
                    Floutage: OFF
                `;
                toggleBlurBtn.classList.remove('btn-blur-on');
            }
        }

        toggleBlurBtn.addEventListener('click', async () => {
            await fetch('/api/blur/toggle', { method: 'POST' });
            await updateBlurButton();
        });

        updateBlurButton();

        async function deleteZone(name) {
            if (!confirm(`Supprimer la zone "${name}" de cette vidÃ©o ?`)) return;
            await fetch(`/api/zones/${encodeURIComponent(currentVideo)}/${encodeURIComponent(name)}`, { method: 'DELETE' });
            await loadZones();
            drawExistingZones();
        }

        document.getElementById('deleteAllZonesBtn').addEventListener('click', async () => {
            if (!currentVideo) return;
            if (!confirm('Supprimer TOUTES les zones de cette vidÃ©o ?')) return;
            await fetch(`/api/zones/${encodeURIComponent(currentVideo)}`, { method: 'DELETE' });
            await loadZones();
            drawExistingZones();
        });

        window.addEventListener('resize', syncCanvasSize);
    
