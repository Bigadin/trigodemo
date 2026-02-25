import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { useMemo, useState, useEffect, useRef } from 'react'
import { useSessionTimer } from '@/hooks/useSessionTimer'
import { useHierarchy } from '@/context/HierarchyContext'
import {
  fetchZones,
  fetchStreams,
  startStream,
  stopStream,
  stopAllStreams,
  fetchCounting,
  fetchVideoInfo,
  fetchDetections,
} from '@/api/tracker'
import { loadSkillsConfig } from '@/api/skills'
import type { ZonesResponse, StreamsResponse, CountingResponse, Detection } from '@/api/tracker'
import { findSiteById, getCameraVideoPath } from '@/utils/hierarchy'
import { icon } from '@/utils/theme'
import VideoPlayer from '@/components/tracker/VideoPlayer'
import CameraGrid, { type CameraWithStatus } from '@/components/tracker/CameraGrid'
import BenefitsOverview from '@/components/tracker/BenefitsOverview'
import BenefitConfigModal from '@/components/tracker/BenefitConfigModal'
import { toggleBenefit, deleteBenefit, syncBenefitZones } from '@/api/benefits'
import DataRoomCards from '@/components/tracker/DataRoomCards'
import LovDropdown from '@/components/ui/LovDropdown'
import styles from './TrackerView.module.css'

type TrackerTab = 'overview' | 'source' | 'settings'

function getSiteAndCam(
  hierarchy: Record<string, { sites?: Record<string, { name?: string; cameras?: Record<string, { name?: string }> }> }>,
  siteId: string | undefined,
  camId: string | null
): { siteName: string; camName: string } | null {
  if (!siteId) return null
  for (const lieu of Object.values(hierarchy)) {
    const site = lieu.sites?.[siteId]
    if (!site) continue
    if (!camId) return { siteName: site.name || siteId, camName: '' }
    const cam = site.cameras?.[camId]
    return { siteName: site.name || siteId, camName: cam?.name || camId }
  }
  return null
}

export default function TrackerView() {
  const { siteId } = useParams<{ siteId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const { hierarchy, loading: hierarchyLoading, refetch: refetchHierarchy } = useHierarchy()

  const camId = searchParams.get('cam')
  const site = useMemo(
    () => (siteId ? findSiteById(hierarchy, siteId) : null),
    [hierarchy, siteId]
  )

  const camerasWithStatus = useMemo((): CameraWithStatus[] => {
    if (!site?.cameras) return []
    const cams = Object.entries(site.cameras)
    return cams.map(([id, cam]) => ({
      ...cam,
      camera_id: id,
      videoPath: getCameraVideoPath(cam),
      isActive: false,
    }))
  }, [site])

  const selectedCam = useMemo(() => {
    if (!camId) return camerasWithStatus[0] ?? null
    return camerasWithStatus.find((c) => c.camera_id === camId) ?? camerasWithStatus[0] ?? null
  }, [camerasWithStatus, camId])

  const videoPath = selectedCam?.videoPath ?? null
  const effectiveCamId = selectedCam?.camera_id ?? camId

  const labels = useMemo(
    () => getSiteAndCam(hierarchy, siteId ?? undefined, effectiveCamId),
    [hierarchy, siteId, effectiveCamId]
  )

  const [zones, setZones] = useState<ZonesResponse['zones'] | null>(null)
  const [streams, setStreams] = useState<StreamsResponse['streams']>([])
  const [counting, setCounting] = useState<CountingResponse | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [videoInfo, setVideoInfo] = useState<{ width: number; height: number } | null>(null)
  const presenceAtStartRef = useRef<number>(0)

  const camerasWithActive = useMemo(() => {
    return camerasWithStatus.map((c) => ({
      ...c,
      isActive: streams.some((s) => s.video === c.videoPath && s.active),
    }))
  }, [camerasWithStatus, streams])

  const loadData = async () => {
    if (!videoPath) return
    try {
      const [zRes, sRes, cRes] = await Promise.all([
        fetchZones(videoPath),
        fetchStreams(),
        fetchCounting(videoPath).catch(() => null),
      ])
      setZones(zRes.zones || {})
      setStreams(sRes.streams || [])
      setCounting(cRes)
    } catch {
      setZones({})
    }
  }

  useEffect(() => {
    if (!videoPath) return
    let cancelled = false
    let intervalId: ReturnType<typeof setInterval> | null = null
    const run = async () => {
      await syncBenefitZones().catch(() => {})
      if (cancelled) return
      await loadData()
      if (cancelled) return
      intervalId = setInterval(loadData, 3000)
    }
    run()
    return () => {
      cancelled = true
      if (intervalId) clearInterval(intervalId)
    }
  }, [videoPath])

  // Start/stop backend processing when video plays/pauses
  useEffect(() => {
    if (!videoPath) return
    if (isStreaming) {
      // Capture presence_time at session start to compute delta later
      const detBenefit = benefits.find((b) => String(b.skill || '').toLowerCase() === 'detection')
      const zoneId = detBenefit?.benefit_id
      const startPresence = zoneId && zones ? (zones[zoneId]?.total_time ?? 0) : 0
      presenceAtStartRef.current = startPresence
      startStream(videoPath).catch(() => {})
    } else {
      stopStream(videoPath).catch(() => {})
      setDetections([])
    }
  }, [isStreaming, videoPath])

  // Poll detections at high frequency when streaming
  useEffect(() => {
    if (!videoPath || !isStreaming) return
    let cancelled = false
    const poll = async () => {
      try {
        const res = await fetchDetections(videoPath)
        if (!cancelled) setDetections(res.detections)
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 500)
    return () => { cancelled = true; clearInterval(interval) }
  }, [isStreaming, videoPath])

  useEffect(() => {
    loadSkillsConfig().catch(() => {})
  }, [])

  useEffect(() => {
    if (videoPath) {
      fetchVideoInfo(videoPath)
        .then((info) => setVideoInfo({ width: info.width, height: info.height }))
        .catch(() => setVideoInfo(null))
    } else {
      setVideoInfo(null)
    }
  }, [videoPath])

  const handleSelectCamera = (id: string) => {
    setSearchParams((p) => {
      const next = new URLSearchParams(p)
      next.set('cam', id)
      return next
    })
  }

  const handleStreamToggle = () => {
    if (!videoPath) return
    setIsStreaming((prev) => !prev)
  }

  const handleStopAll = async () => {
    setIsStreaming(false)
    try {
      await stopAllStreams()
      setIsStreaming(false)
      const sRes = await fetchStreams()
      setStreams(sRes.streams || [])
    } catch (err) {
      console.warn('Stop all failed', err)
    }
  }

  const zoneCount = zones ? Object.keys(zones).length : 0
  const camCount = camerasWithStatus.length
  const occZones = zones
    ? Object.values(zones).filter((z) => z.is_occupied).length
    : 0
  const avgOcc = camCount > 0 ? Math.round((occZones / Math.max(zoneCount, 1)) * 100) : 0

  const [activeTab, setActiveTab] = useState<TrackerTab>('overview')
  const selectedBenefitId = searchParams.get('benefit')
  const [benefitModalOpen, setBenefitModalOpen] = useState(false)
  const [benefitModalMode, setBenefitModalMode] = useState<'create' | 'edit'>('create')
  const [benefitModalBenefitId, setBenefitModalBenefitId] = useState<string | null>(null)

  // Ouvrir le modal création quand on arrive depuis l'Explorer "+ Créer un bénéfice"
  useEffect(() => {
    if (searchParams.get('addBenefit') === '1' && selectedCam) {
      setBenefitModalBenefitId(null)
      setBenefitModalMode('create')
      setBenefitModalOpen(true)
      setSearchParams((p) => {
        const next = new URLSearchParams(p)
        next.delete('addBenefit')
        return next
      })
    }
  }, [searchParams, selectedCam, setSearchParams])
  const benefits = useMemo(() => {
    if (!selectedCam?.benefits) return []
    return Object.values(selectedCam.benefits)
  }, [selectedCam?.benefits])

  const selectedBenefit = useMemo(() => {
    if (!selectedBenefitId) return null
    return benefits.find((b) => b.benefit_id === selectedBenefitId) ?? null
  }, [benefits, selectedBenefitId])

  /* Optimistic: griser les zones dès le toggle off, sans attendre le refetch */
  const [zoneActiveOptimistic, setZoneActiveOptimistic] = useState<boolean | null>(null)
  useEffect(() => {
    setZoneActiveOptimistic(null)
  }, [selectedBenefitId])
  const zoneActive = zoneActiveOptimistic ?? (selectedBenefit?.active !== false)

  /* Timer: s'arrête quand le bénéfice est désactivé (toggle off) */
  const timerActive = isStreaming && zoneActive
  const sessionElapsed = useSessionTimer(timerActive)

  if (hierarchyLoading) {
    return (
      <div className={styles.page}>
        <div className={styles.loading}>Chargement…</div>
      </div>
    )
  }

  const showPlaceholder = !siteId || !site
  const hasActiveStreams = streams.some((s) => s.active)

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Zone Tracker</h1>
          {showPlaceholder && (
            <p className={styles.subtitle}>Surveillance et analyse du temps de présence</p>
          )}
        </div>
        <div className={styles.headerActions}>
          {/* Steps (1 Caméra, 2 Zones, 3 Lancer) */}
          {siteId && (
            <div className={styles.steps}>
              <div className={styles.step} title="Caméra">
                <img src={icon('camera')} className={styles.stepIcon} alt="" />
                <span className={styles.stepNum}>{camCount}</span>
              </div>
              <div className={styles.step} title="Zones">
                <img src={icon('zone')} className={styles.stepIcon} alt="" />
                <span className={styles.stepNum}>{zoneCount}</span>
              </div>
              <div className={styles.step} title="Lancer">
                <img src={icon('check')} className={styles.stepIcon} alt="" />
                <span className={styles.stepNum}>{avgOcc}%</span>
              </div>
            </div>
          )}
          {videoPath && (
            <div className={styles.trackerActions}>
              <button
                type="button"
                className={`${styles.actionBtn} ${isStreaming ? styles.actionBtnOn : ''}`}
                onClick={handleStreamToggle}
                title={isStreaming ? 'Pause vidéo' : 'Lancer la vidéo'}
              >
                {isStreaming ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="4" width="4" height="16" />
                    <rect x="14" y="4" width="4" height="16" />
                  </svg>
                ) : (
                  <img src={icon('play')} className={styles.actionIcon} alt="" />
                )}
              </button>
              <button
                type="button"
                className={styles.actionBtn}
                onClick={handleStopAll}
                disabled={!hasActiveStreams}
                title="Tout arrêter"
              >
                <img src={icon('stop')} className={styles.actionIcon} alt="" />
              </button>
            </div>
          )}
          <div className={`${styles.statusBadge} ${isStreaming ? styles.statusStreaming : styles.statusReady}`}>
            <span className={styles.statusDot} />
            <span>{isStreaming ? 'En ligne' : 'Prêt'}</span>
          </div>
        </div>
      </header>

      <section className={styles.content}>
        {showPlaceholder ? (
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>📹</span>
            <p>Sélectionnez un site pour ouvrir le Zone Tracker</p>
          </div>
        ) : (
          <div className={styles.trackerWrapper}>
            <div className={styles.videoSection}>
              {/* Nav retour + breadcrumb (collé au-dessus de la vidéo, style vanilla) */}
              <div className={styles.trackerNav}>
                <button
                  type="button"
                  className={styles.trackerBackBtn}
                  onClick={() => site?.lieu_id && navigate(`/lieu/${site.lieu_id}`)}
                  disabled={!site?.lieu_id}
                  title={site?.lieu_id ? 'Retour au lieu' : undefined}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M19 12H5" />
                    <path d="M12 19l-7-7 7-7" />
                  </svg>
                  Lieu
                </button>
                <div className={styles.trackerBreadcrumb}>
                  <span className={styles.bcSep}>/</span>
                  <button
                    type="button"
                    className={styles.bcPart}
                    onClick={() => site?.lieu_id && navigate(`/lieu/${site.lieu_id}`)}
                    title={site?.lieu_id ? 'Voir le lieu' : undefined}
                  >
                    <img src={icon('site')} className={styles.bcIcon} alt="" />
                    <span>{labels?.siteName ?? site?.name ?? siteId ?? '—'}</span>
                  </button>
                  <span className={styles.bcSep}>/</span>
                  <div className={styles.bcCamLov}>
                    <LovDropdown
                      options={camerasWithStatus.map((cam) => ({
                        value: cam.camera_id,
                        label: cam.name || cam.camera_id,
                      }))}
                      value={effectiveCamId ?? ''}
                      onChange={(id) => id && handleSelectCamera(id)}
                      placeholder="Caméra…"
                      emptyLabel="Aucune caméra"
                      disabled={camerasWithStatus.length === 0}
                      inline
                      icon={<img src={icon('camera')} alt="" />}
                    />
                  </div>
                </div>
              </div>
              {/* Ligne 1: Video + Tabs (même hauteur) */}
              <div className={styles.trackerTopRow}>
              <div className={styles.videoCard}>
                <VideoPlayer
                  videoPath={videoPath}
                  isStreaming={isStreaming}
                  onStreamStart={handleStreamToggle}
                  zonePolygons={selectedBenefit?.zone_polygons}
                  zonePolygonTypes={selectedBenefit?.zone_polygon_types}
                  videoWidth={videoInfo?.width}
                  videoHeight={videoInfo?.height}
                  zoneRefWidth={selectedBenefit?.zone_ref_width}
                  zoneRefHeight={selectedBenefit?.zone_ref_height}
                  zoneActive={zoneActive}
                  detections={detections}
                />
              </div>

              <div className={styles.rightPanels}>
                <div className={styles.tabs}>
                  <button
                    type="button"
                    className={`${styles.tab} ${activeTab === 'overview' ? styles.tabActive : ''}`}
                    onClick={() => setActiveTab('overview')}
                  >
                    Bénéfice
                  </button>
                  <button
                    type="button"
                    className={`${styles.tab} ${activeTab === 'source' ? styles.tabActive : ''}`}
                    onClick={() => setActiveTab('source')}
                  >
                    Source
                  </button>
                  <button
                    type="button"
                    className={`${styles.tab} ${activeTab === 'settings' ? styles.tabActive : ''}`}
                    onClick={() => setActiveTab('settings')}
                  >
                    Settings
                  </button>
                </div>

                {activeTab === 'overview' && (
                  <div className={`${styles.tabContent} ${styles.tabContentOverview}`}>
                    {selectedCam ? (
                      <BenefitsOverview
                        benefits={benefits}
                        zones={zones}
                        sessionElapsed={sessionElapsed}
                        selectedBenefitId={selectedBenefitId}
                        onSelectBenefit={(benId) => {
                          setSearchParams((p) => {
                            const next = new URLSearchParams(p)
                            if (effectiveCamId) next.set('cam', effectiveCamId)
                            if (benId) next.set('benefit', benId)
                            else next.delete('benefit')
                            return next
                          })
                        }}
                        onToggleBenefit={async (benId, active) => {
                          if (benId === selectedBenefitId) setZoneActiveOptimistic(active)
                          await toggleBenefit(benId, active)
                          refetchHierarchy()
                        }}
                        onEditBenefit={(benId) => {
                          setBenefitModalBenefitId(benId)
                          setBenefitModalMode('edit')
                          setBenefitModalOpen(true)
                        }}
                      />
                    ) : (
                      <div className={styles.tabEmpty}>Sélectionnez une caméra pour voir les bénéfices.</div>
                    )}
                  </div>
                )}

                {activeTab === 'source' && (
                  <div className={styles.tabContent}>
                    <CameraGrid
                      cameras={camerasWithActive}
                      selectedCamId={effectiveCamId}
                      onSelectCamera={handleSelectCamera}
                    />
                    <div className={styles.recapGrid}>
                      <div className={styles.recapCard}>
                        <div className={styles.recapLabel}>Caméras</div>
                        <div className={styles.recapValue}>{camCount}</div>
                      </div>
                      <div className={styles.recapCard}>
                        <div className={styles.recapLabel}>Zones</div>
                        <div className={styles.recapValue}>{zoneCount}</div>
                      </div>
                      <div className={styles.recapCard}>
                        <div className={styles.recapLabel}>Présences actives</div>
                        <div className={styles.recapValue}>{occZones}</div>
                      </div>
                    </div>
                  </div>
                )}

              {activeTab === 'settings' && (
                <div className={styles.tabContent}>
                  <div className={styles.tabEmpty}>
                    {camCount > 0 ? `${camCount} caméra(s) · ${zoneCount} zone(s)` : 'Aucune caméra'}
                  </div>
                </div>
              )}
            </div>
            </div>
          </div>

            {/* Bandeau DATA ROOM en dessous */}
            {selectedCam && (
              <div className={styles.dataRoomBand}>
                <DataRoomCards
                  benefits={benefits}
                  zones={zones}
                  counting={counting}
                  videoPath={videoPath}
                  sessionElapsed={sessionElapsed}
                  presenceAtStart={presenceAtStartRef.current}
                  onRefreshCounting={loadData}
                  onAddBenefit={() => {
                    setBenefitModalBenefitId(null)
                    setBenefitModalMode('create')
                    setBenefitModalOpen(true)
                  }}
                  onEditBenefit={(benId) => {
                    setBenefitModalBenefitId(benId)
                    setBenefitModalMode('edit')
                    setBenefitModalOpen(true)
                  }}
                  onDeleteBenefit={async (benId) => {
                    try {
                      await deleteBenefit(benId)
                      refetchHierarchy()
                      if (benefitModalBenefitId === benId) {
                        setBenefitModalOpen(false)
                        setBenefitModalBenefitId(null)
                      }
                    } catch (e) {
                      console.error('Erreur suppression:', e)
                    }
                  }}
                />
              </div>
            )}
          </div>
        )}
      </section>

      {selectedCam && (
        <BenefitConfigModal
          open={benefitModalOpen}
          onClose={() => setBenefitModalOpen(false)}
          cameraId={effectiveCamId ?? ''}
          cameraName={selectedCam.name ?? selectedCam.camera_id}
          videoPath={videoPath ?? ''}
          benefit={
            benefitModalMode === 'edit' && benefitModalBenefitId
              ? benefits.find((b) => b.benefit_id === benefitModalBenefitId) ?? null
              : null
          }
          onSaved={() => refetchHierarchy()}
          onSaveError={(msg) => {
            refetchHierarchy()
            alert(`Erreur lors de l'enregistrement : ${msg}`)
          }}
        />
      )}
    </div>
  )
}
