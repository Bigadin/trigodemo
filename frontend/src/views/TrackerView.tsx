import { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useSiteStore } from '@/stores/siteStore'
import { useVideoStore } from '@/stores/videoStore'
import { useEditorStore } from '@/stores/editorStore'
import { useStreamsPolling } from '@/hooks/useStreamsPolling'
import { useZonesPolling } from '@/hooks/useZonesPolling'
import VideoPlayer from '@/components/video/VideoPlayer'
import CameraGrid from '@/components/video/CameraGrid'
import BenefitsPanel from '@/components/tracker/BenefitsPanel'
import DataRoom from '@/components/tracker/DataRoom'
import DetectionControls from '@/components/tracker/DetectionControls'
import CountingPanel from '@/components/tracker/CountingPanel'
import CountingParams from '@/components/tracker/CountingParams'
import ZoneEditor from '@/components/editor/ZoneEditor'
import styles from './TrackerView.module.css'

export default function TrackerView() {
  const { siteId } = useParams<{ siteId: string }>()
  const navigate = useNavigate()
  const selectSite = useSiteStore((s) => s.selectSite)
  const selectedSiteId = useSiteStore((s) => s.selectedSiteId)
  const getSite = useSiteStore((s) => s.getSite)
  const getLieuForSite = useSiteStore((s) => s.getLieuForSite)
  const lieux = useSiteStore((s) => s.lieux)
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const currentVideo = useVideoStore((s) => s.currentVideo)
  const openEditor = useEditorStore((s) => s.openEditor)

  useStreamsPolling()
  useZonesPolling()

  useEffect(() => {
    if (siteId) {
      selectSite(siteId)
    } else if (!selectedSiteId && lieux.length > 0 && lieux[0].sites.length > 0) {
      const first = lieux[0].sites[0]
      selectSite(first.id)
      navigate(`/tracker/${first.id}`, { replace: true })
    }
  }, [siteId, selectedSiteId, lieux, selectSite, navigate])

  const site = selectedSiteId ? getSite(selectedSiteId) : undefined
  const lieu = selectedSiteId ? getLieuForSite(selectedSiteId) : undefined

  const totalCams = site?.cameras.length ?? 0
  const totalBens = site?.cameras.reduce((n, c) => n + c.benefits.length, 0) ?? 0
  const activeCams = site?.cameras.filter((c) => activeStreams.has(c.path)).length ?? 0

  if (!site) {
    return (
      <div className={styles.empty}>
        <p>Sélectionnez un site dans la sidebar pour commencer</p>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.backBtn} onClick={() => navigate('/')}>
            ← Sites
          </button>
          <div>
            <h1 className={styles.title}>{site.name}</h1>
            <p className={styles.breadcrumb}>
              {lieu && <span>{lieu.name}</span>}
              <span className={styles.sep}>›</span>
              <span>{site.name}</span>
            </p>
          </div>
        </div>
        <div className={styles.kpis}>
          <div className={styles.kpi}>
            <span className={styles.kpiValue}>{totalCams}</span>
            <span className={styles.kpiLabel}>Caméras</span>
          </div>
          <div className={styles.kpi}>
            <span className={styles.kpiValue}>{totalBens}</span>
            <span className={styles.kpiLabel}>Bénéfices</span>
          </div>
          <div className={styles.kpi}>
            <span className={`${styles.kpiValue} ${activeCams > 0 ? styles.kpiGreen : ''}`}>
              {activeCams}
            </span>
            <span className={styles.kpiLabel}>Actifs</span>
          </div>
        </div>
      </div>

      {/* Detection controls + Editor button */}
      <div className={styles.controlsRow}>
        <DetectionControls />
        {currentVideo && (
          <button
            className={styles.editBtn}
            onClick={() => openEditor(currentVideo)}
          >
            ✎ Éditer zones
          </button>
        )}
      </div>

      {/* Zone Editor modal */}
      <ZoneEditor />

      {/* Main content */}
      <div className={styles.content}>
        <div className={styles.left}>
          <VideoPlayer />
          <CameraGrid />
        </div>
        <div className={styles.right}>
          <BenefitsPanel />
          <CountingPanel />
          <CountingParams />
          <DataRoom />
        </div>
      </div>
    </div>
  )
}
