import { useSiteStore } from '@/stores/siteStore'
import { useVideoStore } from '@/stores/videoStore'
import type { HierarchyCamera } from '@/types/site'
import styles from './CameraGrid.module.css'

export default function CameraGrid() {
  const selectedSiteId = useSiteStore((s) => s.selectedSiteId)
  const getSite = useSiteStore((s) => s.getSite)
  const site = selectedSiteId ? getSite(selectedSiteId) : undefined

  if (!site || site.cameras.length === 0) {
    return <div className={styles.empty}>Aucune caméra sur ce site</div>
  }

  return (
    <div className={styles.grid}>
      {site.cameras.map((cam) => (
        <CameraTile key={cam.id} camera={cam} />
      ))}
    </div>
  )
}

function CameraTile({ camera }: { camera: HierarchyCamera }) {
  const selectCamera = useSiteStore((s) => s.selectCamera)
  const setVideo = useVideoStore((s) => s.setCurrentVideo)
  const setCameraId = useVideoStore((s) => s.setCurrentCamera)
  const currentCameraId = useSiteStore((s) => s.selectedCameraId)
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const isSelected = currentCameraId === camera.id
  const isLive = activeStreams.has(camera.path)

  const handleClick = () => {
    selectCamera(camera.id)
    setCameraId(camera.id)
    setVideo(camera.path)
  }

  const thumbUrl = `/frame/${encodeURIComponent(camera.path)}`

  return (
    <button
      className={`${styles.tile} ${isSelected ? styles.tileSelected : ''}`}
      onClick={handleClick}
    >
      <div className={styles.thumb}>
        <img src={thumbUrl} alt={camera.name} className={styles.thumbImg} />
        {isLive && <span className={styles.liveDot} />}
      </div>
      <div className={styles.info}>
        <span className={styles.name}>{camera.name}</span>
        <span className={styles.meta}>
          {camera.benefits.length} bénéfice{camera.benefits.length !== 1 ? 's' : ''}
        </span>
      </div>
    </button>
  )
}
