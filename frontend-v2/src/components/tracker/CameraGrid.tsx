import type { HierarchyCamera } from '@/types/hierarchy'
import styles from './CameraGrid.module.css'

export interface CameraWithStatus extends HierarchyCamera {
  videoPath: string
  isActive: boolean
}

interface CameraGridProps {
  cameras: CameraWithStatus[]
  selectedCamId: string | null
  onSelectCamera: (camId: string) => void
}

function truncate(str: string, len: number): string {
  return str.length > len ? str.slice(0, len - 3) + '…' : str
}

export default function CameraGrid({
  cameras,
  selectedCamId,
  onSelectCamera,
}: CameraGridProps) {
  if (cameras.length === 0) {
    return (
      <div className={styles.empty}>
        Aucune caméra sur ce site. Cliquez sur <b>+</b> pour en ajouter une.
      </div>
    )
  }

  return (
    <div className={styles.list}>
      {cameras.map((cam) => {
        const isCurrent = cam.camera_id === selectedCamId
        const sourceType = cam.type === 'webcam' || cam.type === 'rtsp' ? cam.type.toUpperCase() : 'VIDÉO'
        const sourceDisplay = cam.type === 'video' || !cam.type ? truncate(cam.path ?? '', 14) : ''
        let statusDot = 'ready'
        let statusTitle = 'Prête'
        const hasVideo = !!cam.videoPath
        if (!hasVideo) {
          statusDot = 'missing'
          statusTitle = 'Manquante'
        } else if (cam.isActive) {
          statusDot = 'online'
          statusTitle = 'En ligne'
        }

        return (
          <button
            key={cam.camera_id}
            type="button"
            className={`${styles.tile} ${isCurrent ? styles.selected : ''}`}
            onClick={() => onSelectCamera(cam.camera_id)}
            title={cam.name || cam.camera_id}
          >
            <span
              className={`${styles.dot} ${styles[statusDot as keyof typeof styles]}`}
              title={statusTitle}
            />
            <div className={styles.body}>
              <span className={styles.name} title={cam.name || cam.camera_id}>
                {(cam.name || cam.camera_id).toUpperCase()}
              </span>
              <span className={styles.meta}>
                {sourceType}
                {sourceDisplay ? ` · ${sourceDisplay}` : ''}
              </span>
            </div>
          </button>
        )
      })}
    </div>
  )
}
