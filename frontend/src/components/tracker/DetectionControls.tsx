import { useVideoStore } from '@/stores/videoStore'
import styles from './DetectionControls.module.css'

export default function DetectionControls() {
  const currentVideo = useVideoStore((s) => s.currentVideo)
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const startDetection = useVideoStore((s) => s.startDetection)
  const stopDetection = useVideoStore((s) => s.stopDetection)
  const stopAll = useVideoStore((s) => s.stopAll)

  const isLive = currentVideo ? activeStreams.has(currentVideo) : false
  const anyActive = activeStreams.size > 0

  const handleToggle = async () => {
    if (!currentVideo) return
    if (isLive) {
      await stopDetection(currentVideo)
    } else {
      await startDetection(currentVideo)
    }
  }

  return (
    <div className={styles.controls}>
      <button
        className={`${styles.btn} ${isLive ? styles.btnDanger : styles.btnPrimary}`}
        onClick={handleToggle}
        disabled={!currentVideo}
      >
        {isLive ? '⏹ Stop' : '▶ Détecter'}
      </button>

      {anyActive && (
        <button
          className={`${styles.btn} ${styles.btnGhost}`}
          onClick={stopAll}
        >
          ⏹ Stop All ({activeStreams.size})
        </button>
      )}

      <div className={styles.status}>
        {activeStreams.size > 0 && (
          <span className={styles.activeBadge}>
            {activeStreams.size} stream{activeStreams.size > 1 ? 's' : ''} actif{activeStreams.size > 1 ? 's' : ''}
          </span>
        )}
      </div>
    </div>
  )
}
