import { useMemo } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import styles from './VideoPlayer.module.css'

export default function VideoPlayer() {
  const currentVideo = useVideoStore((s) => s.currentVideo)
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const isLive = currentVideo ? activeStreams.has(currentVideo) : false

  const streamUrl = useMemo(() => {
    if (!currentVideo) return null
    if (isLive) return `/video_feed/${encodeURIComponent(currentVideo)}`
    return `/frame/${encodeURIComponent(currentVideo)}`
  }, [currentVideo, isLive])

  if (!currentVideo) {
    return (
      <div className={styles.placeholder}>
        <span className={styles.placeholderIcon}>📷</span>
        <span>Sélectionnez une caméra</span>
      </div>
    )
  }

  return (
    <div className={styles.player}>
      <div className={styles.videoWrap}>
        <img
          className={styles.frame}
          src={streamUrl ?? undefined}
          alt={currentVideo}
          key={`${currentVideo}-${isLive}`}
        />
        {isLive && <span className={styles.liveBadge}>LIVE</span>}
      </div>
      <div className={styles.label}>
        {currentVideo}
      </div>
    </div>
  )
}
