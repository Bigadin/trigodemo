import { useEffect, useMemo, useRef, useState } from 'react'
import type { Detection } from '@/api/tracker'
import { getStreamUrl, getFrameUrl } from '@/api/tracker'
import { computeZoneOccupancy } from '@/utils/zoneOccupancy'
import styles from './VideoPlayer.module.css'

interface VideoPlayerProps {
  videoPath: string | null
  isStreaming: boolean
  resetTrigger?: number
  onStreamStart?: () => void
  zonePolygons?: number[][][]
  zonePolygonTypes?: ('include' | 'exclude')[]
  videoWidth?: number
  videoHeight?: number
  zoneRefWidth?: number
  zoneRefHeight?: number
  zoneActive?: boolean
  detections?: Detection[]
}

const ZONE_COLORS = {
  includeIdle:   { fill: 'rgba(34,197,94,0.22)',   stroke: 'rgba(34,197,94,0.75)' },
  includeActive: { fill: 'rgba(34,197,94,0.35)',   stroke: 'rgba(34,197,94,0.95)' },
  exclude:       { fill: 'rgba(240,131,33,0.25)',  stroke: 'rgba(240,131,33,0.8)' },
  inactive:      { fill: 'rgba(148,163,184,0.22)', stroke: 'rgba(148,163,184,0.7)' },
} as const

export default function VideoPlayer({ videoPath, isStreaming, resetTrigger, onStreamStart, zonePolygons, zonePolygonTypes, videoWidth, videoHeight, zoneRefWidth, zoneRefHeight, zoneActive = true, detections }: VideoPlayerProps) {
  // Mirror the same coordinate logic as BenefitConfigModal:
  // zone_ref_width/height → videoWidth/height → 1280/720 fallback
  const viewW = (zoneRefWidth != null && zoneRefWidth > 0) ? zoneRefWidth
    : (videoWidth != null && videoWidth > 0) ? videoWidth : 1280
  const viewH = (zoneRefHeight != null && zoneRefHeight > 0) ? zoneRefHeight
    : (videoHeight != null && videoHeight > 0) ? videoHeight : 720
  const vidW = (videoWidth != null && videoWidth > 0) ? videoWidth : 1920
  const vidH = (videoHeight != null && videoHeight > 0) ? videoHeight : 1080
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => { setError(null) }, [videoPath, isStreaming])

  useEffect(() => {
    const el = videoRef.current
    if (!el) return
    if (isStreaming) {
      el.play().catch(() => setError('Impossible de lire la vidéo'))
    } else {
      el.pause()
    }
  }, [isStreaming])

  useEffect(() => {
    const el = videoRef.current
    if (!el || resetTrigger == null || resetTrigger < 1) return
    el.currentTime = 0
  }, [resetTrigger])

  // Compute zone occupancy from detections (frontend logic).
  // Both zone polygons and YOLO detections are in video-pixel coords,
  // so we only need to rescale if an explicit zoneRef differs from the video resolution.
  const zoneOccupancy = useMemo(() => {
    if (!zonePolygons || !detections || detections.length === 0) return null
    const scaleX = viewW / vidW
    const scaleY = viewH / vidH
    return computeZoneOccupancy(zonePolygons, zonePolygonTypes, detections, scaleX, scaleY)
  }, [zonePolygons, zonePolygonTypes, detections, viewW, viewH, vidW, vidH])

  if (!videoPath) {
    return (
      <div className={styles.placeholder}>
        <span className={styles.placeholderIcon}>📹</span>
        <p>Sélectionnez une caméra dans l&apos;explorateur</p>
      </div>
    )
  }

  const streamSrc = getStreamUrl(videoPath, true)
  const isCamera = videoPath.startsWith('camera:')
  const videoSrc = isCamera ? getFrameUrl(videoPath) : `/videos/${videoPath}`
  const hasZones = zonePolygons && zonePolygons.length > 0
  const hasDetections = detections && detections.length > 0

  // Pour aligner l'overlay sur le flux/vidéo : viewBox = dimensions vidéo, scaler les polygones zone_ref → vidéo
  const overlayViewW = vidW
  const overlayViewH = vidH
  const refW = (zoneRefWidth != null && zoneRefWidth > 0) ? zoneRefWidth : vidW
  const refH = (zoneRefHeight != null && zoneRefHeight > 0) ? zoneRefHeight : vidH
  const scaleToVideoX = overlayViewW / refW
  const scaleToVideoY = overlayViewH / refH

  return (
    <div className={styles.wrapper}>
      {isStreaming && isCamera ? (
        <img
          className={styles.video}
          src={streamSrc}
          alt="Stream vidéo"
          onError={() => setError('Impossible de charger le stream')}
          onLoad={() => setError(null)}
        />
      ) : isStreaming && !isCamera ? (
        <video
          ref={videoRef}
          className={styles.video}
          src={videoSrc}
          muted
          loop
          playsInline
          autoPlay
          onError={() => setError('Impossible de charger la vidéo')}
          onCanPlay={() => setError(null)}
        />
      ) : !isStreaming && isCamera ? (
        <img
          className={styles.video}
          src={videoSrc}
          alt="Aperçu caméra"
          onError={() => setError('Impossible de charger l\'aperçu')}
          onLoad={() => setError(null)}
        />
      ) : (
        <video
          ref={videoRef}
          className={styles.video}
          src={videoSrc}
          muted
          loop
          playsInline
          preload="metadata"
          onError={() => setError('Impossible de charger la vidéo')}
          onCanPlay={() => setError(null)}
        />
      )}

      {/* Zone polygons overlay — affiché pour le bénéfice sélectionné */}
      {hasZones && (
        <svg
          className={styles.zoneOverlay}
          viewBox={`0 0 ${overlayViewW} ${overlayViewH}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {zonePolygons!.map((poly, idx) => {
            if (!Array.isArray(poly) || poly.length < 3) return null
            const pts = poly
              .filter((p): p is [number, number] => Array.isArray(p) && typeof p[0] === 'number' && typeof p[1] === 'number')
              .map((p) => `${p[0] * scaleToVideoX},${p[1] * scaleToVideoY}`)
              .join(' ')
            if (pts.length === 0) return null

            const isInclude = (zonePolygonTypes?.[idx] ?? 'include') === 'include'
            const occupied = zoneOccupancy?.[idx]?.occupied ?? false

            let colors: { fill: string; stroke: string }
            if (!zoneActive) {
              colors = ZONE_COLORS.inactive
            } else if (!isInclude) {
              colors = ZONE_COLORS.exclude
            } else if (occupied) {
              colors = ZONE_COLORS.includeActive
            } else {
              colors = ZONE_COLORS.includeIdle
            }

            return (
              <g key={idx}>
                <polygon
                  points={pts}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={occupied && isInclude && zoneActive ? 3 : 2.5}
                  className={styles.zonePolygon}
                />
                {occupied && isInclude && zoneActive && (
                  <polygon
                    points={pts}
                    fill="none"
                    stroke={colors.stroke}
                    strokeWidth={3}
                    className={styles.zonePulse}
                  />
                )}
              </g>
            )
          })}
        </svg>
      )}

      {/* YOLO detection bounding boxes overlay — toujours affiché quand on a des détections (human.pt) */}
      {hasDetections && (
        <svg
          className={styles.zoneOverlay}
          viewBox={`0 0 ${vidW} ${vidH}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {detections!.map((det, idx) => {
            const w = det.x2 - det.x1
            const h = det.y2 - det.y1
            return (
              <g key={det.track_id ?? idx}>
                <rect
                  x={det.x1} y={det.y1} width={w} height={h}
                  fill="none" stroke="#00ff88" strokeWidth={3} rx={2}
                />
                <rect
                  x={det.x1} y={det.y1 - 22}
                  width={det.track_id != null ? 90 : 50} height={20}
                  fill="rgba(0,0,0,0.7)" rx={2}
                />
                <text
                  x={det.x1 + 4} y={det.y1 - 6}
                  fill="#00ff88" fontSize={14} fontFamily="monospace" fontWeight="bold"
                >
                  {det.track_id != null ? `#${det.track_id} ` : ''}{Math.round(det.conf * 100)}%
                </text>
              </g>
            )
          })}
        </svg>
      )}

      {error && (
        <div className={styles.error}>
          <span>{error}</span>
          {onStreamStart && (
            <button type="button" className={styles.retryBtn} onClick={onStreamStart}>
              {isStreaming ? 'Réessayer le stream' : 'Lancer la vidéo'}
            </button>
          )}
        </div>
      )}
      {isStreaming && (
        <span className={styles.liveBadge}>LIVE</span>
      )}
    </div>
  )
}
