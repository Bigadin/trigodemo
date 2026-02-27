import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import type { Detection } from '@/api/tracker'
import { getStreamUrl, getFrameUrl } from '@/api/tracker'
import { computeZoneOccupancy } from '@/utils/zoneOccupancy'
import styles from './VideoPlayer.module.css'

export interface ZoneHoverStat {
  label: string
  isInclude: boolean
  isOccupied: boolean
  presenceTimeSec: number
  pct: number
}

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
  zoneHoverStats?: (ZoneHoverStat | null)[] | null
}

const ZONE_COLORS = {
  includeIdle:   { fill: 'rgba(34,197,94,0.22)',   stroke: 'rgba(34,197,94,0.75)' },
  includeActive: { fill: 'rgba(34,197,94,0.35)',   stroke: 'rgba(34,197,94,0.95)' },
  exclude:       { fill: 'rgba(240,131,33,0.25)',  stroke: 'rgba(240,131,33,0.8)' },
  inactive:      { fill: 'rgba(148,163,184,0.22)', stroke: 'rgba(148,163,184,0.7)' },
} as const

function formatTime(seconds: number): string {
  const n = Math.max(0, Math.floor(seconds))
  const h = Math.floor(n / 3600)
  const m = Math.floor((n % 3600) / 60)
  const s = n % 60
  return [h, m, s].map((v) => v.toString().padStart(2, '0')).join(':')
}

function clampPct(value: number): number {
  return Math.min(100, Math.max(0, value))
}

function getPctColor(pct: number): string {
  const p = clampPct(pct) / 100
  // Interpole bleu GPU -> violet/magenta en gardant une bonne lisibilité.
  const hue = 216 + (300 - 216) * p
  const sat = 88
  const light = 68
  return `hsl(${hue} ${sat}% ${light}%)`
}

export default function VideoPlayer({ videoPath, isStreaming, resetTrigger, onStreamStart, zonePolygons, zonePolygonTypes, videoWidth, videoHeight, zoneRefWidth, zoneRefHeight, zoneActive = true, detections, zoneHoverStats }: VideoPlayerProps) {
  // Mirror the same coordinate logic as BenefitConfigModal:
  // zone_ref_width/height → videoWidth/height → 1280/720 fallback
  const viewW = (zoneRefWidth != null && zoneRefWidth > 0) ? zoneRefWidth
    : (videoWidth != null && videoWidth > 0) ? videoWidth : 1280
  const viewH = (zoneRefHeight != null && zoneRefHeight > 0) ? zoneRefHeight
    : (videoHeight != null && videoHeight > 0) ? videoHeight : 720
  const vidW = (videoWidth != null && videoWidth > 0) ? videoWidth : 1920
  const vidH = (videoHeight != null && videoHeight > 0) ? videoHeight : 1080
  const [error, setError] = useState<string | null>(null)
  const [hoverState, setHoverState] = useState<{ x: number; y: number; stat: ZoneHoverStat } | null>(null)
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
          className={`${styles.zoneOverlay} ${styles.zoneOverlayInteractive}`}
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

            const stat = zoneHoverStats?.[idx] ?? null

            const handleZoneHover = (e: MouseEvent<SVGPolygonElement>) => {
              if (!stat || !stat.isInclude) return
              const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect()
              if (!box) return
              setHoverState({
                x: e.clientX - box.left,
                y: e.clientY - box.top,
                stat,
              })
            }

            return (
              <g key={idx}>
                <polygon
                  points={pts}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={occupied && isInclude && zoneActive ? 3 : 2.5}
                  className={styles.zonePolygon}
                  style={stat?.isInclude ? { pointerEvents: 'none' } : undefined}
                />
                {/* Zone de hit élargie (stroke 24px) au-dessus : formes modifiées / auto-intersectantes */}
                {stat?.isInclude && (
                  <polygon
                    points={pts}
                    fill="none"
                    stroke="rgba(0,0,0,0.001)"
                    strokeWidth={24}
                    style={{ pointerEvents: 'all' }}
                    onMouseMove={handleZoneHover}
                    onMouseLeave={() => setHoverState(null)}
                  />
                )}
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
            const labelX = det.x2 + 6
            const labelY1 = det.y1
            const labelH = 18
            const labelGap = 4
            const pastilleSize = 6
            const classLabel = 'Humain'
            const idConfLabel = det.track_id != null ? `#${det.track_id} ${Math.round(det.conf * 100)}%` : `${Math.round(det.conf * 100)}%`
            const labelW1 = Math.max(60, classLabel.length * 7)
            const labelW2 = Math.max(70, idConfLabel.length * 7)
            return (
              <g key={det.track_id ?? idx}>
                <rect
                  x={det.x1} y={det.y1} width={w} height={h}
                  fill="none" stroke="#1a1a1a" strokeWidth={2} rx={2}
                />
                {/* Label 1 : classe */}
                <rect x={labelX} y={labelY1} width={labelW1} height={labelH} fill="rgba(0,0,0,0.88)" rx={2} />
                <rect x={labelX + 5} y={labelY1 + (labelH - pastilleSize) / 2} width={pastilleSize} height={pastilleSize} fill="#fff" rx={1} />
                <text x={labelX + 5 + pastilleSize + 5} y={labelY1 + labelH / 2} fill="#e5e7eb" fontSize={11} fontFamily="var(--font-sans)" fontWeight="500" dominantBaseline="middle">
                  {classLabel}
                </text>
                {/* Label 2 : ID + conf */}
                <rect x={labelX} y={labelY1 + labelH + labelGap} width={labelW2} height={labelH} fill="rgba(0,0,0,0.88)" rx={2} />
                <rect x={labelX + 5} y={labelY1 + labelH + labelGap + (labelH - pastilleSize) / 2} width={pastilleSize} height={pastilleSize} fill="#fff" rx={1} />
                <text x={labelX + 5 + pastilleSize + 5} y={labelY1 + labelH + labelGap + labelH / 2} fill="#e5e7eb" fontSize={11} fontFamily="var(--font-sans)" fontWeight="500" dominantBaseline="middle">
                  {idConfLabel}
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
      {hoverState && (
        <div
          className={styles.zoneHoverTip}
          style={{
            left: `${Math.min(Math.max(hoverState.x + 12, 8), 260)}px`,
            top: `${Math.max(hoverState.y - 12, 8)}px`,
          }}
        >
          {(() => {
            const pct = clampPct(hoverState.stat.pct)
            const color = getPctColor(pct)
            const bgSize = pct > 0 ? `${(100 / pct) * 100}% 100%` : '100% 100%'
            return (
              <>
          <div className={styles.zoneHoverTitle}>{hoverState.stat.label}</div>
          <div className={styles.zoneHoverRow}>
            <span className={styles.zoneHoverKey}>Etat</span>
            <span className={styles.zoneHoverValue}>{hoverState.stat.isOccupied ? 'Présent' : 'Absent'}</span>
          </div>
          <div className={styles.zoneHoverRow}>
            <span className={styles.zoneHoverKey}>Temps</span>
            <span className={styles.zoneHoverValue}>{formatTime(hoverState.stat.presenceTimeSec)}</span>
          </div>
          <div className={styles.zoneHoverBarBlock}>
            <div className={styles.zoneHoverRow}>
              <span className={styles.zoneHoverKey}>Présence</span>
              <span className={styles.zoneHoverValue} style={{ color }}>{pct}%</span>
            </div>
            <div className={styles.zoneHoverBarTrack}>
              <div
                className={styles.zoneHoverBarFill}
                style={{ width: `${pct}%`, backgroundSize: bgSize }}
              />
            </div>
          </div>
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}
