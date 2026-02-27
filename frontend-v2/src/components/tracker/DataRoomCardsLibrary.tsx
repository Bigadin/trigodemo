/**
 * Bibliothèque de cartes Data Room — composants réutilisables.
 * Carte Rapport d'activité avec courbes (version graphique, mise de côté pour réutilisation ultérieure).
 */
import { useState, useEffect, useCallback, useRef, useMemo, type MouseEvent } from 'react'
import type { HierarchyBenefit } from '@/types/hierarchy'
import type { ZoneData } from '@/api/tracker'
import CardMenu from '@/components/ui/CardMenu'
import { getDetectionZoneKeys } from './DataRoomCards'
import styles from './DataRoomCards.module.css'

export const ACTIVITY_CHART_INTERVAL_OPTIONS = [
  { value: 2, label: '2 min' },
  { value: 5, label: '5 min' },
  { value: 10, label: '10 min' },
  { value: 15, label: '15 min' },
] as const

const TIME_INTERVAL_OPTIONS = ACTIVITY_CHART_INTERVAL_OPTIONS

interface ChartPoint {
  t: number
  value: number
}

function formatTimeLabel(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t
  const t3 = t2 * t
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
}

const ACTIVITY_SERIES_COLORS = ['#3b82f6', '#4f46e5', '#7c3aed', '#8b5cf6', '#a855f7', '#c026d3']

interface ActivitySeries {
  label: string
  color: string
  points: ChartPoint[]
  displayValue: number
}

const CHART_PAD_LEFT = 44
const CHART_PAD_RIGHT = 20
const CHART_PAD_TOP = 14
const CHART_PAD_BOTTOM = 30

/** Paddings réduits pour affichage compact */
const CHART_PAD_LEFT_COMPACT = 36
const CHART_PAD_RIGHT_COMPACT = 12
const CHART_PAD_TOP_COMPACT = 8
const CHART_PAD_BOTTOM_COMPACT = 18

function getSeriesValueAtTime(s: ActivitySeries, t: number, endT: number): number {
  let pts = s.points.length > 0 ? s.points : [{ t: 0, value: 0 }]
  if (pts.length > 0 && pts[0].t > 0) pts = [{ t: 0, value: 0 }, ...pts]
  const lastT = pts[pts.length - 1].t
  const lastV = pts[pts.length - 1].value
  if (pts.length <= 1) return pts[0]?.value ?? 0
  if (t <= pts[0].t) return pts[0].value
  if (t >= lastT) {
    if (t <= endT && endT > lastT) return lastV + (s.displayValue - lastV) * ((t - lastT) / (endT - lastT))
    return s.displayValue
  }
  let i = 0
  while (i < pts.length - 1 && pts[i + 1].t < t) i++
  const p0 = pts[Math.max(0, i - 1)]
  const p1 = pts[i]
  const p2 = pts[Math.min(pts.length - 1, i + 1)]
  const p3 = pts[Math.min(pts.length - 1, i + 2)]
  const segT = (p2.t - p1.t) > 0 ? (t - p1.t) / (p2.t - p1.t) : 1
  return catmullRom(p0.value, p1.value, p2.value, p3.value, segT)
}

function drawExpertActivityChart(
  ctx: CanvasRenderingContext2D,
  series: ActivitySeries[],
  totalDuration: number,
  currentTime: number,
  w: number,
  h: number,
  bgColor: string,
  gridColor: string,
  tickColor: string,
  compact = false
) {
  if (w <= 0 || h <= 0 || series.length === 0) return
  const padLeft = compact ? CHART_PAD_LEFT_COMPACT : CHART_PAD_LEFT
  const padRight = compact ? CHART_PAD_RIGHT_COMPACT : CHART_PAD_RIGHT
  const padTop = compact ? CHART_PAD_TOP_COMPACT : CHART_PAD_TOP
  const padBottom = compact ? CHART_PAD_BOTTOM_COMPACT : CHART_PAD_BOTTOM
  const drawW = w - padLeft - padRight
  const drawH = h - padTop - padBottom
  const allValues = series.flatMap((s) => s.points.map((p) => p.value)).concat(series.map((s) => s.displayValue))
  const maxVal = Math.max(1, 100, ...allValues)
  const toX = (t: number) => padLeft + (t / totalDuration) * drawW
  const toY = (val: number) => h - padBottom - (val / maxVal) * drawH

  ctx.fillStyle = bgColor
  ctx.fillRect(0, 0, w, h)
  const gridLinesH = compact ? 3 : 5
  const gridLinesV = compact ? 5 : 8
  /* Traits horizontaux supprimés pour un rendu plus épuré */
  /* Traits verticaux : très discrets */
  ctx.strokeStyle = gridColor
  ctx.globalAlpha = compact ? 0.1 : 0.15
  ctx.lineWidth = 1
  for (let i = 1; i < gridLinesV; i++) {
    const x = padLeft + (drawW / gridLinesV) * i + 0.5
    ctx.beginPath()
    ctx.moveTo(x, padTop)
    ctx.lineTo(x, h - padBottom)
    ctx.stroke()
  }
  ctx.globalAlpha = 1
  ctx.font = compact ? '9px Manrope, system-ui, sans-serif' : '10px Manrope, system-ui, sans-serif'
  ctx.fillStyle = tickColor
  ctx.textAlign = 'center'
  const tickY = compact ? h - 6 : h - 10
  for (let i = 0; i <= gridLinesV; i++) {
    const t = (i / gridLinesV) * totalDuration
    if (t > 0) ctx.fillText(formatTimeLabel(t), toX(t), tickY)
  }
  ctx.textAlign = 'right'
  for (let i = 0; i <= gridLinesH; i++) {
    const y = padTop + (drawH / gridLinesH) * i
    ctx.fillText(String(Math.round(maxVal * (1 - i / gridLinesH))), padLeft - 8, y + 4)
  }
  ctx.textAlign = 'left'
  const endT = Math.min(currentTime, totalDuration)

  for (const s of series) {
    let pts = s.points.length > 0 ? s.points : [{ t: 0, value: 0 }]
    if (pts.length > 0 && pts[0].t > 0) pts = [{ t: 0, value: 0 }, ...pts]
    const lastT = pts[pts.length - 1].t
    const lastV = pts[pts.length - 1].value
    const getSmoothY = (t: number): number => {
      if (pts.length <= 1) return toY(pts[0]?.value ?? 0)
      if (t <= pts[0].t) return toY(pts[0].value)
      if (t >= pts[pts.length - 1].t) return toY(pts[pts.length - 1].value)
      let i = 0
      while (i < pts.length - 1 && pts[i + 1].t < t) i++
      const p0 = pts[Math.max(0, i - 1)]
      const p1 = pts[i]
      const p2 = pts[Math.min(pts.length - 1, i + 1)]
      const p3 = pts[Math.min(pts.length - 1, i + 2)]
      const segT = (p2.t - p1.t) > 0 ? (t - p1.t) / (p2.t - p1.t) : 1
      return toY(catmullRom(p0.value, p1.value, p2.value, p3.value, segT))
    }
    ctx.beginPath()
    ctx.moveTo(toX(0), h - padBottom)
    ctx.lineTo(toX(0), getSmoothY(0))
    for (let k = 1; k <= 100; k++) {
      const t = (k / 100) * Math.min(lastT, endT)
      if (t > lastT) break
      ctx.lineTo(toX(t), getSmoothY(t))
    }
    if (lastT < endT) {
      for (let k = 1; k <= 25; k++) {
        const t = lastT + (k / 25) * (endT - lastT)
        const v = lastV + (s.displayValue - lastV) * (k / 25)
        ctx.lineTo(toX(t), toY(v))
      }
    }
    ctx.lineTo(toX(endT), h - padBottom)
    ctx.closePath()
    const [r, g, b] = s.color.startsWith('#')
      ? [parseInt(s.color.slice(1, 3), 16), parseInt(s.color.slice(3, 5), 16), parseInt(s.color.slice(5, 7), 16)]
      : [0, 0, 0]
    ctx.fillStyle = `rgba(${r},${g},${b},0.25)`
    ctx.fill()
    ctx.strokeStyle = s.color
    ctx.lineWidth = 1.5
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    ctx.moveTo(toX(0), getSmoothY(0))
    const strokeEndT = Math.min(lastT, endT)
    for (let k = 1; k <= 100; k++) {
      const t = (k / 100) * strokeEndT
      if (t > strokeEndT) break
      ctx.lineTo(toX(t), getSmoothY(t))
    }
    if (lastT < endT) {
      for (let k = 1; k <= 25; k++) {
        const t = lastT + (k / 25) * (endT - lastT)
        const v = lastV + (s.displayValue - lastV) * (k / 25)
        ctx.lineTo(toX(t), toY(v))
      }
    }
    ctx.stroke()
  }
}

export interface ActivityChartProps {
  zoneItems: { zoneKey: string; label: string }[]
  zones: Record<string, ZoneData> | null
  sessionElapsed: number
  presenceAtStartMap: Record<string, number>
  intervalMinutes?: number
  compact?: boolean
  className?: string
}

/** Composant courbe d'activité réutilisable (compact ou pleine hauteur) */
export function ActivityChart({
  zoneItems,
  zones,
  sessionElapsed,
  presenceAtStartMap,
  intervalMinutes = 2,
  compact = false,
  className,
}: ActivityChartProps) {
  interface HoverState {
    x: number
    left: number
    time: number
    rows: { label: string; color: string; value: number }[]
  }
  const [hover, setHover] = useState<HoverState | null>(null)
  const [pointsPerZone, setPointsPerZone] = useState<Record<string, ChartPoint[]>>({})
  const lastSampleRef = useRef(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const displayedValuesRef = useRef<Record<string, number>>({})
  const rafRef = useRef<number>(0)

  const targetPctPerZone = useMemo(() => {
    const out: Record<string, number> = {}
    for (const { zoneKey } of zoneItems) {
      const zone = zones?.[zoneKey]
      const totalPresence = zone?.total_time ?? 0
      const start = presenceAtStartMap[zoneKey] ?? 0
      const presenceTime = Math.max(0, totalPresence - start)
      out[zoneKey] = sessionElapsed > 0 ? Math.min(100, (presenceTime / sessionElapsed) * 100) : 0
    }
    return out
  }, [zoneItems, zones, sessionElapsed, presenceAtStartMap])

  const totalDuration = intervalMinutes * 60
  const currentTime = Math.min(sessionElapsed, totalDuration)

  useEffect(() => {
    const sampleInterval = 15
    if (sessionElapsed - lastSampleRef.current >= sampleInterval || Object.keys(pointsPerZone).length === 0) {
      lastSampleRef.current = sessionElapsed
      setPointsPerZone((prev) => {
        const maxPoints = totalDuration / sampleInterval
        const next = { ...prev }
        for (const { zoneKey } of zoneItems) {
          const pct = targetPctPerZone[zoneKey] ?? 0
          const arr = next[zoneKey] ?? []
          const point: ChartPoint = { t: sessionElapsed, value: pct }
          const updated = [...arr, point].slice(-Math.ceil(maxPoints))
          next[zoneKey] = updated
        }
        return next
      })
    }
  }, [sessionElapsed, totalDuration, zoneItems, targetPctPerZone, intervalMinutes])

  useEffect(() => {
    setPointsPerZone({})
    lastSampleRef.current = 0
    displayedValuesRef.current = {}
  }, [intervalMinutes])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const dpr = window.devicePixelRatio || 1
    const w = wrap.clientWidth
    const h = wrap.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
    const bgColor = isDark ? '#0E141F' : '#FFFFFF'
    const gridColor = isDark ? 'rgba(74, 85, 104, 0.5)' : 'rgba(126, 126, 143, 0.4)'
    const tickColor = isDark ? '#A0AEC0' : '#9A9AAF'
    const series = zoneItems.map((z, i) => {
      const pts = pointsPerZone[z.zoneKey] ?? []
      const displayVal = displayedValuesRef.current[z.zoneKey] ?? 0
      return {
        label: z.label,
        color: ACTIVITY_SERIES_COLORS[i % ACTIVITY_SERIES_COLORS.length],
        points: pts.length > 0 ? pts : [{ t: 0, value: 0 }],
        displayValue: displayVal,
      }
    })
    if (series.length === 0) return
    drawExpertActivityChart(ctx, series, totalDuration, currentTime, w, h, bgColor, gridColor, tickColor, compact)

    if (!hover) return
    const drawH = h - (compact ? CHART_PAD_TOP_COMPACT : CHART_PAD_TOP) - (compact ? CHART_PAD_BOTTOM_COMPACT : CHART_PAD_BOTTOM)
    const allValues = series.flatMap((s) => s.points.map((p) => p.value)).concat(series.map((s) => s.displayValue))
    const maxVal = Math.max(1, 100, ...allValues)
    const padBottom = compact ? CHART_PAD_BOTTOM_COMPACT : CHART_PAD_BOTTOM
    const padTop = compact ? CHART_PAD_TOP_COMPACT : CHART_PAD_TOP
    const toY = (val: number) => h - padBottom - (val / maxVal) * drawH
    const padLeft = compact ? CHART_PAD_LEFT_COMPACT : CHART_PAD_LEFT
    const padRight = compact ? CHART_PAD_RIGHT_COMPACT : CHART_PAD_RIGHT
    const lineX = Math.min(Math.max(hover.x, padLeft), w - padRight)
    ctx.save()
    ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.55)' : 'rgba(100, 116, 139, 0.55)'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(lineX, padTop)
    ctx.lineTo(lineX, h - padBottom)
    ctx.stroke()
    ctx.setLineDash([])
    for (const s of series) {
      const valueAtHover = getSeriesValueAtTime(s, hover.time, Math.min(currentTime, totalDuration))
      ctx.beginPath()
      ctx.arc(lineX, toY(valueAtHover), 2.5, 0, Math.PI * 2)
      ctx.fillStyle = s.color
      ctx.fill()
    }
    ctx.restore()
  }, [pointsPerZone, zoneItems, currentTime, totalDuration, hover, compact])

  const drawRef = useRef(draw)
  drawRef.current = draw

  useEffect(() => {
    const ease = 0.03
    const threshold = 0.15
    const animate = () => {
      for (const { zoneKey } of zoneItems) {
        const target = targetPctPerZone[zoneKey] ?? 0
        const current = displayedValuesRef.current[zoneKey] ?? 0
        const diff = target - current
        displayedValuesRef.current[zoneKey] = Math.abs(diff) < threshold ? target : current + diff * ease
      }
      drawRef.current()
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [zoneItems, targetPctPerZone])

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const ro = new ResizeObserver(() => draw())
    if (wrapRef.current) ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [draw])

  const handleChartHover = useCallback((event: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    if (rect.width <= 0) return
    const padLeft = compact ? CHART_PAD_LEFT_COMPACT : CHART_PAD_LEFT
    const padRight = compact ? CHART_PAD_RIGHT_COMPACT : CHART_PAD_RIGHT
    const xRaw = event.clientX - rect.left
    const x = Math.min(Math.max(xRaw, padLeft), rect.width - padRight)
    const drawW = rect.width - padLeft - padRight
    if (drawW <= 0) return
    const endT = Math.min(currentTime, totalDuration)
    const relative = (x - padLeft) / drawW
    const rawTime = Math.min(totalDuration, Math.max(0, relative * totalDuration))
    const step = 5
    const hoverTime = Math.min(endT, Math.round(rawTime / step) * step)
    const series = zoneItems.map((z, i) => ({
      label: z.label,
      color: ACTIVITY_SERIES_COLORS[i % ACTIVITY_SERIES_COLORS.length],
      points: pointsPerZone[z.zoneKey] ?? [{ t: 0, value: 0 }],
      displayValue: displayedValuesRef.current[z.zoneKey] ?? 0,
    }))
    const rows = series.map((s) => ({
      label: s.label,
      color: s.color,
      value: getSeriesValueAtTime(s, hoverTime, endT),
    }))
    const left = Math.min(Math.max(x + 10, 8), Math.max(8, rect.width - 170))
    setHover({ x, left, time: hoverTime, rows })
  }, [zoneItems, pointsPerZone, currentTime, totalDuration, compact])

  return (
    <div ref={wrapRef} className={className ?? styles.presenceChartWrap}>
      <canvas ref={canvasRef} onMouseMove={handleChartHover} onMouseLeave={() => setHover(null)} />
      {hover && (
        <div className={styles.chartHoverTip} style={{ left: `${hover.left}px` }}>
          <div className={styles.chartHoverTime}>{formatTimeLabel(hover.time)}</div>
          {hover.rows.map((r) => (
            <div key={r.label} className={styles.chartHoverRow}>
              <span className={styles.chartHoverDot} style={{ background: r.color }} />
              <span className={styles.chartHoverLabel}>{r.label}</span>
              <span className={styles.chartHoverValue}>{Math.round(r.value)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export interface PresenceChartCardProps {
  benefit: HierarchyBenefit
  zones: Record<string, ZoneData> | null
  sessionElapsed: number
  presenceAtStart?: number | Record<string, number>
  onEditBenefit?: (benefitId: string) => void
  onHideCard?: (benefitId: string) => void
  onSyncZones?: () => Promise<void>
}

/** Carte Rapport d'activité avec courbes — version graphique (bibliothèque, réutilisable plus tard) */
export function PresenceChartCard({
  benefit,
  zones,
  sessionElapsed,
  presenceAtStart = 0,
  onEditBenefit,
  onHideCard,
  onSyncZones,
}: PresenceChartCardProps) {
  const [intervalMinutes, setIntervalMinutes] = useState(2)
  const zoneItems = getDetectionZoneKeys(benefit, zones)
  const polys = benefit.zone_polygons ?? []
  const types = benefit.zone_polygon_types ?? polys.map(() => 'include' as const)
  const includeCount = polys.filter((_, i) => types[i] === 'include').length
  const needsSyncForFormes = includeCount >= 2 && zoneItems.length === 1 && zoneItems[0]?.label === 'Présence' && !!onSyncZones
  const lastAutoSyncRef = useRef<string | null>(null)

  useEffect(() => {
    if (!needsSyncForFormes || !onSyncZones || lastAutoSyncRef.current === benefit.benefit_id) return
    lastAutoSyncRef.current = benefit.benefit_id
    onSyncZones().catch(() => {})
  }, [needsSyncForFormes, onSyncZones, benefit.benefit_id])

  const presenceAtStartMap = typeof presenceAtStart === 'object' ? presenceAtStart : { [benefit.benefit_id]: presenceAtStart }

  return (
    <article className={`${styles.card} ${styles.cardWide}`}>
      <div className={styles.cardMain}>
        <div className={styles.cardTop}>
          <div className={styles.cardTopLeft}>
            <span className={styles.cardTitle}>Rapport d&apos;activité (courbes)</span>
            <span className={styles.chartTimeBadge}>{intervalMinutes} min</span>
          </div>
          <div className={styles.chartControls}>
            <select
              value={intervalMinutes}
              onChange={(e) => setIntervalMinutes(Number(e.target.value))}
              className={styles.chartIntervalSelect}
              aria-label="Intervalle de temps"
            >
              {TIME_INTERVAL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <CardMenu
              options={[
                ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
                ...(onHideCard ? [{ label: 'Supprimer la carte', onClick: () => onHideCard(benefit.benefit_id) }] : []),
              ]}
            />
          </div>
        </div>
        <ActivityChart
          zoneItems={zoneItems}
          zones={zones}
          sessionElapsed={sessionElapsed}
          presenceAtStartMap={presenceAtStartMap}
          intervalMinutes={intervalMinutes}
          compact={false}
          className={styles.presenceChartWrap}
        />
        <div className={styles.chartLegend}>
          {zoneItems.map((z, i) => (
            <div key={z.zoneKey} className={styles.chartLegendItem}>
              <span
                className={styles.chartLegendDot}
                style={{ background: ACTIVITY_SERIES_COLORS[i % ACTIVITY_SERIES_COLORS.length] }}
              />
              <span className={styles.chartLegendLabel}>{z.label}</span>
            </div>
          ))}
        </div>
      </div>
    </article>
  )
}
