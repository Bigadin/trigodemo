import { useState, useEffect, useRef, useCallback } from 'react'
import { fetchMetrics, type MetricsData } from '@/api/metrics'
import styles from './PerfChart.module.css'

const DEFAULT_HIDDEN = new Set(['YOLO Inference', 'FPS', 'Active Detections'])

function hexToRgba(hex: string, a: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${a})`
}

function drawStepChart(
  ctx: CanvasRenderingContext2D,
  data: MetricsData,
  hidden: Set<string>,
  w: number,
  h: number
) {
  ctx.fillStyle = '#0A0E13'
  ctx.fillRect(0, 0, w, h)

  const gridLines = 4
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 1
  for (let i = 1; i < gridLines; i++) {
    const y = Math.round((h / gridLines) * i) + 0.5
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
    ctx.stroke()
  }

  const firstSeries = Object.values(data)[0]
  const sampleLen = firstSeries?.data?.length ?? 200
  const gridStep = Math.max(1, Math.floor(sampleLen / 8))
  ctx.strokeStyle = 'rgba(255,255,255,0.03)'
  for (let i = gridStep; i < sampleLen; i += gridStep) {
    const x = Math.round((i / (sampleLen - 1)) * w) + 0.5
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, h)
    ctx.stroke()
  }

  const pad = 4
  const drawH = h - pad * 2

  for (const [name, series] of Object.entries(data)) {
    if (hidden.has(name)) continue
    const seriesData = series.data
    if (!seriesData || seriesData.length < 2) continue

    const sMin = series.min ?? 0
    const sMax = series.max ?? 100
    const range = sMax - sMin || 1;

    const toY = (v: number) => h - pad - ((v - sMin) / range) * drawH

    ctx.beginPath()
    let prevY = toY(seriesData[0].v)
    ctx.moveTo(0, prevY)

    for (let i = 1; i < seriesData.length; i++) {
      const x = (i / (seriesData.length - 1)) * w
      const y = toY(seriesData[i].v)
      ctx.lineTo(x, prevY)
      ctx.lineTo(x, y)
      prevY = y
    }
    ctx.lineTo(w, prevY)

    ctx.strokeStyle = series.color
    ctx.lineWidth = 1.5
    ctx.lineJoin = 'miter'
    ctx.stroke()

    ctx.lineTo(w, h)
    ctx.lineTo(0, h)
    ctx.closePath()
    ctx.fillStyle = hexToRgba(series.color, 0.05)
    ctx.fill()
  }
}

export default function PerfChart() {
  const [data, setData] = useState<MetricsData | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(DEFAULT_HIDDEN)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !data) return

    const dpr = window.devicePixelRatio || 1
    const w = wrap.clientWidth
    const h = wrap.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr);

    drawStepChart(ctx, data, hidden, w, h)
  }, [data, hidden])

  useEffect(() => {
    let cancelled = false
    fetchMetrics(120)
      .then((res) => { if (!cancelled) setData(res) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    draw()
  }, [draw])

  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [draw])

  const toggleSeries = (name: string) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  if (!data) return null

  return (
    <div className={styles.perfCard}>
      <div className={styles.perfHeader}>
        <div className={styles.perfLegends}>
          {Object.entries(data).map(([name, series]) => {
            const vals = series.data.map((d) => d.v)
            const min = Math.round(Math.min(...vals))
            const max = Math.round(Math.max(...vals))
            const unit = series.unit || ''
            const isOff = hidden.has(name)
            return (
              <button
                key={name}
                type="button"
                className={`${styles.perfLegend} ${isOff ? styles.isOff : ''}`}
                style={{ color: series.color }}
                onClick={() => toggleSeries(name)}
              >
                <span className={styles.perfLegendCheck} />
                <span className={styles.perfLegendLabel}>
                  {name} [{min}{unit ? ` ${unit}` : ''} – {max}{unit ? ` ${unit}` : ''}]
                </span>
              </button>
            )
          })}
        </div>
      </div>
      <div ref={wrapRef} className={styles.perfCanvasWrap}>
        <canvas ref={canvasRef} />
      </div>
    </div>
  )
}
