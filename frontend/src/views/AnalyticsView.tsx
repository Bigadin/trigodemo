import { useEffect, useState, useCallback } from 'react'
import { fetchMetrics, type MetricsPoint } from '@/api/metrics'
import { useVideoStore } from '@/stores/videoStore'
import { useSiteStore } from '@/stores/siteStore'
import styles from './AnalyticsView.module.css'

export default function AnalyticsView() {
  const [metrics, setMetrics] = useState<MetricsPoint[]>([])
  const [loading, setLoading] = useState(true)
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const lieux = useSiteStore((s) => s.lieux)

  const totalCams = lieux.reduce((n, l) => n + l.sites.reduce((m, s) => m + s.cameras.length, 0), 0)
  const totalBens = lieux.reduce(
    (n, l) => n + l.sites.reduce((m, s) => m + s.cameras.reduce((k, c) => k + c.benefits.length, 0), 0),
    0,
  )

  const loadMetrics = useCallback(async () => {
    try {
      const res = await fetchMetrics(120)
      setMetrics(res.metrics)
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => {
    loadMetrics()
    const id = setInterval(loadMetrics, 5000)
    return () => clearInterval(id)
  }, [loadMetrics])

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.title}>Analytics</h1>
        <p className={styles.subtitle}>Tableau de bord temps réel</p>
      </div>

      {/* KPI Cards */}
      <div className={styles.kpis}>
        <KpiCard label="Streams actifs" value={activeStreams.size} accent />
        <KpiCard label="Total caméras" value={totalCams} />
        <KpiCard label="Total bénéfices" value={totalBens} />
        <KpiCard label="FPS" value={latest?.fps?.toFixed(1) ?? '—'} />
        <KpiCard label="Inférence" value={latest ? `${latest.inference_ms.toFixed(0)} ms` : '—'} />
        <KpiCard label="CPU" value={latest ? `${latest.cpu_percent.toFixed(0)}%` : '—'} />
        <KpiCard label="Mémoire" value={latest ? `${latest.memory_mb.toFixed(0)} MB` : '—'} />
      </div>

      {/* Mini Chart */}
      {loading ? (
        <div className={styles.loading}>Chargement des métriques…</div>
      ) : metrics.length === 0 ? (
        <div className={styles.empty}>Aucune métrique disponible</div>
      ) : (
        <div className={styles.chartSection}>
          <div className={styles.chartTitle}>Performance (dernières 2 min)</div>
          <div className={styles.charts}>
            <MiniChart data={metrics} field="fps" label="FPS" color="#3b82f6" />
            <MiniChart data={metrics} field="inference_ms" label="Inférence (ms)" color="#f59e0b" />
            <MiniChart data={metrics} field="cpu_percent" label="CPU (%)" color="#ef4444" />
            <MiniChart data={metrics} field="memory_mb" label="Mémoire (MB)" color="#10b981" />
          </div>
        </div>
      )}
    </div>
  )
}

function KpiCard({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className={styles.kpiCard}>
      <span className={`${styles.kpiValue} ${accent ? styles.kpiAccent : ''}`}>{value}</span>
      <span className={styles.kpiLabel}>{label}</span>
    </div>
  )
}

function MiniChart({
  data,
  field,
  label,
  color,
}: {
  data: MetricsPoint[]
  field: keyof MetricsPoint
  label: string
  color: string
}) {
  const values = data.map((d) => Number(d[field]))
  const max = Math.max(...values, 1)
  const min = Math.min(...values, 0)
  const range = max - min || 1
  const h = 60
  const w = 200

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / range) * h
      return `${x},${y}`
    })
    .join(' ')

  const latest = values[values.length - 1]

  return (
    <div className={styles.miniChart}>
      <div className={styles.chartHeader}>
        <span className={styles.chartLabel}>{label}</span>
        <span className={styles.chartCurrent} style={{ color }}>
          {latest?.toFixed(1)}
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className={styles.svg}>
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  )
}
