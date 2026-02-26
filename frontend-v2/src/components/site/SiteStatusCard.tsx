import { useState, useEffect } from 'react'
import { fetchMetrics } from '@/api/metrics'
import styles from './SiteStatusCard.module.css'

const SPARKLINE_POINTS = 24
const LATENCY_COLOR = '#f59e0b'

type Props = {
  activeBenefitsCount: number
  streamingCount: number
}

export function SiteStatusCard({ activeBenefitsCount, streamingCount }: Props) {
  const [latencyData, setLatencyData] = useState<number[]>([])

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const m = await fetchMetrics(SPARKLINE_POINTS)
        const series = m['Latency Cam']
        if (series?.data?.length) {
          const vals = series.data.slice(-SPARKLINE_POINTS).map((d) => d.v)
          if (!cancelled) setLatencyData(vals)
        }
      } catch {
        if (!cancelled) setLatencyData([])
      }
    }
    load()
    const t = setInterval(load, 4000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [])

  const lastLatency = latencyData.length > 0 ? Math.round(latencyData[latencyData.length - 1]) : null
  const maxVal = Math.max(...latencyData, 1)
  const pathD =
    latencyData.length > 1
      ? latencyData
          .map((v, i) => {
            const x = (i / (latencyData.length - 1)) * 100
            const y = 100 - (v / maxVal) * 90
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
          })
          .join(' ')
      : ''

  const isRunning = activeBenefitsCount > 0 || streamingCount > 0

  return (
    <div className={styles.wrap}>
      <div className={styles.sparklineWrap}>
        {pathD ? (
          <svg className={styles.sparkline} viewBox="0 0 100 100" preserveAspectRatio="none">
            <path d={pathD} fill="none" stroke={LATENCY_COLOR} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <div className={styles.sparklinePlaceholder}>—</div>
        )}
        {lastLatency != null && (
          <span className={styles.latencyLabel} title="Latence caméra">
            {lastLatency} ms
          </span>
        )}
      </div>
      <div className={styles.statusRow}>
        <span className={`${styles.statusDot} ${isRunning ? styles.statusDotOn : ''}`} />
        <span className={styles.statusText}>
          {isRunning
            ? [
                activeBenefitsCount > 0 && `${activeBenefitsCount} bénéfice${activeBenefitsCount > 1 ? 's' : ''} actif${activeBenefitsCount > 1 ? 's' : ''}`,
                streamingCount > 0 && `${streamingCount} flux`,
              ]
                .filter(Boolean)
                .join(' · ')
            : 'En veille'}
        </span>
      </div>
    </div>
  )
}
