import { useMemo } from 'react'
import { useSession } from '@/context/SessionContext'
import { useHierarchy } from '@/context/HierarchyContext'
import styles from './GpuUsageBar.module.css'

const BARS_COUNT = 24
const GPU_PER_BENEFIT = 10

function computeGpuLoad(activeBenefitsCount: number): number {
  return Math.min(100, activeBenefitsCount * GPU_PER_BENEFIT)
}

function getBarLevels(loadPct: number): number[] {
  const segment = 100 / BARS_COUNT
  return Array.from({ length: BARS_COUNT }, (_, i) => {
    const segmentStart = i * segment
    const segmentEnd = (i + 1) * segment
    if (loadPct <= segmentStart) return 0
    if (loadPct >= segmentEnd) return 1
    return (loadPct - segmentStart) / segment
  })
}

const BAR_COLORS = [
  '#3b82f6', '#4f46e5', '#6366f1', '#7c3aed', '#8b5cf6', '#a855f7',
  '#c026d3', '#d946ef', '#e879f9', '#ec4899', '#ef4444', '#f43f5e',
]

export default function GpuUsageBar() {
  const { benefitElapsed } = useSession()
  const { hierarchy } = useHierarchy()

  const activeBenefitKeys = useMemo(() => {
    const keys = new Set<string>()
    for (const [lieuId, lieu] of Object.entries(hierarchy)) {
      void lieuId
      for (const [siteId, site] of Object.entries(lieu.sites || {})) {
        for (const [camId, cam] of Object.entries(site.cameras || {})) {
          for (const [benefitId, ben] of Object.entries(cam.benefits || {})) {
            if (ben.active !== false) keys.add(`${siteId}:${camId}:${benefitId}`)
          }
        }
      }
    }
    return keys
  }, [hierarchy])

  const { loadPct, activeCount, barLevels } = useMemo(() => {
    const active = Object.keys(benefitElapsed).filter((k) => (benefitElapsed[k] ?? 0) > 0 && activeBenefitKeys.has(k))
    const load = computeGpuLoad(active.length)
    return { loadPct: load, activeCount: active.length, barLevels: getBarLevels(load) }
  }, [benefitElapsed, activeBenefitKeys])

  return (
    <div className={styles.wrap}>
      <h4 className={styles.title}>Utilisation GPU</h4>
      <div
        className={styles.bars}
        role="progressbar"
        aria-valuenow={loadPct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Utilisation GPU : ${Math.round(loadPct)}%`}
      >
        {barLevels.map((level, i) => (
          <div key={i} className={styles.barSlot}>
            <div
              className={styles.barFill}
              style={{
                height: `${level * 100}%`,
                backgroundColor: level > 0 ? BAR_COLORS[i % BAR_COLORS.length] : undefined,
              }}
            />
          </div>
        ))}
      </div>
      <p className={styles.label}>
        {activeCount > 0 ? (
          <>
            <span className={styles.mainText}>{Math.round(loadPct)}%</span>
            <span className={styles.sep}> — </span>
            <span>{activeCount} bénéfice{activeCount > 1 ? 's' : ''} actif{activeCount > 1 ? 's' : ''}</span>
            <span className={styles.licence}> · Licence exploit</span>
          </>
        ) : (
          'Aucun flux actif'
        )}
      </p>
    </div>
  )
}
