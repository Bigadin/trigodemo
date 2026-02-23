import { useSiteStore } from '@/stores/siteStore'
import { useVideoStore, type PresenceData } from '@/stores/videoStore'
import type { HierarchyBenefit } from '@/types/site'
import styles from './DataRoom.module.css'

export default function DataRoom() {
  const selectedSiteId = useSiteStore((s) => s.selectedSiteId)
  const getSite = useSiteStore((s) => s.getSite)
  const presence = useVideoStore((s) => s.presence)
  const site = selectedSiteId ? getSite(selectedSiteId) : undefined

  if (!site) return null

  const benefits: Array<{ benefit: HierarchyBenefit; camPath: string }> = []
  for (const cam of site.cameras) {
    for (const ben of cam.benefits) {
      benefits.push({ benefit: ben, camPath: cam.path })
    }
  }

  if (benefits.length === 0) {
    return (
      <div className={styles.panel}>
        <div className={styles.title}>DATA ROOM</div>
        <div className={styles.empty}>Aucun bénéfice actif</div>
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.title}>DATA ROOM</div>
      <div className={styles.grid}>
        {benefits.map(({ benefit, camPath }) => (
          <RoiCard
            key={benefit.id}
            benefit={benefit}
            presence={presence[camPath]}
          />
        ))}
      </div>
    </div>
  )
}

function RoiCard({
  benefit,
  presence: pres,
}: {
  benefit: HierarchyBenefit
  presence?: Record<string, PresenceData>
}) {
  const zoneName = benefit.name
  const zoneData = pres?.[zoneName]
  const occupied = zoneData?.occupied ?? false
  const totalTime = zoneData?.total_time ?? 0

  const formatTime = (sec: number) => {
    if (sec < 60) return `${Math.round(sec)}s`
    if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
    return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
  }

  const skillLabel: Record<string, string> = {
    detection: 'Détection présence',
    counting: 'Comptage',
    heatmap: 'Heatmap',
    quality: 'Qualité',
  }

  return (
    <div className={`${styles.card} ${occupied ? styles.cardOccupied : ''}`}>
      <div className={styles.cardHeader}>
        <span className={styles.cardName}>{benefit.name}</span>
        <span className={`${styles.statusDot} ${occupied ? styles.statusActive : ''}`} />
      </div>
      <div className={styles.cardSkill}>
        {skillLabel[benefit.skill] ?? benefit.skill}
      </div>
      <div className={styles.cardMetrics}>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Statut</span>
          <span className={`${styles.metricValue} ${occupied ? styles.metricGreen : ''}`}>
            {occupied ? 'Occupé' : 'Libre'}
          </span>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Temps présence</span>
          <span className={styles.metricValue}>{formatTime(totalTime)}</span>
        </div>
      </div>
      <div className={styles.cardCategories}>
        {benefit.categories.map((cat) => (
          <span key={cat} className={styles.catChip}>
            {cat.split('::').pop()}
          </span>
        ))}
      </div>
    </div>
  )
}
