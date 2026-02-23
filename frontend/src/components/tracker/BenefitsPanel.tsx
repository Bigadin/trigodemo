import { useSiteStore } from '@/stores/siteStore'
import { useVideoStore } from '@/stores/videoStore'
import type { HierarchyBenefit } from '@/types/site'
import styles from './BenefitsPanel.module.css'

const SKILL_ICON: Record<string, string> = {
  detection: '🔍',
  counting: '🔢',
  heatmap: '🌡️',
  quality: '✅',
}

export default function BenefitsPanel() {
  const selectedCameraId = useSiteStore((s) => s.selectedCameraId)
  const getCamera = useSiteStore((s) => s.getCamera)
  const camera = selectedCameraId ? getCamera(selectedCameraId) : undefined

  if (!camera) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>Bénéfices</div>
        <div className={styles.empty}>Sélectionnez une caméra</div>
      </div>
    )
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        Bénéfices
        <span className={styles.count}>{camera.benefits.length}</span>
      </div>
      {camera.benefits.length === 0 ? (
        <div className={styles.empty}>Aucun bénéfice configuré</div>
      ) : (
        <div className={styles.list}>
          {camera.benefits.map((ben) => (
            <BenefitRow key={ben.id} benefit={ben} />
          ))}
        </div>
      )}
    </div>
  )
}

function BenefitRow({ benefit }: { benefit: HierarchyBenefit }) {
  const currentVideo = useVideoStore((s) => s.currentVideo)
  const presence = useVideoStore((s) =>
    currentVideo ? s.presence[currentVideo] : undefined,
  )

  const zoneKeys = benefit.zone_polygons.map((_, i) => `${benefit.name}` + (i > 0 ? `_${i}` : ''))
  const isOccupied = zoneKeys.some((k) => presence?.[k]?.occupied)

  return (
    <div className={`${styles.row} ${isOccupied ? styles.rowOccupied : ''}`}>
      <span className={styles.icon}>{SKILL_ICON[benefit.skill] ?? '◆'}</span>
      <div className={styles.rowInfo}>
        <span className={styles.rowName}>{benefit.name}</span>
        <span className={styles.rowSkill}>{benefit.skill_item}</span>
      </div>
      <div className={styles.rowBadges}>
        {benefit.categories.map((cat) => (
          <span key={cat} className={styles.catBadge}>
            {cat.split('::').pop()}
          </span>
        ))}
      </div>
      <span className={`${styles.dot} ${benefit.active ? styles.dotActive : ''}`} />
    </div>
  )
}
