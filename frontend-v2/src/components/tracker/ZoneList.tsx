import type { ZonesResponse } from '@/api/tracker'
import styles from './ZoneList.module.css'

interface ZoneListProps {
  zones: ZonesResponse['zones'] | null
  loading?: boolean
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return [h, m, s].map((n) => n.toString().padStart(2, '0')).join(':')
}

export default function ZoneList({ zones, loading }: ZoneListProps) {
  if (loading) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.loading}>Chargement…</div>
      </div>
    )
  }

  const zoneNames = zones ? Object.keys(zones) : []

  if (zoneNames.length === 0) {
    return (
      <div className={styles.wrapper}>
        <div className={styles.empty}>Aucune zone configurée</div>
      </div>
    )
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>Zones</div>
      <ul className={styles.list}>
        {zoneNames.map((name) => {
          const zone = zones![name]
          const isOccupied = zone?.is_occupied ?? false
          const totalTime = zone?.total_time ?? 0

          return (
            <li key={name} className={styles.item}>
              <span
                className={`${styles.dot} ${isOccupied ? styles.occupied : ''}`}
                title={isOccupied ? 'Présence détectée' : 'Vide'}
              />
              <span className={styles.name}>{name}</span>
              <span className={styles.time}>{formatTime(totalTime)}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
