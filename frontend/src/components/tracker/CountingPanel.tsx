import { useEffect, useMemo } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useCountingStore, type CountingConfig } from '@/stores/countingStore'
import styles from './CountingPanel.module.css'

export default function CountingPanel() {
  const currentVideo = useVideoStore((s) => s.currentVideo)
  const zones = useVideoStore((s) => s.zones)
  const configs = useCountingStore((s) => s.configs)
  const refresh = useCountingStore((s) => s.refresh)
  const toggle = useCountingStore((s) => s.toggle)
  const flip = useCountingStore((s) => s.flip)
  const reset = useCountingStore((s) => s.reset)
  const configure = useCountingStore((s) => s.configure)

  useEffect(() => {
    if (currentVideo) refresh(currentVideo)
  }, [currentVideo, refresh])

  useEffect(() => {
    if (!currentVideo) return
    const id = setInterval(() => refresh(currentVideo), 2000)
    return () => clearInterval(id)
  }, [currentVideo, refresh])

  const cfg: CountingConfig | undefined = currentVideo ? configs[currentVideo] : undefined
  const videoZones = useMemo(() => {
    if (!currentVideo) return []
    const zMap = zones[currentVideo]
    return zMap ? Object.keys(zMap) : []
  }, [currentVideo, zones])

  if (!currentVideo) {
    return (
      <div className={styles.panel}>
        <div className={styles.header}>COMPTAGE</div>
        <div className={styles.empty}>Sélectionnez une caméra</div>
      </div>
    )
  }

  const selectedZone = cfg?.zone_name ?? ''
  const mode = cfg?.mode ?? 'simple'
  const enabled = cfg?.enabled ?? false
  const count = cfg?.count ?? 0
  const angle = cfg?.angle

  const handleZoneChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (e.target.value && currentVideo) {
      configure(currentVideo, e.target.value, mode)
    }
  }

  const handleModeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (selectedZone && currentVideo) {
      configure(currentVideo, selectedZone, e.target.value)
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>COMPTAGE</div>

      <div className={styles.row}>
        <label className={styles.label}>Zone</label>
        <select
          className={styles.select}
          value={selectedZone}
          onChange={handleZoneChange}
        >
          <option value="">— choisir —</option>
          {videoZones.map((z) => (
            <option key={z} value={z}>{z}</option>
          ))}
        </select>
      </div>

      <div className={styles.row}>
        <label className={styles.label}>Mode</label>
        <select
          className={styles.select}
          value={mode}
          onChange={handleModeChange}
          disabled={!selectedZone}
        >
          <option value="simple">Simple (gradient)</option>
          <option value="complex">Complex (MOG2)</option>
        </select>
      </div>

      <div className={styles.counter}>
        <span className={styles.countValue}>{count}</span>
        <span className={styles.countLabel}>passages</span>
      </div>

      {angle !== null && angle !== undefined && (
        <div className={styles.angleInfo}>
          Direction : {angle}°
        </div>
      )}

      <div className={styles.actions}>
        <button
          className={`${styles.btn} ${enabled ? styles.btnDanger : styles.btnPrimary}`}
          onClick={() => toggle(currentVideo)}
          disabled={!selectedZone}
        >
          {enabled ? '⏹ Stop' : '▶ Démarrer'}
        </button>
        <button
          className={styles.btn}
          onClick={() => flip(currentVideo)}
          disabled={!selectedZone}
          title="Tourner la direction de 90°"
        >
          ↻ Flip
        </button>
        <button
          className={`${styles.btn} ${styles.btnWarn}`}
          onClick={() => reset(currentVideo)}
          disabled={!selectedZone}
        >
          ↺ Reset
        </button>
      </div>
    </div>
  )
}
