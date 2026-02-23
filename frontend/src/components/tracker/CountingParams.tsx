import { useEffect, useState } from 'react'
import { useCountingStore } from '@/stores/countingStore'
import styles from './CountingParams.module.css'

export default function CountingParams() {
  const params = useCountingStore((s) => s.params)
  const loadParams = useCountingStore((s) => s.loadParams)
  const saveParams = useCountingStore((s) => s.saveParams)

  const [threshold, setThreshold] = useState(30)
  const [cooldown, setCooldown] = useState(15)

  useEffect(() => { loadParams() }, [loadParams])

  useEffect(() => {
    if (params) {
      setThreshold(params.simple_gradient_threshold)
      setCooldown(params.simple_cooldown_frames)
    }
  }, [params])

  const handleSave = () => {
    saveParams({ simple_gradient_threshold: threshold, simple_cooldown_frames: cooldown })
  }

  return (
    <div className={styles.panel}>
      <div className={styles.header}>PARAMÈTRES COMPTAGE</div>

      <div className={styles.row}>
        <label className={styles.label}>
          Seuil gradient
          <span className={styles.value}>{threshold}</span>
        </label>
        <input
          type="range"
          min={5}
          max={100}
          step={1}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
          className={styles.range}
        />
      </div>

      <div className={styles.row}>
        <label className={styles.label}>
          Cooldown frames
          <span className={styles.value}>{cooldown}</span>
        </label>
        <input
          type="range"
          min={1}
          max={60}
          step={1}
          value={cooldown}
          onChange={(e) => setCooldown(Number(e.target.value))}
          className={styles.range}
        />
      </div>

      <button className={styles.saveBtn} onClick={handleSave}>
        Enregistrer
      </button>
    </div>
  )
}
