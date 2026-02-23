import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchLogs, clearLogs } from '@/api/logs'
import type { LogEntry } from '@/types/api'
import styles from './LogsView.module.css'

const CATEGORIES = ['', 'detection', 'counting', 'zone', 'system', 'config']

export default function LogsView() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [category, setCategory] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try {
      const res = await fetchLogs(500, category || undefined)
      setLogs(res.logs)
      setTotal(res.total)
    } catch { /* ignore */ }
    setLoading(false)
  }, [category])

  useEffect(() => {
    load()
    const id = setInterval(load, 3000)
    return () => clearInterval(id)
  }, [load])

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleClear = async () => {
    await clearLogs()
    setLogs([])
    setTotal(0)
  }

  const levelColor: Record<string, string> = {
    info: styles.levelInfo,
    warning: styles.levelWarn,
    error: styles.levelError,
    debug: styles.levelDebug,
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Logs</h1>
          <p className={styles.subtitle}>{total} entrée{total > 1 ? 's' : ''}</p>
        </div>
        <div className={styles.controls}>
          <select
            className={styles.select}
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            <option value="">Toutes catégories</option>
            {CATEGORIES.filter(Boolean).map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <label className={styles.checkLabel}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>
          <button className={styles.clearBtn} onClick={handleClear}>
            Effacer
          </button>
        </div>
      </div>

      {loading ? (
        <div className={styles.loading}>Chargement…</div>
      ) : logs.length === 0 ? (
        <div className={styles.empty}>Aucun log</div>
      ) : (
        <div className={styles.console} ref={containerRef}>
          {logs.map((log, i) => (
            <div key={i} className={styles.line}>
              <span className={styles.ts}>
                {new Date(log.timestamp).toLocaleTimeString('fr-FR')}
              </span>
              <span className={`${styles.level} ${levelColor[log.level] ?? ''}`}>
                {log.level.toUpperCase()}
              </span>
              <span className={styles.cat}>[{log.category}]</span>
              <span className={styles.msg}>{log.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
