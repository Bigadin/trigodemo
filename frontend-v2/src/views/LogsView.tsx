import { useState, useEffect, useCallback } from 'react'
import { fetchLogs, clearLogs, type AuditLogEntry } from '@/api/logs'
import PerfChart from '@/components/logs/PerfChart'
import styles from './LogsView.module.css'

const LOG_FILTERS: { id: string; label: string }[] = [
  { id: '', label: 'Tout' },
  { id: 'zone', label: 'Zones' },
  { id: 'stream', label: 'Streams' },
  { id: 'camera', label: 'Caméras' },
  { id: 'video', label: 'Vidéos' },
  { id: 'system', label: 'Système' },
  { id: 'blur', label: 'Blur' },
  { id: 'detection', label: 'Détection' },
]

function formatLogTimestamp(isoStr: string): string {
  try {
    const d = new Date(isoStr)
    const pad = (n: number) => n.toString().padStart(2, '0')
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  } catch {
    return isoStr
  }
}

export default function LogsView() {
  const [logs, setLogs] = useState<AuditLogEntry[]>([])
  const [total, setTotal] = useState(0)
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const loadLogs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchLogs(500, filter)
      setLogs(res.logs || [])
      setTotal(res.total ?? res.logs?.length ?? 0)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erreur de chargement')
      setLogs([])
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    loadLogs()
  }, [loadLogs])

  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(loadLogs, 3000)
    return () => clearInterval(interval)
  }, [autoRefresh, loadLogs])

  const handleClear = async () => {
    if (!confirm("Effacer tout le journal d'audit ?")) return
    try {
      await clearLogs()
      loadLogs()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erreur lors de l\'effacement')
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Log / Historique</h1>
          <p className={styles.subtitle}>Journal d'événements et audit</p>
        </div>
      </header>

      <section className={styles.section}>
        <div className={styles.logCard}>
          <div className={styles.cardHeader}>
            <div className={styles.cardHeaderLeft}>
              <div className={styles.cardSubtitle}>Audit</div>
              <div className={styles.cardTitle}>Journal d'événements</div>
            </div>
            <div className={styles.cardHeaderRight}>
              <div className={styles.logFilters}>
                {LOG_FILTERS.map((f) => (
                  <button
                    key={f.id}
                    type="button"
                    className={`${styles.logFilterBtn} ${filter === f.id ? styles.active : ''}`}
                    onClick={() => setFilter(f.id)}
                    data-cat={f.id}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                className={styles.iconBtn}
                onClick={handleClear}
                title="Effacer les logs"
              >
                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
              <button
                type="button"
                className={styles.iconBtn}
                onClick={() => loadLogs()}
                title="Rafraîchir"
              >
                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>
          </div>

          <div className={styles.logConsole}>
            <div className={styles.logConsoleInner}>
              {error && (
                <div className={styles.logEmpty}>{error}</div>
              )}
              {!error && loading && logs.length === 0 && (
                <div className={styles.logEmpty}>Chargement du journal…</div>
              )}
              {!error && !loading && logs.length === 0 && (
                <div className={styles.logEmpty}>Aucun événement enregistré</div>
              )}
              {!error && logs.length > 0 && logs.map((entry, idx) => (
                <div key={`${entry.ts}-${idx}`} className={styles.logEntry}>
                  <span className={styles.logTs}>{formatLogTimestamp(entry.ts)}</span>
                  <span className={`${styles.logLevel} ${styles[`logLevel${(entry.level || 'info').charAt(0).toUpperCase() + (entry.level || 'info').slice(1)}`] ?? styles.logLevelInfo}`} />
                  <span className={`${styles.logCat} ${styles[`logCat${(entry.category || 'system').charAt(0).toUpperCase() + (entry.category || 'system').slice(1)}`] ?? styles.logCatSystem}`}>
                    {entry.category || 'system'}
                  </span>
                  <span className={styles.logAction}>{entry.action || ''}</span>
                  <span className={styles.logDetail}>{entry.detail || ''}</span>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.logStatusbar}>
            <span>{total} événement(s)</span>
            <span>
              Auto-refresh: {autoRefresh ? (
                <button type="button" className={styles.iconBtn} onClick={() => setAutoRefresh(false)} style={{ padding: '2px 6px', fontSize: '12px' }}>
                  ON
                </button>
              ) : (
                <button type="button" className={styles.iconBtn} onClick={() => setAutoRefresh(true)} style={{ padding: '2px 6px', fontSize: '12px' }}>
                  OFF
                </button>
              )}
            </span>
          </div>
        </div>

        <PerfChart />
      </section>
    </div>
  )
}
