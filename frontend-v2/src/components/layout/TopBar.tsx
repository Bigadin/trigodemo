import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { img } from '@/utils/theme'
import styles from './TopBar.module.css'

function getPageLabel(pathname: string): string {
  if (pathname === '/') return 'Sites'
  if (pathname.startsWith('/tracker')) return 'Zone Tracker'
  if (pathname.startsWith('/analytics')) return 'Analytics'
  if (pathname.startsWith('/logs')) return 'Log / Historique'
  return 'Sites'
}

function formatDateTime(): string {
  const d = new Date()
  const time = d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })
  const dayName = d.toLocaleDateString('fr-FR', { weekday: 'long' }).slice(0, 3)
  const dayCap = dayName.charAt(0).toUpperCase() + dayName.slice(1)
  const date = d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' })
  return `${time} - ${dayCap} ${date}`
}

export default function TopBar() {
  const location = useLocation()
  const baseLabel = getPageLabel(location.pathname)
  const [dateTime, setDateTime] = useState(formatDateTime)

  useEffect(() => {
    if (!location.pathname.startsWith('/tracker')) return
    const id = setInterval(() => setDateTime(formatDateTime), 1000)
    return () => clearInterval(id)
  }, [location.pathname])

  const label = location.pathname.startsWith('/tracker') ? dateTime : baseLabel

  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        <img
          className={styles.logo}
          src={img('arcy-logo-white')}
          alt="Arcy"
        />
      </div>
      <div className={styles.right}>
        <span className={styles.label}>{label}</span>
        <button className={styles.menuBtn} title="Options" type="button">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="5" cy="12" r="2" />
            <circle cx="12" cy="12" r="2" />
            <circle cx="19" cy="12" r="2" />
          </svg>
        </button>
      </div>
    </header>
  )
}
