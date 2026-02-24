import { useLocation } from 'react-router-dom'
import styles from './TopBar.module.css'

function getPageLabel(pathname: string): string {
  if (pathname === '/' || pathname.startsWith('/tracker')) return 'Zone Tracker'
  if (pathname.startsWith('/analytics')) return 'Analytics'
  if (pathname.startsWith('/logs')) return 'Log / Historique'
  return 'Zone Tracker'
}

export default function TopBar() {
  const location = useLocation()
  const label = getPageLabel(location.pathname)

  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        <img
          className={styles.logo}
          src="/static/assets_youn/YrysUIPackage/ArcyWhitelogo.svg"
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
