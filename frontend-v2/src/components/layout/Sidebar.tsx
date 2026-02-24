import { NavLink } from 'react-router-dom'
import styles from './Sidebar.module.css'

const NAV_ITEMS = [
  { to: '/', end: true, label: 'Sites', icon: 'folder-svgrepo-com.svg' },
  { to: '/tracker', end: false, label: 'Zone Tracker', icon: 'desktop-svgrepo-com.svg' },
  { to: '/analytics', end: true, label: 'Analytics', icon: 'chart-line-svgrepo-com.svg' },
  { to: '/logs', end: true, label: 'Log / Historique', icon: 'terminal-svgrepo-com.svg' },
]

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <img
          className={styles.logoImg}
          src="/static/assets_youn/Arcy icon.png"
          alt="Arcy"
        />
        <span className={styles.logoText}>YRYS</span>
      </div>

      <div className={styles.sectionNav}>
        <div className={styles.sectionLabel}>Navigation</div>
        <nav className={styles.nav}>
          {NAV_ITEMS.map(({ to, end, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `${styles.navItem} ${isActive ? styles.active : ''}`
              }
            >
              <img
                className={styles.navIcon}
                src={`/static/assets_youn/SvIcons/${icon}`}
                alt=""
              />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className={styles.sectionTree}>
        <div className={styles.sectionLabel}>Sites</div>
        <div className={styles.treeContent}>
          <span className={styles.empty}>Chargement…</span>
        </div>
      </div>

      <div className={styles.footer}>
        <div className={styles.version}>&copy;2025 Arcy &mdash; v0.1.0-alpha</div>
      </div>
    </aside>
  )
}
