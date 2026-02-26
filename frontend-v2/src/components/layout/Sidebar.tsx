import { NavLink } from 'react-router-dom'
import ExplorerTree from '@/components/sidebar/ExplorerTree'
import GpuUsageBar from '@/components/sidebar/GpuUsageBar'
import { icon, img } from '@/utils/theme'
import styles from './Sidebar.module.css'

const NAV_ITEMS = [
  { to: '/', end: true, label: 'Vue d\'ensemble', icon: 'folder' },
  { to: '/tracker', end: false, label: 'Zone Tracker', icon: 'desktop' },
  { to: '/analytics', end: true, label: 'Analytics', icon: 'chart' },
  { to: '/logs', end: true, label: 'Log / Historique', icon: 'terminal' },
]

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <img
          className={styles.logoImg}
          src={img('arcy-logo')}
          alt="Arcy"
        />
        <span className={styles.logoText}>YRYS</span>
      </div>

      <div className={styles.sectionNav}>
        <div className={styles.sectionLabel}>Navigation</div>
        <nav className={styles.nav}>
          {NAV_ITEMS.map(({ to, end, label, icon: iconName }) => (
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
                src={icon(iconName)}
                alt=""
              />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className={styles.sectionGpu}>
        <GpuUsageBar />
      </div>

      <div className={styles.sectionTree}>
        <div className={styles.sectionLabel}>Explorateur</div>
        <div className={styles.treeContent}>
          <ExplorerTree />
        </div>
      </div>

      <div className={styles.footer}>
        <div className={styles.version}>&copy;2025 Arcy &mdash; v0.1.0-alpha</div>
      </div>
    </aside>
  )
}
