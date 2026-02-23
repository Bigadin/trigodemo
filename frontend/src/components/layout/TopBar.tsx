import { NavLink, useLocation } from 'react-router-dom'
import { useSiteStore } from '@/stores/siteStore'
import styles from './TopBar.module.css'

const NAV_GROUPS = [
  {
    label: null,
    items: [{ to: '/', label: 'Sites', end: true }],
  },
  {
    label: 'MONITOR',
    items: [
      { to: '/tracker', label: 'Tracker' },
      { to: '/analytics', label: 'Analytics' },
      { to: '/logs', label: 'Logs' },
    ],
  },
] as const

export default function TopBar() {
  const location = useLocation()
  const lieux = useSiteStore((s) => s.lieux)
  const totalSites = lieux.reduce((n, l) => n + l.sites.length, 0)
  const totalCams = lieux.reduce(
    (n, l) => n + l.sites.reduce((m, s) => m + s.cameras.length, 0),
    0,
  )

  return (
    <header className={styles.topbar}>
      <div className={styles.brand}>
        <span className={styles.logo}>▲</span>
        <span className={styles.title}>TRIGO</span>
      </div>

      <div className={styles.divider} />

      <nav className={styles.nav}>
        {NAV_GROUPS.map((group, gi) => (
          <div key={gi} className={styles.navGroup}>
            {group.label && (
              <span className={styles.groupLabel}>{group.label}</span>
            )}
            {group.items.map(({ to, label, ...rest }) => (
              <NavLink
                key={to}
                to={to}
                end={'end' in rest && rest.end}
                className={({ isActive }) =>
                  `${styles.navItem} ${isActive ? styles.active : ''}`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className={styles.spacer} />

      <div className={styles.stats}>
        <span className={styles.stat}>{lieux.length} lieux</span>
        <span className={styles.statDot}>·</span>
        <span className={styles.stat}>{totalSites} sites</span>
        <span className={styles.statDot}>·</span>
        <span className={styles.stat}>{totalCams} cam</span>
      </div>

      <div className={styles.pathBadge}>
        {location.pathname === '/'
          ? 'Accueil'
          : location.pathname.replace(/^\//, '')}
      </div>
    </header>
  )
}
