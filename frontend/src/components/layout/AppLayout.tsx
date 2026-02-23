import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { useSiteStore } from '@/stores/siteStore'
import TopBar from './TopBar'
import Sidebar from './Sidebar'
import styles from './AppLayout.module.css'

export default function AppLayout() {
  const loadHierarchy = useSiteStore((s) => s.loadHierarchy)

  useEffect(() => {
    loadHierarchy()
  }, [loadHierarchy])

  return (
    <div className={styles.app}>
      <TopBar />
      <div className={styles.body}>
        <Sidebar />
        <main className={styles.main}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
