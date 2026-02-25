import { Outlet } from 'react-router-dom'
import { useHierarchy } from '@/context/HierarchyContext'
import TopBar from './TopBar'
import Sidebar from './Sidebar'
import styles from './AppLayout.module.css'

export default function AppLayout() {
  const { error } = useHierarchy()

  return (
    <div className={styles.appLayout}>
      <TopBar />
      <Sidebar />
      <main className={styles.main}>
        {error && (
          <div style={{ padding: 16, background: '#fef2f2', color: '#b91c1c', borderRadius: 8, marginBottom: 16 }}>
            Erreur API : {error}. Vérifiez que le backend tourne sur localhost:8000.
          </div>
        )}
        <Outlet />
      </main>
    </div>
  )
}
