import { Outlet } from 'react-router-dom'
import TopBar from './TopBar'
import Sidebar from './Sidebar'
import styles from './AppLayout.module.css'

export default function AppLayout() {
  return (
    <div className={styles.appLayout}>
      <TopBar />
      <Sidebar />
      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  )
}
