import ExplorerTree from '@/components/sidebar/ExplorerTree'
import styles from './Sidebar.module.css'

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <span className={styles.label}>Explorateur</span>
      </div>
      <div className={styles.content}>
        <ExplorerTree />
      </div>
    </aside>
  )
}
