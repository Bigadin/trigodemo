import styles from './PlaceholderView.module.css'

export default function AnalyticsView() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Analytics</h1>
          <p className={styles.subtitle}>Métriques et tableaux de bord</p>
        </div>
      </header>
      <section className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.placeholderIcon}>📊</span>
          <p>Section Analytics — à venir</p>
        </div>
      </section>
    </div>
  )
}
