import styles from './PlaceholderView.module.css'

export default function TrackerView() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Zone Tracker</h1>
          <p className={styles.subtitle}>Surveillance et analyse du temps de présence</p>
        </div>
      </header>
      <section className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.placeholderIcon}>▶</span>
          <p>Section Zone Tracker — à venir</p>
        </div>
      </section>
    </div>
  )
}
