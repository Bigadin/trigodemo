import styles from './PlaceholderView.module.css'

export default function LogsView() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Log / Historique</h1>
          <p className={styles.subtitle}>Journal d'événements et audit</p>
        </div>
      </header>
      <section className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.placeholderIcon}>📋</span>
          <p>Section Logs — à venir</p>
        </div>
      </section>
    </div>
  )
}
