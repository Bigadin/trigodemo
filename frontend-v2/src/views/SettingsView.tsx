import styles from './PlaceholderView.module.css'

export default function SettingsView() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Paramètres</h1>
          <p className={styles.subtitle}>Configuration de l'application</p>
        </div>
      </header>
      <section className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.placeholderIcon}>⚙️</span>
          <p>Section Paramètres — à venir</p>
        </div>
      </section>
    </div>
  )
}
