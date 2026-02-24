import styles from './PlaceholderView.module.css'

export default function SitesView() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Sites</h1>
          <p className={styles.subtitle}>Liste des lieux et sites</p>
        </div>
      </header>
      <section className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.placeholderIcon}>▦</span>
          <p>Section Sites — à venir</p>
        </div>
      </section>
    </div>
  )
}
