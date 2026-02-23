import { useNavigate } from 'react-router-dom'
import { useSiteStore } from '@/stores/siteStore'
import { useUiStore } from '@/stores/uiStore'
import type { HierarchyLieu, HierarchySite } from '@/types/site'
import styles from './SitesView.module.css'

export default function SitesView() {
  const lieux = useSiteStore((s) => s.lieux)
  const loading = useSiteStore((s) => s.loading)
  const viewMode = useUiStore((s) => s.siteViewMode)
  const setViewMode = useUiStore((s) => s.setSiteViewMode)

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>Sites</h1>
          <p className={styles.subtitle}>
            {lieux.length} lieu{lieux.length > 1 ? 'x' : ''} ·{' '}
            {lieux.reduce((n, l) => n + l.sites.length, 0)} site(s)
          </p>
        </div>
        <div className={styles.actions}>
          <button
            className={`${styles.viewBtn} ${viewMode === 'grid' ? styles.viewBtnActive : ''}`}
            onClick={() => setViewMode('grid')}
            title="Grille"
          >
            ▦
          </button>
          <button
            className={`${styles.viewBtn} ${viewMode === 'list' ? styles.viewBtnActive : ''}`}
            onClick={() => setViewMode('list')}
            title="Liste"
          >
            ☰
          </button>
        </div>
      </div>

      {loading ? (
        <div className={styles.loading}>Chargement…</div>
      ) : lieux.length === 0 ? (
        <div className={styles.empty}>
          Aucun site configuré. Créez un lieu puis un site pour commencer.
        </div>
      ) : viewMode === 'grid' ? (
        <div className={styles.grid}>
          {lieux.map((lieu) =>
            lieu.sites.map((site) => (
              <SiteCard key={site.id} site={site} lieu={lieu} />
            )),
          )}
        </div>
      ) : (
        <div className={styles.list}>
          {lieux.map((lieu) =>
            lieu.sites.map((site) => (
              <SiteRow key={site.id} site={site} lieu={lieu} />
            )),
          )}
        </div>
      )}
    </div>
  )
}

function SiteCard({ site, lieu }: { site: HierarchySite; lieu: HierarchyLieu }) {
  const navigate = useNavigate()
  const selectSite = useSiteStore((s) => s.selectSite)
  const totalBenefits = site.cameras.reduce((n, c) => n + c.benefits.length, 0)

  const handleClick = () => {
    selectSite(site.id)
    navigate(`/tracker/${site.id}`)
  }

  return (
    <button className={styles.card} onClick={handleClick}>
      <div className={styles.cardHeader}>
        <span className={styles.cardIcon}>🏢</span>
        <span className={styles.cardName}>{site.name}</span>
      </div>
      <div className={styles.cardMeta}>
        <span className={styles.cardLieu}>📍 {lieu.name}</span>
      </div>
      <div className={styles.cardStats}>
        <span className={styles.cardStat}>{site.cameras.length} caméra{site.cameras.length > 1 ? 's' : ''}</span>
        <span className={styles.cardStatDot}>·</span>
        <span className={styles.cardStat}>{totalBenefits} bénéfice{totalBenefits > 1 ? 's' : ''}</span>
      </div>
      {site.description && (
        <div className={styles.cardDesc}>{site.description}</div>
      )}
    </button>
  )
}

function SiteRow({ site, lieu }: { site: HierarchySite; lieu: HierarchyLieu }) {
  const navigate = useNavigate()
  const selectSite = useSiteStore((s) => s.selectSite)
  const totalBenefits = site.cameras.reduce((n, c) => n + c.benefits.length, 0)

  const handleClick = () => {
    selectSite(site.id)
    navigate(`/tracker/${site.id}`)
  }

  return (
    <button className={styles.row} onClick={handleClick}>
      <span className={styles.rowIcon}>🏢</span>
      <span className={styles.rowName}>{site.name}</span>
      <span className={styles.rowLieu}>📍 {lieu.name}</span>
      <span className={styles.rowStat}>{site.cameras.length} cam</span>
      <span className={styles.rowStat}>{totalBenefits} ben.</span>
    </button>
  )
}
