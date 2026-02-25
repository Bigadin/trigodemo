import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useHierarchy } from '@/context/HierarchyContext'
import type { HierarchyLieu, HierarchySite, HierarchyCamera } from '@/types/hierarchy'
import { getLocColor, getLieuIcon, icon } from '@/utils/theme'
import { countForms } from '@/utils/hierarchy'
import { getCameraVideoPath } from '@/utils/hierarchy'
import { getFrameUrl } from '@/api/tracker'
import { createLieu } from '@/api/lieux'
import CardMenu from '@/components/ui/CardMenu'
import { MessageLoading } from '@/components/ui/MessageLoading'
import styles from './SitesView.module.css'

const CAM_THUMB_SLOTS = 2

function CameraThumb({ cam }: { cam: HierarchyCamera }) {
  const videoPath = getCameraVideoPath(cam)
  const [src] = useState(() => getFrameUrl(videoPath))
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)

  return (
    <div className={styles.camThumbWrap}>
      <div
        className={`${styles.camThumbPlaceholder} ${loaded || failed ? styles.camThumbPlaceholderHidden : ''}`}
        aria-hidden
      >
        {!loaded && !failed && <MessageLoading className={styles.camThumbLoading} />}
      </div>
      {!failed && (
        <img
          src={src}
          alt={cam.name || cam.camera_id}
          className={`${styles.camThumb} ${loaded ? styles.camThumbVisible : ''}`}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
      )}
    </div>
  )
}

function CameraThumbEmpty() {
  return (
    <div className={styles.camThumbEmpty}>
      <img src={icon('camera')} className={styles.camThumbPlaceholderIcon} alt="" />
    </div>
  )
}

export default function SitesView() {
  const { hierarchy, refetch } = useHierarchy()
  const navigate = useNavigate()
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [showAddLieu, setShowAddLieu] = useState(false)
  const [newLieuId, setNewLieuId] = useState('')
  const [newLieuName, setNewLieuName] = useState('')
  const [createLoading, setCreateLoading] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const lieux = Object.entries(hierarchy)
  const flatSites: { site: HierarchySite; lieu: HierarchyLieu }[] = []
  for (const [, lieu] of lieux) {
    for (const [, site] of Object.entries(lieu.sites || {})) {
      flatSites.push({ site, lieu })
    }
  }

  const handleCreateLieu = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newLieuId.trim() || !newLieuName.trim()) return
    setCreateLoading(true)
    setCreateError(null)
    try {
      await createLieu({
        lieu_id: newLieuId.trim().toLowerCase().replace(/\s+/g, '_'),
        name: newLieuName.trim(),
      })
      await refetch()
      setShowAddLieu(false)
      setNewLieuId('')
      setNewLieuName('')
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Erreur')
    } finally {
      setCreateLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.pageContent}>
      {/* Bloc principal : même structure que LieuView pour transition fluide */}
      <div className={styles.overviewHeader}>
        <div className={styles.overviewHeaderLeft}>
          <div className={styles.overviewIconWrap}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
          </div>
          <div className={styles.overviewTitleWrap}>
            <h2 className={styles.overviewTitle}>Vue d&apos;ensemble</h2>
            <div className={styles.overviewMetrics}>
              <span className={styles.overviewMetric}>
                <strong>{lieux.length}</strong> lieu{lieux.length > 1 ? 'x' : ''}
              </span>
              <span className={styles.overviewMetric}>
                <strong>{flatSites.length}</strong> site{flatSites.length > 1 ? 's' : ''}
              </span>
            </div>
          </div>
        </div>
        {!showAddLieu && (
          <button
            type="button"
            className={styles.addLieuBtnHeader}
            onClick={() => setShowAddLieu(true)}
          >
            + Créer un lieu
          </button>
        )}
      </div>

      <header className={styles.header}>
        <div className={styles.headerSpacer} />
        <div className={styles.viewToggle}>
          <button
            className={`${styles.toggleBtn} ${viewMode === 'grid' ? styles.active : ''}`}
            onClick={() => setViewMode('grid')}
            title="Grille"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7" rx="1" />
              <rect x="14" y="3" width="7" height="7" rx="1" />
              <rect x="3" y="14" width="7" height="7" rx="1" />
              <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
          </button>
          <button
            className={`${styles.toggleBtn} ${viewMode === 'list' ? styles.active : ''}`}
            onClick={() => setViewMode('list')}
            title="Liste"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="8" y1="6" x2="21" y2="6" />
              <line x1="8" y1="12" x2="21" y2="12" />
              <line x1="8" y1="18" x2="21" y2="18" />
              <circle cx="4" cy="6" r="1" fill="currentColor" />
              <circle cx="4" cy="12" r="1" fill="currentColor" />
              <circle cx="4" cy="18" r="1" fill="currentColor" />
            </svg>
          </button>
        </div>
      </header>

      {showAddLieu && (
        <div className={styles.addLieuBlock}>
          <form className={styles.addLieuForm} onSubmit={handleCreateLieu}>
            <input
              type="text"
              className={styles.addLieuInput}
              placeholder="ID du lieu (ex: usine_paris)"
              value={newLieuId}
              onChange={(e) => setNewLieuId(e.target.value)}
              required
            />
            <input
              type="text"
              className={styles.addLieuInput}
              placeholder="Nom du lieu"
              value={newLieuName}
              onChange={(e) => setNewLieuName(e.target.value)}
              required
            />
            <div className={styles.addLieuActions}>
              <button type="submit" className={styles.addLieuSubmit} disabled={createLoading}>
                {createLoading ? 'Création…' : 'Créer'}
              </button>
              <button
                type="button"
                className={styles.addLieuCancel}
                onClick={() => {
                  setShowAddLieu(false)
                  setNewLieuId('')
                  setNewLieuName('')
                  setCreateError(null)
                }}
              >
                Annuler
              </button>
            </div>
            {createError && <p className={styles.addLieuError}>{createError}</p>}
          </form>
        </div>
      )}

      <section className={styles.content}>
        {flatSites.length === 0 ? (
          <div className={styles.empty}>Aucun site. Créez un lieu pour commencer.</div>
        ) : viewMode === 'grid' ? (
          <div className={styles.grid}>
            {lieux.map(([lieuId, lieu]) => {
              const sites = Object.entries(lieu.sites || {})
              const totalCams = sites.reduce((n, [, s]) => n + Object.keys(s.cameras || {}).length, 0)
              const totalForms = sites.reduce((n, [, s]) => n + countForms(s), 0)
              const locColor = getLocColor(lieu.name)
              const lieuIconSrc = getLieuIcon(lieuId, lieu.icon)

              return (
                <div key={lieuId} className={styles.locGroup}>
                  <div
                    className={styles.locGroupHeader}
                    style={{ ['--card-loc-color' as string]: locColor }}
                    role="button"
                    tabIndex={0}
                    onClick={() => navigate(`/lieu/${lieuId}`)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/lieu/${lieuId}`)
                      }
                    }}
                    title="Voir le lieu"
                  >
                    <span className={styles.locSq} style={{ background: locColor }} />
                    <span className={styles.locName}>{lieu.name}</span>
                    <span className={styles.locBadges}>
                      {[
                        [sites.length, 'site', 'site'],
                        [totalCams, 'camera', 'cam'],
                        [totalForms, 'zone', 'zone'],
                      ].map(([n, ico, label]) => (
                        <span key={label} className={styles.badge}>
                          <img src={icon(ico as string)} className={styles.hierIcon} alt="" />
                          <strong>{n}</strong> {label}{(n as number) > 1 ? 's' : ''}
                        </span>
                      ))}
                    </span>
                  </div>
                  <div className={styles.locCards}>
                    {sites.map(([siteId, site]) => {
                      const cams = Object.values(site.cameras || {})
                      const forms = countForms(site)

                      return (
                        <button
                          key={siteId}
                          className={styles.siteCard}
                          style={{ ['--card-loc-color' as string]: locColor }}
                          onClick={() => navigate(`/lieu/${lieuId}?site=${siteId}`)}
                        >
                          <div className={styles.cardCamStrip}>
                            {Array.from({ length: CAM_THUMB_SLOTS }, (_, i) =>
                              cams[i] ? (
                                <CameraThumb key={cams[i].camera_id} cam={cams[i]} />
                              ) : (
                                <CameraThumbEmpty key={`empty-${i}`} />
                              )
                            )}
                          </div>
                          <div className={styles.cardTop}>
                            <div>
                              <div className={styles.cardName}>
                                <img src={icon('site')} className={styles.hierIcon} alt="" />
                                {site.name}
                              </div>
                              <div className={styles.cardLocation}>
                                {lieuIconSrc ? (
                                  <img src={lieuIconSrc} alt="" className={styles.cardLocThumb} />
                                ) : (
                                  <img src={icon('location')} className={styles.hierIcon} alt="" />
                                )}
                                {lieu.name}
                              </div>
                            </div>
                            <CardMenu
                              className={styles.menuBtn}
                              options={[
                                { label: 'Ouvrir', onClick: () => navigate(`/lieu/${lieuId}?site=${siteId}`) },
                                { label: 'Modifier', onClick: () => console.log('Modifier', siteId) },
                                { label: 'Exporter', onClick: () => console.log('Exporter', siteId) },
                              ]}
                              title="Options"
                            />
                          </div>
                          <div className={styles.cardStats}>
                            {[
                              [cams.length, 'cam'],
                              [forms, 'zone'],
                            ].map(([n, label]) => (
                              <span key={label} className={styles.stat}>
                                <strong>{n}</strong> {label}{(n as number) > 1 ? 's' : ''}
                              </span>
                            ))}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className={styles.list}>
            {lieux.map(([lieuId, lieu]) => {
              const sites = Object.entries(lieu.sites || {})
              const totalCams = sites.reduce((n, [, s]) => n + Object.keys(s.cameras || {}).length, 0)
              const totalForms = sites.reduce((n, [, s]) => n + countForms(s), 0)
              const locColor = getLocColor(lieu.name)

              return (
                <div key={lieuId} className={styles.locGroupList}>
                  <div
                    className={styles.listHeader}
                    role="button"
                    tabIndex={0}
                    onClick={() => navigate(`/lieu/${lieuId}`)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/lieu/${lieuId}`)
                      }
                    }}
                    title="Voir le lieu"
                  >
                    <span className={styles.listLocSq} style={{ background: locColor }} />
                    <span>{lieu.name}</span>
                    <span className={styles.listCount}>{sites.length} site{sites.length > 1 ? 's' : ''}</span>
                    <span className={styles.listTotals}>
                      {[
                        [totalCams, 'camera', 'cams'],
                        [totalForms, 'zone', 'zones'],
                      ].map(([n, ico, label]) => (
                        <span key={label} className={styles.listStat}>
                          <img src={icon(ico as string)} className={styles.hierIcon} alt="" />
                          <strong>{n}</strong> {label}
                        </span>
                      ))}
                    </span>
                  </div>
                  {sites.map(([siteId, site]) => {
                    const cams = Object.values(site.cameras || {})
                    const forms = countForms(site)

                    return (
                      <button
                        key={siteId}
                        className={styles.listRow}
                        style={{ ['--card-loc-color' as string]: locColor }}
                        onClick={() => navigate(`/lieu/${lieuId}?site=${siteId}`)}
                      >
                        <span className={styles.statusSq} style={{ background: '#6E7180' }} title="Inactif" />
                        <div className={styles.listName}>
                          <img src={icon('site')} className={styles.hierIcon} alt="" />
                          {site.name}
                        </div>
                        <div className={styles.listStats}>
                          {[
                            [cams.length, 'cam'],
                            [forms, 'zone'],
                          ].map(([n, label]) => (
                            <span key={label} className={styles.listStat}>
                              <strong>{n}</strong> {label}{(n as number) > 1 ? 's' : ''}
                            </span>
                          ))}
                        </div>
                        <CardMenu
                          className={styles.menuBtn}
                          options={[
                            { label: 'Ouvrir', onClick: () => navigate(`/lieu/${lieuId}?site=${siteId}`) },
                            { label: 'Modifier', onClick: () => console.log('Modifier', siteId) },
                            { label: 'Exporter', onClick: () => console.log('Exporter', siteId) },
                          ]}
                          title="Options"
                        />
                      </button>
                    )
                  })}
                </div>
              )
            })}
          </div>
        )}
      </section>
      </div>
    </div>
  )
}
