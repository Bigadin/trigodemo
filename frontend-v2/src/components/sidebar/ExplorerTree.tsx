import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { useHierarchy } from '@/context/HierarchyContext'
import { useSession, benefitElapsedKey } from '@/context/SessionContext'
import { toggleBenefit } from '@/api/benefits'
import { getLocColor, icon } from '@/utils/theme'
import styles from './ExplorerTree.module.css'
import animStyles from './ExplorerTree.animations.module.css'

const getBenefitColor = () => '#c47b5a'

function formatTime(seconds: number): string {
  const n = Math.max(0, Number(seconds) || 0)
  const h = Math.floor(n / 3600)
  const m = Math.floor((n % 3600) / 60)
  const s = Math.floor(n % 60)
  return [h, m, s].map((v) => v.toString().padStart(2, '0')).join(':')
}

export default function ExplorerTree() {
  const { hierarchy, loading, refetch } = useHierarchy()
  const { benefitElapsed, streamingSiteId, streamingCamId } = useSession()
  const navigate = useNavigate()
  const { siteId } = useParams<{ siteId?: string; lieuId?: string }>()
  const [searchParams] = useSearchParams()
  const selectedCamId = searchParams.get('cam') ?? null
  const selectedSiteFromLieu = searchParams.get('site') ?? null
  const selectedBenefitIdFromUrl = searchParams.get('benefit')
  const [selectedBenefit, setSelectedBenefit] = useState<{ camId: string; benId: string } | null>(null)

  const [collapsedSites, setCollapsedSites] = useState<Record<string, boolean>>({})
  const [collapsedCams, setCollapsedCams] = useState<Record<string, boolean>>({})

  /* Quand on sélectionne un site : déplier le site. En Vue Tracker, déplier aussi les caméras. En Vue Lieu, garder les caméras repliées (bénéfices masqués). */
  useEffect(() => {
    const siteToExpand = selectedSiteFromLieu || siteId
    if (!siteToExpand) return
    setCollapsedSites((p) => {
      const isCollapsed = p[siteToExpand] ?? true
      if (isCollapsed) return { ...p, [siteToExpand]: false }
      return p
    })
    if (!siteId || siteToExpand !== siteId) return
    const camsToExpand = selectedCamId
      ? [selectedCamId]
      : (() => {
          let ids: string[] = []
          for (const lieu of Object.values(hierarchy)) {
            const site = lieu.sites?.[siteToExpand]
            if (site) {
              ids = Object.keys(site.cameras || {})
              break
            }
          }
          return ids
        })()
    if (camsToExpand.length > 0) {
      setCollapsedCams((p) => {
        const next = { ...p }
        let changed = false
        camsToExpand.forEach((cid) => {
          if (p[cid] ?? true) { next[cid] = false; changed = true }
        })
        return changed ? next : p
      })
    }
  }, [selectedSiteFromLieu, siteId, selectedCamId, hierarchy])

  /* Quand on sélectionne un bénéfice dans Overview, déplier l'explorer et sélectionner */
  useEffect(() => {
    if (!selectedBenefitIdFromUrl || !siteId || !selectedCamId) return
    setSelectedBenefit({ camId: selectedCamId, benId: selectedBenefitIdFromUrl })
    setCollapsedSites((p) => ({ ...p, [siteId]: false }))
    setCollapsedCams((p) => ({ ...p, [selectedCamId]: false }))
  }, [selectedBenefitIdFromUrl, siteId, selectedCamId])

  const handleBenefitToggle = async (e: React.MouseEvent, benId: string, currentActive: boolean) => {
    e.stopPropagation()
    try {
      await toggleBenefit(benId, !currentActive)
      refetch()
    } catch (err) {
      console.warn('Toggle benefit failed:', err)
    }
  }

  if (loading) {
    return <span className={styles.empty}>Chargement…</span>
  }

  const lieux = Object.entries(hierarchy)
  if (lieux.length === 0) {
    return <span className={styles.empty}>Aucun lieu configuré</span>
  }

  return (
    <div className={styles.tree}>
      {lieux.map(([lieuId, lieu]) => {
        const sites = Object.entries(lieu.sites || {})
        const locColor = getLocColor(lieu.name)

        return (
          <div key={lieuId} className={styles.group} style={{ ['--sb-loc-color' as string]: locColor }}>
            <div
              className={styles.loc}
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
              <span className={styles.locName}>
                <img src={icon('location')} className={styles.hierIcon} alt="" />
                <span>{lieu.name}</span>
                <span className={styles.meta}>{sites.length}</span>
              </span>
            </div>

            <div className={styles.children}>
              {sites.map(([sid, site]) => {
                const cams = Object.entries(site.cameras || {})
                const isSiteActive = (!!lieuId && selectedSiteFromLieu === sid) || siteId === sid

                const isSiteCollapsed = collapsedSites[sid] ?? true

                return (
                  <div key={sid}>
                    <div
                      className={`${styles.item} ${styles.site} ${isSiteActive ? styles.active : ''}`}
                      onClick={() => {
                        const isActive = (!!lieuId && selectedSiteFromLieu === sid) || siteId === sid
                        if (isActive) {
                          const willCollapse = !(collapsedSites[sid] ?? true)
                          setCollapsedSites((p) => ({ ...p, [sid]: willCollapse }))
                          if (willCollapse) {
                            setCollapsedCams((p) => {
                              const next = { ...p }
                              cams.forEach(([camId]) => { next[camId] = true })
                              return next
                            })
                          }
                        } else {
                          navigate(`/lieu/${lieuId}?site=${sid}`)
                          setCollapsedSites((p) => ({ ...p, [sid]: false }))
                        }
                      }}
                      title={isSiteActive ? (isSiteCollapsed ? 'Déplier les caméras' : 'Replier les caméras') : undefined}
                    >
                      <img src={icon('site')} className={styles.hierIcon} alt="" />
                      <span className={styles.itemName}>{site.name}</span>
                      <span className={styles.meta}>{cams.length}</span>
                    </div>

                    <div
                      className={`${animStyles.collapsible} ${!isSiteCollapsed ? animStyles.expanded : ''}`}
                    >
                      <div className={`${animStyles.collapsibleInner} ${styles.childrenSite}`}>
                      {cams.map(([camId, cam]) => {
                        const benefits = Object.entries(cam.benefits || {})
                        const isCamActive = isSiteActive && selectedCamId === camId
                        const isCamCollapsed = collapsedCams[camId] ?? true

                        return (
                          <div key={camId}>
                            <div
                              className={`${styles.item} ${styles.cam} ${isCamActive ? styles.active : ''}`}
                              onClick={() => {
                                if (isCamActive) {
                                  setCollapsedCams((p) => ({ ...p, [camId]: !(p[camId] ?? true) }))
                                } else {
                                  navigate(`/tracker/${sid}?cam=${camId}`)
                                  setCollapsedSites((p) => ({ ...p, [sid]: false }))
                                  setCollapsedCams((p) => ({ ...p, [camId]: false }))
                                }
                              }}
                              title={isCamActive ? (isCamCollapsed ? 'Déplier les bénéfices' : 'Replier les bénéfices') : undefined}
                            >
                              <img src={icon('camera')} className={styles.hierIcon} alt="" />
                              <span className={styles.itemName}>{cam.name || camId}</span>
                              <span className={styles.meta}>{benefits.length}</span>
                            </div>

                            <div
                              className={`${animStyles.collapsible} ${!isCamCollapsed ? animStyles.expanded : ''}`}
                            >
                              <div className={`${animStyles.collapsibleInner} ${styles.childrenCam}`}>
                              {benefits.map(([benId, ben]) => {
                                const isSelected =
                                  selectedBenefit?.camId === camId && selectedBenefit?.benId === benId
                                const benEnabled = ben.active !== false
                                const isStreamingThisCam = streamingSiteId === sid && streamingCamId === camId
                                const key = benefitElapsedKey(sid, camId, benId)
                                const timerText = isStreamingThisCam ? formatTime(benefitElapsed[key] ?? 0) : '00:00:00'

                                return (
                                  <div
                                    key={benId}
                                    className={`${styles.itemBenefit} ${!isCamActive ? styles.dim : ''} ${isSelected ? styles.selected : ''}`}
                                    onClick={(e) => {
                                      if ((e.target as HTMLElement).closest(`.${styles.toggle}`)) return
                                      setSelectedBenefit({ camId, benId })
                                      navigate(`/tracker/${sid}?cam=${camId}&benefit=${benId}`)
                                    }}
                                  >
                                    <span
                                      className={styles.benefitDot}
                                      style={{ background: getBenefitColor() }}
                                    />
                                    <span className={styles.benefitName}>
                                      {(ben.name || benId).length > 10
                                        ? (ben.name || benId).slice(0, 10) + '…'
                                        : ben.name || benId}
                                    </span>
                                    <span className={styles.time}>{timerText}</span>
                                    <button
                                      type="button"
                                      className={`${styles.toggle} ${benEnabled ? styles.on : ''}`}
                                      onClick={(e) => handleBenefitToggle(e, benId, benEnabled)}
                                      title={benEnabled ? 'Désactiver' : 'Activer'}
                                    >
                                      <span className={styles.toggleTrack}>
                                        <span className={styles.toggleThumb} />
                                      </span>
                                    </button>
                                  </div>
                                )
                              })}
                              <div
                                className={styles.addBenefit}
                                onClick={() => navigate(`/tracker/${sid}?cam=${camId}&addBenefit=1`)}
                              >
                                <span className={styles.addBenefitTitle}>+ Créer un bénéfice</span>
                              </div>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}
