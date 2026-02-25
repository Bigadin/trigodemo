import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { useMemo } from 'react'
import { useHierarchy } from '@/context/HierarchyContext'
import { findLieuById, getFirstCameraId, getCameraVideoPath, countForms, getSiteBenefits } from '@/utils/hierarchy'
import { getFrameUrl, getStreamUrl } from '@/api/tracker'
import { fetchStreams } from '@/api/tracker'
import { useState, useEffect, useRef } from 'react'
import type { StreamInfo } from '@/api/tracker'
import { createSite } from '@/api/sites'
import { getLocColor, getLieuIcon, icon } from '@/utils/theme'
import { MessageLoading } from '@/components/ui/MessageLoading'
import {
  loadSkillsConfig,
  getSkillGroups,
  getCategoryGroupsBySkill,
} from '@/api/skills'

const FALLBACK_ICON = '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg'
const MAX_CATEGORIES_DISPLAY = 3
const TONE_BY_SKILL: Record<string, string> = {
  detection: 'success',
  counting: 'primary',
  heatmap: 'info',
  quality: 'warning',
}
const ACCENT_PALETTE = ['mint', 'sky', 'violet', 'peach', 'rose', 'sand', 'teal', 'slate', 'lime', 'coral', 'azure', 'amber']

function accentFromKey(key: string): string {
  return ACCENT_PALETTE[Math.abs([...key].reduce((h, c) => ((h << 5) - h) + c.charCodeAt(0), 0)) % ACCENT_PALETTE.length]
}

function toWord(txt: string): string {
  return String(txt || '').trim().split(/\s+/)[0] || ''
}

function SiteBenefitsCard({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [canScroll, setCanScroll] = useState(false)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const check = () => {
      setCanScroll(el.scrollHeight > el.clientHeight)
    }
    const t = setTimeout(check, 0)
    const ro = new ResizeObserver(check)
    ro.observe(el)
    return () => {
      clearTimeout(t)
      ro.disconnect()
    }
  }, [children])

  const scroll = (dy: number) => {
    scrollRef.current?.scrollBy({ top: dy, behavior: 'smooth' })
  }
  return (
    <div className={className}>
      {canScroll && (
      <div className={styles.siteBenefitsScrollBtns}>
      <button
        type="button"
        className={styles.siteBenefitsScrollBtn}
        onClick={(e) => {
          e.stopPropagation()
          scroll(-48)
        }}
          title="Défiler vers le haut"
          aria-label="Défiler vers le haut"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 15l-6-6-6 6" />
          </svg>
        </button>
      <button
        type="button"
        className={styles.siteBenefitsScrollBtn}
        onClick={(e) => {
          e.stopPropagation()
          scroll(48)
        }}
          title="Défiler vers le bas"
          aria-label="Défiler vers le bas"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 9l6 6 6-6" />
          </svg>
                        </button>
      </div>
      )}
      <div ref={scrollRef} className={styles.siteBenefitsScrollArea}>
        {children}
      </div>
    </div>
  )
}

import styles from './LieuView.module.css'

function CameraCell({
  cameraId,
  name,
  videoPath,
  isStreaming,
  onOpen,
}: {
  cameraId: string
  name: string
  videoPath: string
  isStreaming: boolean
  onOpen: (e?: React.MouseEvent) => void
}) {
  const src = isStreaming ? getStreamUrl(videoPath, false) : getFrameUrl(videoPath)
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)

  return (
    <button
      type="button"
      className={styles.camCell}
      onClick={(e) => {
        e.stopPropagation()
        onOpen(e)
      }}
      title={`Ouvrir ${name}`}
    >
      {!failed && (
        <img
          src={src}
          alt={name}
          className={`${styles.camImg} ${loaded ? styles.camImgVisible : ''}`}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
      )}
      <div className={`${styles.camPlaceholder} ${loaded || failed ? styles.camPlaceholderHidden : styles.camPlaceholderVisible}`}>
        {!loaded && !failed && <MessageLoading className={styles.camPlaceholderLoading} />}
      </div>
      <span className={styles.camLabel}>{name || cameraId}</span>
    </button>
  )
}

function CameraCellEmpty() {
  return (
    <div className={styles.camCellEmpty}>
      <img src={icon('camera')} className={styles.camPlaceholderIcon} alt="" />
    </div>
  )
}

export default function LieuView() {
  const { lieuId } = useParams<{ lieuId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedSiteId = searchParams.get('site')
  const navigate = useNavigate()

  const selectSite = (siteId: string) => {
    setSearchParams((p) => {
      const next = new URLSearchParams(p)
      if (selectedSiteId === siteId) {
        next.delete('site')
      } else {
        next.set('site', siteId)
      }
      return next
    })
  }
  const { hierarchy, loading, refetch } = useHierarchy()
  const [showAddSite, setShowAddSite] = useState(false)
  const [newSiteName, setNewSiteName] = useState('')
  const [newSiteId, setNewSiteId] = useState('')
  const [createLoading, setCreateLoading] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [streams, setStreams] = useState<StreamInfo[]>([])

  const lieu = useMemo(() => {
    if (!lieuId) return null
    return findLieuById(hierarchy, lieuId)
  }, [hierarchy, lieuId])

  useEffect(() => {
    fetchStreams()
      .then((r) => setStreams(r.streams || []))
      .catch(() => setStreams([]))
  }, [])

  useEffect(() => {
    loadSkillsConfig()
  }, [])

  useEffect(() => {
    if (selectedSiteId) {
      const el = document.getElementById(`site-selected-${selectedSiteId}`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [selectedSiteId])

  const activeStreams = useMemo(() => {
    const set = new Set<string>()
    streams.filter((s) => s.active).forEach((s) => set.add(s.video))
    return set
  }, [streams])

  const sites = useMemo(() => Object.entries(lieu?.sites || {}), [lieu])
  const lieuMetrics = useMemo(() => {
    let totalCams = 0
    let totalZones = 0
    for (const [, s] of sites) {
      totalCams += Object.keys(s.cameras || {}).length
      totalZones += countForms(s)
    }
    return { sites: sites.length, cameras: totalCams, zones: totalZones }
  }, [sites])
  const [expandedSites, setExpandedSites] = useState<Record<string, boolean>>({})
  const [lieuImgFailed, setLieuImgFailed] = useState(false)
  const lieuIconSrc = lieuId ? getLieuIcon(lieuId, lieu?.icon) : ''

  useEffect(() => {
    setLieuImgFailed(false)
  }, [lieuId])

  if (loading) {
    return (
      <div className={styles.page}>
        <div className={styles.loading}>Chargement…</div>
      </div>
    )
  }

  if (!lieuId || !lieu) {
    return (
      <div className={styles.page}>
        <div className={styles.empty}>Lieu introuvable</div>
      </div>
    )
  }

  const handleCreateSite = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!lieuId || !newSiteName.trim() || !newSiteId.trim()) return
    setCreateLoading(true)
    setCreateError(null)
    try {
      await createSite({
        site_id: newSiteId.trim().toLowerCase().replace(/\s+/g, '_'),
        name: newSiteName.trim(),
        lieu_id: lieuId,
      })
      await refetch()
      setShowAddSite(false)
      setNewSiteName('')
      setNewSiteId('')
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Erreur')
    } finally {
      setCreateLoading(false)
    }
  }

  if (sites.length === 0) {
    return (
      <div className={styles.page}>
        <div className={styles.lieuHeader}>
          <div className={styles.lieuHeaderLeft}>
            {lieuIconSrc && !lieuImgFailed ? (
              <div className={styles.lieuAvatarWrap}>
                <img
                  src={lieuIconSrc}
                  alt=""
                  className={styles.lieuAvatar}
                  onError={() => setLieuImgFailed(true)}
                />
              </div>
            ) : (
              <div className={styles.lieuAvatarFallback}>
                <img src={icon('location')} alt="" />
              </div>
            )}
            <div className={styles.lieuTitleWrap}>
              <h2 className={styles.lieuTitle}>{lieu.name}</h2>
            </div>
          </div>
          {!showAddSite && (
            <button
              type="button"
              className={styles.addSiteBtnHeader}
              onClick={() => setShowAddSite(true)}
            >
              + Créer un site
            </button>
          )}
        </div>
        <header className={styles.header}>
          <div className={styles.trackerNav}>
            <button
              type="button"
              className={styles.trackerBackBtn}
              onClick={() => navigate('/')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 12H5" />
                <path d="M12 19l-7-7 7-7" />
              </svg>
              Vue d&apos;ensemble
            </button>
            <div className={styles.trackerBreadcrumb}>
              <span className={styles.bcSep}>/</span>
              <img src={icon('location')} className={styles.bcIcon} alt="" />
              <span className={styles.bcCurrent}>{lieu.name}</span>
            </div>
          </div>
        </header>
        {showAddSite && (
          <div className={styles.addSiteBlock}>
            <form className={styles.addSiteForm} onSubmit={handleCreateSite}>
              <input
                type="text"
                className={styles.addSiteInput}
                placeholder="ID du site (ex: site_entrepot)"
                value={newSiteId}
                onChange={(e) => setNewSiteId(e.target.value)}
                required
              />
              <input
                type="text"
                className={styles.addSiteInput}
                placeholder="Nom du site"
                value={newSiteName}
                onChange={(e) => setNewSiteName(e.target.value)}
                required
              />
              <div className={styles.addSiteActions}>
                <button type="submit" className={styles.addSiteSubmit} disabled={createLoading}>
                  {createLoading ? 'Création…' : 'Créer'}
                </button>
                <button
                  type="button"
                  className={styles.addSiteCancel}
                  onClick={() => {
                    setShowAddSite(false)
                    setNewSiteName('')
                    setNewSiteId('')
                    setCreateError(null)
                  }}
                >
                  Annuler
                </button>
              </div>
              {createError && <p className={styles.addSiteError}>{createError}</p>}
            </form>
          </div>
        )}
        {!showAddSite && <p className={styles.empty}>Aucun site dans ce lieu</p>}
      </div>
    )
  }

  const locColor = getLocColor(lieu.name)

  return (
    <div className={styles.page}>
      <div className={styles.lieuHeader}>
        <div className={styles.lieuHeaderLeft}>
          {lieuIconSrc && !lieuImgFailed ? (
            <div className={styles.lieuAvatarWrap}>
              <img
                src={lieuIconSrc}
                alt=""
                className={styles.lieuAvatar}
                onError={() => setLieuImgFailed(true)}
              />
            </div>
          ) : (
            <div className={styles.lieuAvatarFallback}>
              <img src={icon('location')} alt="" />
            </div>
          )}
          <div className={styles.lieuTitleWrap}>
            <h2 className={styles.lieuTitle}>{lieu.name}</h2>
            <div className={styles.lieuMetrics}>
            <span className={styles.lieuMetric}>
              <strong>{lieuMetrics.sites}</strong> site{lieuMetrics.sites > 1 ? 's' : ''}
            </span>
            <span className={styles.lieuMetric}>
              <strong>{lieuMetrics.cameras}</strong> caméra{lieuMetrics.cameras > 1 ? 's' : ''}
            </span>
            <span className={styles.lieuMetric}>
              <strong>{lieuMetrics.zones}</strong> zone{lieuMetrics.zones !== 1 ? 's' : ''}
            </span>
            {lieu.address && (
              <span className={styles.lieuMetric}>
                {lieu.address}
              </span>
            )}
          </div>
        </div>
      </div>
        {!showAddSite && (
          <button
            type="button"
            className={styles.addSiteBtnHeader}
            onClick={() => setShowAddSite(true)}
          >
            + Créer un site
          </button>
        )}
      </div>

      <header className={styles.header}>
        <div className={styles.trackerNav}>
          <button
            type="button"
            className={styles.trackerBackBtn}
            onClick={() => navigate('/')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5" />
              <path d="M12 19l-7-7 7-7" />
            </svg>
            Vue d&apos;ensemble
          </button>
          <div className={styles.trackerBreadcrumb}>
            <span className={styles.bcSep}>/</span>
            <img src={icon('location')} className={styles.bcIcon} alt="" />
            <span className={styles.bcCurrent}>{lieu.name}</span>
          </div>
        </div>
      </header>

      {showAddSite && (
        <div className={styles.addSiteBlock}>
          <form className={styles.addSiteForm} onSubmit={handleCreateSite}>
            <input
              type="text"
              className={styles.addSiteInput}
              placeholder="ID du site (ex: site_entrepot)"
              value={newSiteId}
              onChange={(e) => setNewSiteId(e.target.value)}
              required
            />
            <input
              type="text"
              className={styles.addSiteInput}
              placeholder="Nom du site"
              value={newSiteName}
              onChange={(e) => setNewSiteName(e.target.value)}
              required
            />
            <div className={styles.addSiteActions}>
              <button type="submit" className={styles.addSiteSubmit} disabled={createLoading}>
                {createLoading ? 'Création…' : 'Créer'}
              </button>
              <button
                type="button"
                className={styles.addSiteCancel}
                onClick={() => {
                  setShowAddSite(false)
                  setNewSiteName('')
                  setNewSiteId('')
                  setCreateError(null)
                }}
              >
                Annuler
              </button>
            </div>
            {createError && <p className={styles.addSiteError}>{createError}</p>}
          </form>
        </div>
      )}

      <section className={styles.content}>
        {sites.map(([siteId, site]) => {
          const cams = Object.entries(site.cameras || {})
          const forms = countForms(site)
          const firstCamId = getFirstCameraId(site)
          const isExpanded = expandedSites[siteId] ?? false
          const defaultSlots = 3
          const canExpand = cams.length > defaultSlots
          const slotCount = isExpanded ? cams.length : defaultSlots

          const isSelected = selectedSiteId === siteId
          return (
            <article
              key={siteId}
              id={isSelected ? `site-selected-${siteId}` : undefined}
              className={`${styles.siteBlock} ${isSelected ? styles.siteBlockSelected : ''} ${styles.siteBlockClickable}`}
              style={{ ['--site-color' as string]: locColor }}
              role="button"
              tabIndex={0}
              onClick={() => selectSite(siteId)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  selectSite(siteId)
                }
              }}
              title="Sélectionner la carte"
            >
              <div className={styles.siteInfo}>
                <div className={styles.siteInfoHeader}>
                  <span className={styles.siteInfoSq} style={{ background: locColor }} />
                  <h3 className={styles.siteInfoTitle}>{site.name}</h3>
                </div>
                <div className={styles.siteInfoStats}>
                  <span className={styles.siteInfoStat}>
                    <img src={icon('camera')} className={styles.siteInfoIcon} alt="" />
                    {cams.length} caméra{cams.length > 1 ? 's' : ''}
                  </span>
                  <span className={styles.siteInfoStat}>
                    <img src={icon('zone')} className={styles.siteInfoIcon} alt="" />
                    {forms} zone{forms !== 1 ? 's' : ''}
                  </span>
                </div>
                {site.address && (
                  <p className={styles.siteInfoAddr}>{site.address}</p>
                )}
                <button
                  type="button"
                  className={styles.siteOpenBtn}
                  onClick={(e) => {
                    e.stopPropagation()
                    firstCamId && navigate(`/tracker/${siteId}?cam=${firstCamId}`)
                  }}
                  disabled={!firstCamId}
                  title={firstCamId ? 'Ouvrir le Zone Tracker' : 'Aucune caméra'}
                >
                  Ouvrir le site
                </button>
              </div>

              <div className={styles.siteBlockRight}>
                {(() => {
                  const benefits = getSiteBenefits(site)
                  const hasBenefits = benefits.length > 0
                  const skillGroups = getSkillGroups()
                  return (
                    <>
                      {hasBenefits && (
                        <SiteBenefitsCard className={styles.siteBenefitsCard}>
                          <div className={styles.siteBenefitsList}>
                              {benefits.map((b) => {
                                const sk = b.skill ?? 'detection'
                                const tone = TONE_BY_SKILL[sk] ?? 'primary'
                                const activeSkill = skillGroups.find((g) => g.key === sk)
                                const categoryGroups = getCategoryGroupsBySkill(sk)
                                const catByKey: Record<string, { icon: string; label: string }> = {}
                                categoryGroups.forEach((g) => (g.items ?? []).forEach((it) => {
                                  catByKey[`${g.key}::${it.id}`] = { icon: it.icon || g.icon, label: it.label || g.label || '' }
                                }))
                                const selectedCats = (b.categories ?? [])
                                  .map((k) => {
                                    const c = catByKey[k]
                                    return c ? { key: k, ...c } : null
                                  })
                                  .filter((x): x is { key: string; icon: string; label: string } => !!x)
                                  .slice(0, MAX_CATEGORIES_DISPLAY)
                                const skillSubItem = activeSkill?.items?.find((i) => i.id === (b.skill_item ?? ''))
                                const chips: { icon: string; label: string; appearance: 'solid' | 'ghost'; tone: string; accent?: string; iconOnly: boolean }[] = []
                                if (activeSkill) {
                                  chips.push({ icon: activeSkill.icon, label: toWord(activeSkill.label), appearance: 'solid', tone, iconOnly: false })
                                }
                                if (selectedCats.length > 0) {
                                  selectedCats.forEach(({ key, icon: ico }) => chips.push({
                                    icon: ico,
                                    label: '',
                                    appearance: 'ghost',
                                    tone,
                                    accent: accentFromKey(key),
                                    iconOnly: true,
                                  }))
                                } else if (skillSubItem) {
                                  chips.push({
                                    icon: skillSubItem.icon || (activeSkill?.icon ?? ''),
                                    label: '',
                                    appearance: 'ghost',
                                    tone,
                                    accent: accentFromKey(b.skill_item ?? ''),
                                    iconOnly: true,
                                  })
                                }
                                return (
                                  <div key={b.benefit_id} className={styles.siteBenefitRow}>
                                    <div className={styles.siteBenefitChips}>
                                      {chips.map((c, i) => (
                                        <span
                                          key={i}
                                          className={`${styles.siteBenefitChip} ${styles[`siteBenefitChip--${c.appearance}`]} ${styles[`siteBenefitChip--${c.tone}`]}${c.iconOnly ? ` ${styles.siteBenefitChipIconOnly}` : ''}${c.accent ? ` ${styles[`siteBenefitChipAccent${c.accent.charAt(0).toUpperCase() + c.accent.slice(1)}`]}` : ''}`}
                                        >
                                          {c.icon && <img className={styles.siteBenefitChipIcon} src={c.icon} alt="" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />}
                                          {!c.iconOnly && <span>{c.label}</span>}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                        </SiteBenefitsCard>
                      )}
                      <div className={styles.camGridWrap} style={hasBenefits ? { marginLeft: 292 } : undefined}>
                  {canExpand && (
                    <button
                      type="button"
                      className={`${styles.expandBtn} ${isExpanded ? styles.expanded : ''}`}
                      onClick={(e) => {
                        e.stopPropagation()
                        setExpandedSites((p) => ({ ...p, [siteId]: !isExpanded }))
                      }}
                      title={isExpanded ? 'Replier' : 'Voir toutes les caméras'}
                    >
                      {isExpanded ? 'Replier' : `+${cams.length - defaultSlots} caméra${cams.length - defaultSlots > 1 ? 's' : ''}`}
                    </button>
                  )}
                  <div className={`${styles.camGrid} ${isExpanded ? styles.camGridExpanded : ''}`}>
                    {Array.from({ length: slotCount }).map((_, i) => {
                      const cam = cams[i]
                      if (!cam) return <CameraCellEmpty key={`empty-${i}`} />
                      const [camId, camera] = cam
                      const videoPath = getCameraVideoPath(camera)
                      const isStreaming = activeStreams.has(videoPath)
                      return (
                        <CameraCell
                          key={camId}
                          cameraId={camId}
                          name={camera.name || camId}
                          videoPath={videoPath}
                          isStreaming={isStreaming}
                          onOpen={() => navigate(`/tracker/${siteId}?cam=${camId}`)}
                        />
                      )
                    })}
                  </div>
                </div>
                    </>
                  )
                })()}
              </div>
            </article>
          )
        })}
      </section>
    </div>
  )
}
