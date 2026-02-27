import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import type { HierarchyBenefit } from '@/types/hierarchy'
import type { ZoneData } from '@/api/tracker'
import type { CountingResponse } from '@/api/tracker'
import {
  configCounting,
  toggleCounting,
  flipCounting,
  resetCounting,
  fetchCountingParams,
  updateCountingParams,
} from '@/api/counting'
import { resetZoneTimer } from '@/api/tracker'
import CardMenu from '@/components/ui/CardMenu'
import LovDropdown from '@/components/ui/LovDropdown'
import { benefitElapsedKey } from '@/context/SessionContext'
import { ActivityChart } from './DataRoomCardsLibrary'
import { ACTIVITY_CHART_INTERVAL_OPTIONS } from './DataRoomCardsLibrary'
import styles from './DataRoomCards.module.css'

const CAT_ICONS: Record<string, string> = {
  human: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
  silhouette: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
  voiture: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg',
  velo: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg',
  transport: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg',
}

const CAT_LABELS: Record<string, string> = {
  human: 'Humain',
  silhouette: 'Humain',
  voiture: 'Voiture',
  velo: 'Vélo',
  transport: 'Transport',
}

const SKILL_ICONS: Record<string, string> = {
  detection: '/static/assets_youn/SvIcons/SVGnew/Yclassify.svg',
  counting: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg',
}

/**
 * Catégories ayant un modèle de détection (bbox + .pt) côté backend.
 * Seules ces classes affichent des données réelles ; les autres affichent "—" / 0.
 * À étendre quand voiture.pt, velo.pt, etc. seront disponibles.
 */
const TRACKED_DETECTION_CATEGORIES = new Set(['human'])

function parseCategories(cats: string[] | undefined): { key: string; category: string; subcategory: string }[] {
  if (!cats?.length) return []
  return cats.map((c) => {
    const [cat, sub] = (c || '').split('::')
    return { key: c, category: cat || '', subcategory: sub || '' }
  })
}

/** Clés de zone pour un bénéfice détection : toujours une courbe par forme include (bid:0, bid:1).
 * Labels = Forme 1, Forme 2… (alignés sur l'éditeur de zones). */
export function getDetectionZoneKeys(
  b: HierarchyBenefit,
  _zones?: Record<string, ZoneData> | null
): { zoneKey: string; label: string }[] {
  const polys = b.zone_polygons ?? []
  const types = b.zone_polygon_types ?? polys.map(() => 'include' as const)
  let includeNum = 0
  const perZone = polys
    .map((_, idx) => {
      if (types[idx] !== 'include') return null
      includeNum += 1
      return { zoneKey: `${b.benefit_id}:${idx}`, label: `Forme ${includeNum}` }
    })
    .filter((x): x is { zoneKey: string; label: string } => x != null)

  if (perZone.length > 0) return perZone
  return [{ zoneKey: b.benefit_id, label: 'Présence' }]
}

function formatTimer(totalSeconds: number): string {
  const n = Math.max(0, Math.floor(totalSeconds))
  const h = Math.floor(n / 3600)
  const m = Math.floor((n % 3600) / 60)
  const s = n % 60
  return [h, m, s].map((v) => v.toString().padStart(2, '0')).join(':')
}

interface PresenceRow {
  key: string
  label: string
  icon: string
  isOccupied: boolean
  presenceTime: number
  pct: number
  /** True si la classe a un modèle de détection (bbox/.pt) ; sinon affiche "—" */
  tracked: boolean
}

function getPresenceRows(
  benefit: HierarchyBenefit | null,
  zones: Record<string, ZoneData> | null,
  sessionElapsed: number,
  presenceAtStart: number | Record<string, number>,
): PresenceRow[] {
  if (!benefit) return []
  const zoneKeys = benefit ? getDetectionZoneKeys(benefit, zones) : []
  let totalPresence = 0
  let isOccupied = false
  let startVal = 0
  if (zoneKeys.length > 0) {
    for (const { zoneKey } of zoneKeys) {
      const z = zones?.[zoneKey]
      totalPresence += z?.total_time ?? 0
      if (z?.is_occupied) isOccupied = true
      startVal += typeof presenceAtStart === 'object' ? (presenceAtStart[zoneKey] ?? 0) : 0
    }
    if (typeof presenceAtStart === 'number' && zoneKeys.length === 1) startVal = presenceAtStart
  } else {
    const zone = zones?.[benefit.benefit_id]
    totalPresence = zone?.total_time ?? 0
    isOccupied = zone?.is_occupied ?? false
    startVal = typeof presenceAtStart === 'object' ? (presenceAtStart[benefit.benefit_id] ?? 0) : presenceAtStart
  }
  const presenceTime = Math.max(0, totalPresence - startVal)
  const pct = sessionElapsed > 0 ? Math.min(100, Math.round((presenceTime / sessionElapsed) * 100)) : 0

  const cats = parseCategories(benefit.categories)
  const rows = cats.length > 0 ? cats : []

  if (rows.length === 0) {
    return [{
      key: 'presence',
      label: 'Présence',
      icon: CAT_ICONS.human,
      isOccupied,
      presenceTime,
      pct,
      tracked: true,
    }]
  }

  return rows.map((c) => {
    const iconKey = c.subcategory || c.category
    const tracked = TRACKED_DETECTION_CATEGORIES.has(c.category)
    return {
      key: c.key,
      label: CAT_LABELS[iconKey] ?? CAT_LABELS[c.category] ?? (c.category || c.subcategory || 'Présence'),
      icon: CAT_ICONS[iconKey] ?? CAT_ICONS[c.category] ?? CAT_ICONS.human,
      isOccupied: tracked ? isOccupied : false,
      presenceTime: tracked ? presenceTime : 0,
      pct: tracked ? pct : 0,
      tracked,
    }
  })
}

function getCountingItems(
  countingBenefit: HierarchyBenefit | null,
  countValue: number
): { label: string; icon: string; count: number }[] {
  const cats = parseCategories(countingBenefit?.categories)
  if (cats.length >= 2) {
    const humanCat = cats.find((c) => c.category === 'human')
    const transportCat = cats.find((c) => c.category === 'transport')
    const items: { label: string; icon: string; count: number }[] = []
    if (humanCat) items.push({ label: 'Personnes', icon: CAT_ICONS.human, count: countValue })
    if (transportCat?.subcategory === 'velo') items.push({ label: 'Vélos', icon: CAT_ICONS.velo, count: Math.floor(countValue * 0.4) })
    if (transportCat?.subcategory === 'voiture') items.push({ label: 'Voitures', icon: CAT_ICONS.voiture, count: Math.floor(countValue * 0.3) })
    if (items.length >= 2) return items
  }
  return [{ label: 'Nombre', icon: '/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg', count: countValue }]
}

/** Options de zone pour le comptage: depuis zones API (backend) + labels depuis benefits */
function getCountingZoneOptions(
  benefits: HierarchyBenefit[],
  zones: Record<string, ZoneData> | null
): { id: string; label: string }[] {
  const countingBenefits = benefits.filter(
    (b) => String(b.skill || '').toLowerCase() === 'counting' && (b.zone_polygons?.length ?? 0) > 0
  )
  const zoneNames = zones ? Object.keys(zones) : []
  const opts: { id: string; label: string }[] = []
  for (const b of countingBenefits) {
    const name = b.name || b.benefit_id
    const types: ('include' | 'exclude')[] =
      b.zone_polygon_types?.length === b.zone_polygons!.length
        ? (b.zone_polygon_types as ('include' | 'exclude')[])
        : b.zone_polygons!.map(() => 'include' as const)
    const matchingZones = zoneNames.filter(
      (z) => z === b.benefit_id || z.startsWith(`${b.benefit_id}:`)
    )
    const shortName = (name.length > 12 ? name.slice(0, 10) + '…' : name)
    if (matchingZones.length === 0) {
      opts.push({ id: b.benefit_id, label: shortName })
    } else {
      for (const z of matchingZones.sort()) {
        if (z === b.benefit_id) {
          opts.push({ id: z, label: types.length > 1 ? `${shortName} · Toutes` : shortName })
        } else {
          const idx = z.split(':')[1]
          opts.push({ id: z, label: `${shortName} · Z${Number(idx) + 1}` })
        }
      }
    }
  }
  return opts
}

interface PresenceCardProps {
  benefit: HierarchyBenefit
  zones: Record<string, ZoneData> | null
  sessionElapsed: number
  presenceAtStart?: number | Record<string, number>
  onEditBenefit?: (benefitId: string) => void
  onDeleteBenefit?: (benefitId: string) => void
  onHideCard?: (benefitId: string) => void
  onRefresh?: () => void
}

/** Format compatible avec StatsSeriesPoint : { date, value } */
export interface StatsSeriesPoint {
  date: string
  value: number
}

const ACTIVITY_ZONE_COLORS = ['#3b82f6', '#4f46e5', '#7c3aed', '#8b5cf6', '#a855f7', '#c026d3']

function getPctColor(pct: number): string {
  const p = Math.min(100, Math.max(0, pct)) / 100
  const hue = 216 + (300 - 216) * p
  return `hsl(${hue} 88% 68%)`
}

/** Miniature SVG de la forme de zone */
function ZoneShapePreview({ poly, type = 'include' }: { poly: number[][]; type?: 'include' | 'exclude' }) {
  if (!poly || poly.length < 3) return null
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of poly) {
    minX = Math.min(minX, p[0])
    minY = Math.min(minY, p[1])
    maxX = Math.max(maxX, p[0])
    maxY = Math.max(maxY, p[1])
  }
  const w = Math.max(1, maxX - minX)
  const h = Math.max(1, maxY - minY)
  const pad = 6
  const vw = 72
  const vh = 72
  const sx = (vw - pad * 2) / w
  const sy = (vh - pad * 2) / h
  const s = Math.min(sx, sy)
  const pts = poly.map((p) => {
    const x = (p[0] - minX) * s + pad
    const y = (p[1] - minY) * s + pad
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const colors = type === 'include'
    ? { fill: 'rgba(34,197,94,0.25)', stroke: '#22c55e' }
    : { fill: 'rgba(240,131,33,0.25)', stroke: '#f08321' }
  return (
    <svg viewBox={`0 0 ${vw} ${vh}`} width={72} height={72} className={styles.zoneShapeSvg}>
      <polygon points={pts} fill={colors.fill} stroke={colors.stroke} strokeWidth={2} />
    </svg>
  )
}

interface ActivityReportCardProps {
  benefit: HierarchyBenefit
  zones: Record<string, ZoneData> | null
  sessionElapsed: number
  presenceAtStart?: number | Record<string, number>
  onEditBenefit?: (benefitId: string) => void
  onHideCard?: (benefitId: string) => void
  onSyncZones?: () => Promise<void>
}

function ActivityReportCard({
  benefit,
  zones,
  sessionElapsed,
  presenceAtStart = 0,
  onEditBenefit,
  onHideCard,
  onSyncZones,
}: ActivityReportCardProps) {
  const [intervalMinutes, setIntervalMinutes] = useState(2)
  const zoneItems = useMemo(() => getDetectionZoneKeys(benefit, zones), [benefit, zones])
  const polys = benefit.zone_polygons ?? []
  const types = benefit.zone_polygon_types ?? polys.map(() => 'include' as const)
  const includeCount = polys.filter((_, i) => types[i] === 'include').length
  const needsSyncForFormes = includeCount >= 2 && zoneItems.length === 1 && zoneItems[0]?.label === 'Présence' && !!onSyncZones
  const lastAutoSyncRef = useRef<string | null>(null)

  useEffect(() => {
    if (!needsSyncForFormes || !onSyncZones || lastAutoSyncRef.current === benefit.benefit_id) return
    lastAutoSyncRef.current = benefit.benefit_id
    onSyncZones().catch(() => {})
  }, [needsSyncForFormes, onSyncZones, benefit.benefit_id])

  const presenceAtStartMap = typeof presenceAtStart === 'object' ? presenceAtStart : { [benefit.benefit_id]: presenceAtStart }

  const zoneStats = useMemo(() => {
    return zoneItems.map((z, i) => {
      const zone = zones?.[z.zoneKey]
      const totalPresence = zone?.total_time ?? 0
      const start = presenceAtStartMap[z.zoneKey] ?? 0
      const presenceTime = Math.max(0, totalPresence - start)
      const pct = sessionElapsed > 0 ? Math.min(100, (presenceTime / sessionElapsed) * 100) : 0
      const polyIdx = z.zoneKey.includes(':') ? parseInt(z.zoneKey.split(':')[1], 10) : 0
      const safePolyIdx = Number.isFinite(polyIdx) ? polyIdx : 0
      const poly = polys[safePolyIdx] && Array.isArray(polys[safePolyIdx]) ? polys[safePolyIdx] : null
      const polyType = types[safePolyIdx] ?? 'include'
      return {
        ...z,
        presenceTime,
        pct,
        isOccupied: zone?.is_occupied ?? false,
        color: ACTIVITY_ZONE_COLORS[i % ACTIVITY_ZONE_COLORS.length],
        poly: poly as number[][] | null,
        polyType,
      }
    })
  }, [zoneItems, zones, sessionElapsed, presenceAtStartMap, polys, types])

  return (
    <article className={`${styles.card} ${styles.cardWide}`}>
      <div className={styles.cardMain}>
        <div className={styles.cardTop}>
          <div className={styles.cardTopLeft}>
            <span className={styles.cardTitle}>Rapport d&apos;activité</span>
          </div>
          <div className={styles.chartControls}>
            <LovDropdown
              options={ACTIVITY_CHART_INTERVAL_OPTIONS.map((opt) => ({
                value: String(opt.value),
                label: opt.label,
              }))}
              value={String(intervalMinutes)}
              onChange={(v) => setIntervalMinutes(Number(v))}
              placeholder="Laps"
              className={styles.cardLovCompact}
            />
            <CardMenu
              options={[
                ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
                ...(onHideCard ? [{ label: 'Supprimer la carte', onClick: () => onHideCard(benefit.benefit_id) }] : []),
              ]}
            />
          </div>
        </div>
        <div className={styles.activityZoneCards}>
          {zoneStats.map((stat) => (
            <div key={stat.zoneKey} className={styles.activityZoneCard}>
              <div className={styles.activityZoneCardShape}>
                {stat.poly ? (
                  <ZoneShapePreview poly={stat.poly} type={stat.polyType} />
                ) : (
                  <div className={styles.activityZoneCardPlaceholder} style={{ background: stat.color }} />
                )}
              </div>
              <div className={styles.activityZoneCardContent}>
                <div className={styles.activityZoneCardTitle}>{stat.label}</div>
                <div className={styles.activityZoneCardRow}>
                  <span className={styles.activityZoneCardKey}>Temps</span>
                  <span className={styles.activityZoneCardValue}>{formatTimer(stat.presenceTime)}</span>
                </div>
                <div className={styles.activityZoneCardBarBlock}>
                  <div className={styles.activityZoneCardRow}>
                    <span className={styles.activityZoneCardKey}>Présence</span>
                    <span className={styles.activityZoneCardValue} style={{ color: getPctColor(stat.pct) }}>
                      {Math.round(stat.pct)}%
                    </span>
                  </div>
                  <div className={styles.activityZoneCardBarTrack}>
                    <div
                      className={styles.activityZoneCardBarFill}
                      style={{
                        width: `${stat.pct}%`,
                        background: stat.pct >= 100
                          ? 'linear-gradient(90deg, #3b82f6 0%, #4f46e5 35%, #7c3aed 70%, #a855f7 100%)'
                          : getPctColor(stat.pct),
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className={styles.activityChartCompact}>
          <ActivityChart
            zoneItems={zoneItems}
            zones={zones}
            sessionElapsed={sessionElapsed}
            presenceAtStartMap={presenceAtStartMap}
            intervalMinutes={intervalMinutes}
            compact
            className={styles.activityChartCompactWrap}
          />
        </div>
      </div>
    </article>
  )
}

const COUNTING_MODE_OPTIONS = [
  { value: 'simple', label: 'Gradient' },
  { value: 'complex', label: 'MOG2' },
] as const

const PRESENCE_SORT_OPTIONS = [
  { value: 'time', label: 'Par temps' },
  { value: 'pct', label: 'Par %' },
  { value: 'name', label: 'Par nom' },
] as const

function PresenceCard({ benefit, zones, sessionElapsed, presenceAtStart = 0, onEditBenefit, onDeleteBenefit, onHideCard, onRefresh }: PresenceCardProps) {
  const [busy, setBusy] = useState(false)
  const [sortBy, setSortBy] = useState<'time' | 'pct' | 'name'>('time')
  const presenceRowsRaw = getPresenceRows(benefit, zones, sessionElapsed, presenceAtStart)
  const presenceRows = useMemo(() => {
    const rows = [...presenceRowsRaw]
    if (sortBy === 'time') rows.sort((a, b) => b.presenceTime - a.presenceTime)
    else if (sortBy === 'pct') rows.sort((a, b) => b.pct - a.pct)
    else rows.sort((a, b) => a.label.localeCompare(b.label))
    return rows
  }, [presenceRowsRaw, sortBy])

  const handleReset = async () => {
    if (busy) return
    setBusy(true)
    try {
      const zoneKeys = getDetectionZoneKeys(benefit, zones)
      for (const { zoneKey } of zoneKeys) {
        await resetZoneTimer(zoneKey)
      }
      onRefresh?.()
    } catch (err) {
      console.warn('Reset détection:', err)
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className={styles.card}>
      <div className={styles.cardMain}>
        <div className={styles.cardTop}>
          <div className={styles.cardTopLeft}>
            <span className={styles.dot} style={{ background: getBenefitColor(benefit.benefit_id) }} />
            <span className={styles.cardTitle}>{benefit.name || 'DÉTECTION PRÉSENCE'}</span>
          </div>
          <div className={styles.chartControls}>
            <LovDropdown
              options={PRESENCE_SORT_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
              value={sortBy}
              onChange={(v) => setSortBy(v as 'time' | 'pct' | 'name')}
              placeholder="Tri"
              className={styles.cardLovCompact}
            />
            <CardMenu
            options={[
              ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
              ...(presenceRows.length > 0 ? [{ label: 'Reset', onClick: handleReset }] : []),
              ...(onHideCard ? [{ label: 'Supprimer la carte', onClick: () => onHideCard(benefit.benefit_id) }] : []),
              ...(onDeleteBenefit ? [{ label: 'Supprimer le bénéfice', danger: true, onClick: () => confirm('Supprimer ce bénéfice ?') && onDeleteBenefit(benefit.benefit_id) }] : []),
              { label: 'Exporter', onClick: () => console.log('Exporter détection') },
            ]}
            />
          </div>
        </div>
        <div className={styles.presenceRows}>
          {presenceRows.map((row) => (
            <div key={row.key} className={styles.presenceRow}>
              <div className={styles.presenceRowHeader}>
                <div className={styles.presenceLabel}>
                  <img src={row.icon} className={styles.presenceIco} alt="" />
                  <span>{row.label}</span>
                </div>
                <span className={`${styles.presenceBadge} ${row.tracked && row.isOccupied ? styles.presenceBadgeOn : ''}`}>
                  {row.tracked ? (row.isOccupied ? 'Présent' : 'Absent') : '—'}
                </span>
              </div>
              <div className={styles.presenceBar}>
                {row.tracked ? (
                  <div className={`${styles.presenceFill} ${row.isOccupied ? styles.presenceFillActive : ''}`} style={{ width: `${row.pct}%` }} />
                ) : (
                  <div className={styles.presenceFillUntracked} title="Modèle de détection non disponible pour cette classe" />
                )}
              </div>
              <div className={styles.presenceStats}>
                <span className={styles.presenceTime}>{row.tracked ? formatTimer(row.presenceTime) : '—'}</span>
                <span className={styles.presencePct}>{row.tracked ? `${row.pct}%` : '—'}</span>
              </div>
            </div>
          ))}
          {presenceRows.length === 0 && <div className={styles.emptyRow}>Aucune classe sélectionnée</div>}
        </div>
      </div>
    </article>
  )
}

interface CountingCardProps {
  benefit: HierarchyBenefit
  videoPath: string | null
  zones: Record<string, ZoneData> | null
  counting: CountingResponse | null
  countingBenefits: HierarchyBenefit[]
  onRefresh?: () => void
  onEditBenefit?: (benefitId: string) => void
  onDeleteBenefit?: (benefitId: string) => void
  onHideCard?: (benefitId: string) => void
}

function CountingCard({
  benefit,
  videoPath,
  zones,
  counting,
  countingBenefits,
  onRefresh,
  onEditBenefit,
  onDeleteBenefit,
  onHideCard,
}: CountingCardProps) {
  const [paramsOpen, setParamsOpen] = useState(false)
  const [threshold, setThreshold] = useState(30)
  const [cooldown, setCooldown] = useState(15)
  const [busy, setBusy] = useState(false)

  const zoneOptions = getCountingZoneOptions(countingBenefits, zones)
  const selectedZone = counting?.zone_name ?? ''
  const mode = counting?.mode ?? 'simple'
  const enabled = counting?.enabled ?? false
  const count = counting?.count ?? 0
  const angle = counting?.angle

  const refresh = useCallback(() => {
    onRefresh?.()
  }, [onRefresh])

  useEffect(() => {
    fetchCountingParams()
      .then((p) => {
        setThreshold(p.simple_gradient_threshold ?? 30)
        setCooldown(p.simple_cooldown_frames ?? 15)
      })
      .catch(() => {})
  }, [paramsOpen])

  const handleZoneChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const zone = e.target.value
    if (!videoPath || !zone) return
    setBusy(true)
    try {
      await configCounting(videoPath, { zone_name: zone, mode })
      refresh()
    } catch (err) {
      console.warn('Config counting:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleModeChange = async (newMode: 'simple' | 'complex') => {
    if (!videoPath || !selectedZone) return
    setBusy(true)
    try {
      await configCounting(videoPath, { zone_name: selectedZone, mode: newMode })
      refresh()
    } catch (err) {
      console.warn('Config counting:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleToggle = async () => {
    if (!videoPath) return
    setBusy(true)
    try {
      await toggleCounting(videoPath)
      refresh()
    } catch (err) {
      console.warn('Toggle counting:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleFlip = async () => {
    if (!videoPath) return
    setBusy(true)
    try {
      await flipCounting(videoPath)
      refresh()
    } catch (err) {
      console.warn('Flip counting:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleReset = async () => {
    if (!videoPath || busy) return
    setBusy(true)
    try {
      await resetCounting(videoPath)
      refresh()
    } catch (err) {
      console.warn('Reset counting:', err)
    } finally {
      setBusy(false)
    }
  }

  const handleSaveParams = async () => {
    setBusy(true)
    try {
      await updateCountingParams({ simple_gradient_threshold: threshold, simple_cooldown_frames: cooldown })
    } catch (err) {
      console.warn('Save params:', err)
    } finally {
      setBusy(false)
    }
  }

  const baseBenefitId = selectedZone.includes(':') ? selectedZone.split(':')[0] : selectedZone
  const primaryCountingBenefit = countingBenefits.find((b) => b.benefit_id === baseBenefitId) ?? benefit
  const countingItems = getCountingItems(primaryCountingBenefit, count)
  const countMode = mode === 'complex' ? 'MOG2' : 'Gradient'

  return (
    <article className={styles.card}>
      <div className={styles.cardMain}>
        <div className={styles.cardTop}>
          <div className={styles.cardTopLeft}>
            <span
              className={styles.dot}
              style={{ background: primaryCountingBenefit ? getBenefitColor(primaryCountingBenefit.benefit_id) : 'rgba(15,23,42,0.25)' }}
            />
            <span className={styles.cardTitle}>{benefit.name || 'COMPTAGE ZONE'}</span>
          </div>
          <div className={styles.chartControls}>
            <LovDropdown
              options={COUNTING_MODE_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
              value={mode}
              onChange={(v) => handleModeChange(v as 'simple' | 'complex')}
              placeholder="Mode"
              className={styles.cardLovCompact}
              disabled={!selectedZone || busy}
            />
            <CardMenu
            options={[
              ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
              ...(selectedZone ? [{ label: 'Reset', onClick: handleReset }] : []),
              ...(onHideCard ? [{ label: 'Supprimer la carte', onClick: () => onHideCard(benefit.benefit_id) }] : []),
              ...(onDeleteBenefit ? [{ label: 'Supprimer le bénéfice', danger: true, onClick: () => confirm('Supprimer ce bénéfice ?') && onDeleteBenefit(benefit.benefit_id) }] : []),
              { label: 'Exporter', onClick: () => console.log('Exporter comptage') },
            ]}
            />
          </div>
        </div>
        <div className={styles.countingControls}>
          <div className={styles.countingRow}>
            <span className={styles.countingLabel}>Zone</span>
            <select
              className={styles.countingSelect}
              value={selectedZone}
              onChange={handleZoneChange}
              disabled={!videoPath || busy}
            >
              <option value="">{zoneOptions.length ? 'Choisir' : 'Créer zone'}</option>
              {zoneOptions.map((z) => (
                <option key={z.id} value={z.id}>{z.label}</option>
              ))}
            </select>
          </div>
          <div className={styles.counters}>
            {countingItems.map((it) => (
              <div key={it.label} className={styles.counterWrap}>
                <img src={it.icon} className={styles.counterIcon} alt="" />
                <span className={styles.counterValue}>{it.count}</span>
                <span className={styles.counterLabel}>{it.label}</span>
              </div>
            ))}
          </div>
          {angle != null && (
            <div className={styles.counterSub}>Direction : {angle}°</div>
          )}
          <div className={styles.counterSub}>{countMode}</div>

          <div className={styles.countingActions}>
            <button
              type="button"
              className={`${styles.countingBtn} ${enabled ? styles.countingBtnDanger : styles.countingBtnPrimary}`}
              onClick={handleToggle}
              disabled={!selectedZone || busy}
            >
              {enabled ? '⏹ Stop' : '▶ Démarrer'}
            </button>
            <button
              type="button"
              className={styles.countingBtn}
              onClick={handleFlip}
              disabled={!selectedZone || busy}
              title="Tourner la direction de 90°"
            >
              ↻ Flip
            </button>
          </div>

          <div className={styles.countingParams}>
            <button
              type="button"
              className={styles.countingParamsToggle}
              onClick={() => setParamsOpen((p) => !p)}
            >
              {paramsOpen ? '▼' : '▶'} Paramètres comptage
            </button>
            {paramsOpen && (
              <div className={styles.countingParamsContent}>
                <div className={styles.countingParamsRow}>
                  <label>Seuil gradient</label>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                  />
                  <span>{threshold}</span>
                </div>
                <div className={styles.countingParamsRow}>
                  <label>Cooldown frames</label>
                  <input
                    type="range"
                    min={1}
                    max={60}
                    value={cooldown}
                    onChange={(e) => setCooldown(Number(e.target.value))}
                  />
                  <span>{cooldown}</span>
                </div>
                <button type="button" className={styles.countingBtn} onClick={handleSaveParams} disabled={busy}>
                  Enregistrer
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </article>
  )
}

function getBenefitColor(benefitId: string): string {
  const colors = ['#5a8fb8', '#4d9d8a', '#8b7fb5', '#c47b5a']
  let h = 0
  for (let i = 0; i < benefitId.length; i++) h = ((h << 5) - h) + benefitId.charCodeAt(i)
  return colors[Math.abs(h) % colors.length]
}

const HIDDEN_CARDS_KEY = 'trigodemo-dataRoom-hiddenCards'

function loadHiddenCards(): Set<string> {
  try {
    const raw = localStorage.getItem(HIDDEN_CARDS_KEY)
    if (raw) {
      const arr = JSON.parse(raw) as string[]
      return new Set(Array.isArray(arr) ? arr : [])
    }
  } catch {
    /* ignore */
  }
  return new Set()
}

function saveHiddenCards(set: Set<string>) {
  localStorage.setItem(HIDDEN_CARDS_KEY, JSON.stringify([...set]))
}

interface DataRoomCardsProps {
  benefits: HierarchyBenefit[]
  zones: Record<string, ZoneData> | null
  counting: CountingResponse | null
  videoPath: string | null
  benefitElapsed?: Record<string, number>
  siteId?: string
  camId?: string
  presenceAtStart?: number | Record<string, number>
  onAddBenefit?: () => void
  onEditBenefit?: (benefitId: string) => void
  onDeleteBenefit?: (benefitId: string) => void
  onRefreshCounting?: () => void
  onRefreshDetection?: () => void
  onSyncZones?: () => Promise<void>
}

export default function DataRoomCards({
  benefits,
  zones,
  counting,
  videoPath,
  benefitElapsed = {},
  siteId = '',
  camId = '',
  presenceAtStart = {},
  onAddBenefit,
  onEditBenefit,
  onDeleteBenefit,
  onRefreshCounting,
  onRefreshDetection,
  onSyncZones,
}: DataRoomCardsProps) {
  const [hiddenCards, setHiddenCards] = useState<Set<string>>(loadHiddenCards)

  useEffect(() => {
    const benefitIds = new Set(benefits.map((b) => b.benefit_id))
    setHiddenCards((prev) => {
      const toRemove = [...prev].filter((id) => !benefitIds.has(id))
      if (toRemove.length === 0) return prev
      const next = new Set(prev)
      toRemove.forEach((id) => next.delete(id))
      saveHiddenCards(next)
      return next
    })
  }, [benefits])

  const handleHideCard = useCallback((benefitId: string) => {
    setHiddenCards((prev) => {
      const next = new Set(prev)
      next.add(benefitId)
      saveHiddenCards(next)
      return next
    })
  }, [])

  const detectionBenefits = benefits.filter(
    (b) =>
      String(b.skill || '').toLowerCase() === 'detection' &&
      (b.skill_item === 'detection_presence' || /présence|absence|presence/i.test(b.skill_item || b.name || '')) &&
      !hiddenCards.has(b.benefit_id)
  )
  const countingBenefits = benefits.filter(
    (b) =>
      String(b.skill || '').toLowerCase() === 'counting' &&
      (b.zone_polygons?.length ?? 0) > 0 &&
      !hiddenCards.has(b.benefit_id)
  )

  return (
    <div className={styles.block}>
      <div className={styles.header}>
        <h3 className={styles.title}>DATA ROOM</h3>
        <p className={styles.subtitle}>Bénéfices et mesures</p>
      </div>
      <div className={styles.grid}>
        {detectionBenefits.length > 0 && (
          <ActivityReportCard
            key={`report-${detectionBenefits[0].benefit_id}`}
            benefit={detectionBenefits[0]}
            zones={zones}
            sessionElapsed={benefitElapsed[benefitElapsedKey(siteId, camId, detectionBenefits[0].benefit_id)] ?? 0}
            presenceAtStart={presenceAtStart}
            onEditBenefit={onEditBenefit}
            onHideCard={handleHideCard}
            onSyncZones={onSyncZones}
          />
        )}
        {detectionBenefits.map((benefit) => (
          <PresenceCard
            key={benefit.benefit_id}
            benefit={benefit}
            zones={zones}
            sessionElapsed={benefitElapsed[benefitElapsedKey(siteId, camId, benefit.benefit_id)] ?? 0}
            presenceAtStart={presenceAtStart}
            onEditBenefit={onEditBenefit}
            onDeleteBenefit={onDeleteBenefit}
            onHideCard={handleHideCard}
            onRefresh={onRefreshDetection}
          />
        ))}
        {detectionBenefits.length === 0 && onAddBenefit && (
          <article className={styles.card}>
            <div className={styles.cardMain}>
              <div className={styles.cardTop}>
                <span className={styles.dot} style={{ background: 'rgba(15,23,42,0.25)' }} />
                <span className={styles.cardTitle}>DÉTECTION PRÉSENCE</span>
              </div>
              <div className={styles.emptyRow}>
                <button type="button" className={styles.countingBtn} onClick={onAddBenefit}>
                  + Créer un bénéfice Détection présence
                </button>
              </div>
            </div>
          </article>
        )}

        {countingBenefits.map((benefit) => (
          <CountingCard
            key={benefit.benefit_id}
            benefit={benefit}
            videoPath={videoPath}
            zones={zones}
            counting={counting}
            countingBenefits={countingBenefits}
            onRefresh={onRefreshCounting}
            onEditBenefit={onEditBenefit}
            onDeleteBenefit={onDeleteBenefit}
            onHideCard={handleHideCard}
          />
        ))}
        {countingBenefits.length === 0 && onAddBenefit && (
          <article className={styles.card}>
            <div className={styles.cardMain}>
              <div className={styles.cardTop}>
                <span className={styles.dot} style={{ background: 'rgba(15,23,42,0.25)' }} />
                <span className={styles.cardTitle}>COMPTAGE ZONE</span>
              </div>
              <div className={styles.emptyRow}>
                <button type="button" className={styles.countingBtn} onClick={onAddBenefit}>
                  + Créer un bénéfice Comptage zone
                </button>
              </div>
            </div>
          </article>
        )}
      </div>
    </div>
  )
}
