import { useState, useEffect, useCallback } from 'react'
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
import CardMenu from '@/components/ui/CardMenu'
import { benefitElapsedKey } from '@/context/SessionContext'
import styles from './DataRoomCards.module.css'

const CAT_ICONS: Record<string, string> = {
  human: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
  voiture: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg',
  velo: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg',
}

const CAT_LABELS: Record<string, string> = {
  human: 'Humain',
  voiture: 'Voiture',
  velo: 'Vélo',
}

const SKILL_ICONS: Record<string, string> = {
  detection: '/static/assets_youn/SvIcons/SVGnew/Yclassify.svg',
  counting: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg',
}

const TRACKED_CATEGORIES = new Set(['human'])

function parseCategories(cats: string[] | undefined): { key: string; category: string; subcategory: string }[] {
  if (!cats?.length) return []
  return cats.map((c) => {
    const [cat, sub] = (c || '').split('::')
    return { key: c, category: cat || '', subcategory: sub || '' }
  })
}

function formatTimer(totalSeconds: number): string {
  const n = Math.max(0, Math.floor(totalSeconds))
  const h = Math.floor(n / 3600)
  const m = Math.floor((n % 3600) / 60)
  const s = n % 60
  return [h, m, s].map((v) => v.toString().padStart(2, '0')).join(':')
}

interface PresenceRow {
  label: string
  icon: string
  isOccupied: boolean
  presenceTime: number
  pct: number
}

function getPresenceRows(
  benefit: HierarchyBenefit | null,
  zones: Record<string, ZoneData> | null,
  sessionElapsed: number,
  presenceAtStart: number,
): PresenceRow[] {
  if (!benefit) return []
  const zone = zones?.[benefit.benefit_id]
  const isOccupied = zone?.is_occupied ?? false
  const totalPresence = zone?.total_time ?? 0
  const presenceTime = Math.max(0, totalPresence - presenceAtStart)
  const pct = sessionElapsed > 0 ? Math.min(100, Math.round((presenceTime / sessionElapsed) * 100)) : 0

  const cats = parseCategories(benefit.categories)
  const tracked = cats.filter((c) => TRACKED_CATEGORIES.has(c.category))
  const rows = tracked.length > 0 ? tracked : cats.slice(0, 1)

  if (rows.length === 0) {
    return [{
      label: 'Présence',
      icon: CAT_ICONS.human,
      isOccupied,
      presenceTime,
      pct,
    }]
  }

  return rows.map((c) => {
    const iconKey = c.subcategory || c.category
    const isTracked = TRACKED_CATEGORIES.has(c.category)
    return {
      label: CAT_LABELS[iconKey] ?? CAT_LABELS[c.category] ?? c.category,
      icon: CAT_ICONS[iconKey] ?? CAT_ICONS[c.category] ?? CAT_ICONS.human,
      isOccupied: isTracked ? isOccupied : false,
      presenceTime: isTracked ? presenceTime : 0,
      pct: isTracked ? pct : 0,
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
    if (matchingZones.length === 0) {
      opts.push({ id: b.benefit_id, label: name })
    } else {
      for (const z of matchingZones.sort()) {
        if (z === b.benefit_id) {
          opts.push({ id: z, label: types.length > 1 ? `${name} (toutes zones)` : name })
        } else {
          const idx = z.split(':')[1]
          opts.push({ id: z, label: `${name} (zone include ${Number(idx) + 1})` })
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
  presenceAtStart: number
  onEditBenefit?: (benefitId: string) => void
  onDeleteBenefit?: (benefitId: string) => void
}

function PresenceCard({ benefit, zones, sessionElapsed, presenceAtStart, onEditBenefit, onDeleteBenefit }: PresenceCardProps) {
  const presenceRows = getPresenceRows(benefit, zones, sessionElapsed, presenceAtStart)
  return (
    <article className={styles.card}>
      <div className={styles.cardMain}>
        <div className={styles.cardTop}>
          <div className={styles.cardTopLeft}>
            <span className={styles.dot} style={{ background: getBenefitColor(benefit.benefit_id) }} />
            <span className={styles.cardTitle}>{benefit.name || 'DÉTECTION PRÉSENCE'}</span>
          </div>
          <CardMenu
            options={[
              ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
              ...(onDeleteBenefit ? [{ label: 'Supprimer le bénéfice', danger: true, onClick: () => confirm('Supprimer ce bénéfice ?') && onDeleteBenefit(benefit.benefit_id) }] : []),
              { label: 'Exporter', onClick: () => console.log('Exporter détection') },
            ]}
          />
        </div>
        <div className={styles.chips}>
          <span className={styles.chip}>
            <img src={SKILL_ICONS.detection} className={styles.chipIcon} alt="" />
            Présence / Absence
          </span>
        </div>
        <div className={styles.presenceRows}>
          {presenceRows.map((row) => (
            <div key={row.label} className={styles.presenceRow}>
              <div className={styles.presenceRowHeader}>
                <div className={styles.presenceLabel}>
                  <img src={row.icon} className={styles.presenceIco} alt="" />
                  <span>{row.label}</span>
                </div>
                <span className={`${styles.presenceBadge} ${row.isOccupied ? styles.presenceBadgeOn : ''}`}>
                  {row.isOccupied ? 'Présent' : 'Absent'}
                </span>
              </div>
              <div className={styles.presenceBar}>
                <div className={`${styles.presenceFill} ${row.isOccupied ? styles.presenceFillActive : ''}`} style={{ width: `${row.pct}%` }} />
              </div>
              <div className={styles.presenceStats}>
                <span className={styles.presenceTime}>{formatTimer(row.presenceTime)}</span>
                <span className={styles.presencePct}>{row.pct}%</span>
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

  const handleModeChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newMode = e.target.value as 'simple' | 'complex'
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
    if (!videoPath) return
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
          <CardMenu
            options={[
              ...(onEditBenefit ? [{ label: 'Modifier', onClick: () => onEditBenefit(benefit.benefit_id) }] : []),
              ...(onDeleteBenefit ? [{ label: 'Supprimer le bénéfice', danger: true, onClick: () => confirm('Supprimer ce bénéfice ?') && onDeleteBenefit(benefit.benefit_id) }] : []),
              { label: 'Exporter', onClick: () => console.log('Exporter comptage') },
            ]}
          />
        </div>
        <div className={styles.chips}>
          <span className={styles.chip}>
            <img src={SKILL_ICONS.counting} className={styles.chipIcon} alt="" />
            Comptage
          </span>
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
              <option value="">{zoneOptions.length ? '— choisir —' : 'Créer un bénéfice Comptage zone et dessiner une zone'}</option>
              {zoneOptions.map((z) => (
                <option key={z.id} value={z.id}>{z.label}</option>
              ))}
            </select>
          </div>
          <div className={styles.countingRow}>
            <span className={styles.countingLabel}>Mode</span>
            <select
              className={styles.countingSelect}
              value={mode}
              onChange={handleModeChange}
              disabled={!selectedZone || busy}
            >
              <option value="simple">Simple (gradient)</option>
              <option value="complex">Complex (MOG2)</option>
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
            <button
              type="button"
              className={styles.countingBtn}
              onClick={handleReset}
              disabled={!selectedZone || busy}
            >
              ↺ Reset
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

interface DataRoomCardsProps {
  benefits: HierarchyBenefit[]
  zones: Record<string, ZoneData> | null
  counting: CountingResponse | null
  videoPath: string | null
  benefitElapsed?: Record<string, number>
  siteId?: string
  camId?: string
  presenceAtStart?: number
  onAddBenefit?: () => void
  onEditBenefit?: (benefitId: string) => void
  onDeleteBenefit?: (benefitId: string) => void
  onRefreshCounting?: () => void
}

export default function DataRoomCards({
  benefits,
  zones,
  counting,
  videoPath,
  benefitElapsed = {},
  siteId = '',
  camId = '',
  presenceAtStart = 0,
  onAddBenefit,
  onEditBenefit,
  onDeleteBenefit,
  onRefreshCounting,
}: DataRoomCardsProps) {
  const detectionBenefits = benefits.filter(
    (b) =>
      String(b.skill || '').toLowerCase() === 'detection' &&
      (b.skill_item === 'detection_presence' || /présence|absence|presence/i.test(b.skill_item || b.name || ''))
  )
  const countingBenefits = benefits.filter(
    (b) => String(b.skill || '').toLowerCase() === 'counting' && (b.zone_polygons?.length ?? 0) > 0
  )

  return (
    <div className={styles.block}>
      <div className={styles.header}>
        <h3 className={styles.title}>DATA ROOM</h3>
        <p className={styles.subtitle}>Bénéfices et mesures</p>
      </div>
      <div className={styles.grid}>
        {detectionBenefits.map((benefit) => (
          <PresenceCard
            key={benefit.benefit_id}
            benefit={benefit}
            zones={zones}
            sessionElapsed={benefitElapsed[benefitElapsedKey(siteId, camId, benefit.benefit_id)] ?? 0}
            presenceAtStart={presenceAtStart}
            onEditBenefit={onEditBenefit}
            onDeleteBenefit={onDeleteBenefit}
          />
        ))}
        {detectionBenefits.length === 0 && onAddBenefit && (
          <article className={styles.card}>
            <div className={styles.cardMain}>
              <div className={styles.cardTop}>
                <span className={styles.dot} style={{ background: 'rgba(15,23,42,0.25)' }} />
                <span className={styles.cardTitle}>DÉTECTION PRÉSENCE</span>
              </div>
              <div className={styles.chips}>
                <span className={styles.chip}>
                  <img src={SKILL_ICONS.detection} className={styles.chipIcon} alt="" />
                  Présence / Absence
                </span>
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
          />
        ))}
        {countingBenefits.length === 0 && onAddBenefit && (
          <article className={styles.card}>
            <div className={styles.cardMain}>
              <div className={styles.cardTop}>
                <span className={styles.dot} style={{ background: 'rgba(15,23,42,0.25)' }} />
                <span className={styles.cardTitle}>COMPTAGE ZONE</span>
              </div>
              <div className={styles.chips}>
                <span className={styles.chip}>
                  <img src={SKILL_ICONS.counting} className={styles.chipIcon} alt="" />
                  Comptage
                </span>
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
