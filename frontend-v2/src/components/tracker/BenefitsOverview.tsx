import { useState, useEffect } from 'react'
import type { HierarchyBenefit } from '@/types/hierarchy'
import type { ZoneData } from '@/api/tracker'
import { Toggle } from '@/components/ui/Toggle'
import { getSkillGroups, getCategoryGroupsBySkill } from '@/api/skills'
import styles from './BenefitsOverview.module.css'

function formatTime(seconds: number): string {
  const n = Math.max(0, Number(seconds) || 0)
  const h = Math.floor(n / 3600)
  const m = Math.floor((n % 3600) / 60)
  const s = Math.floor(n % 60)
  return [h, m, s].map((v) => v.toString().padStart(2, '0')).join(':')
}

const FALLBACK_ICON = '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg'

function getSkillLabelAndIcon(skill?: string): { label: string; icon: string } {
  const g = getSkillGroups().find((s) => s.key === skill)
  return { label: g?.label ?? skill ?? '—', icon: g?.icon ?? FALLBACK_ICON }
}

function getSkillItemLabel(skill?: string, skillItem?: string): string {
  const g = getSkillGroups().find((s) => s.key === skill)
  const item = g?.items?.find((i) => i.id === skillItem) ?? g?.items?.[0]
  return item?.label ?? skillItem ?? '—'
}

function getCategoryIcons(skill: string, categories: string[]): { key: string; icon: string }[] {
  const groups = getCategoryGroupsBySkill(skill)
  const map: Record<string, string> = {}
  groups.forEach((grp) => (grp.items ?? []).forEach((it) => {
    map[`${grp.key}::${it.id}`] = it.icon || grp.icon
  }))
  return (categories ?? [])
    .filter((k) => map[k])
    .map((key) => ({ key, icon: map[key] }))
}

const ACCENT_PALETTE = ['mint', 'sky', 'violet', 'peach', 'rose', 'sand', 'teal', 'slate', 'lime', 'coral', 'azure', 'amber'] as const

function getAccentFromKey(key: string): (typeof ACCENT_PALETTE)[number] {
  const idx = Math.abs([...key].reduce((h, c) => ((h << 5) - h) + c.charCodeAt(0), 0)) % ACCENT_PALETTE.length
  return ACCENT_PALETTE[idx]
}

const BENEFIT_COLORS = ['#5a8fb8', '#4d9d8a', '#8b7fb5', '#c47b5a', '#6b9b7a', '#c4a055']

function getBenefitColor(benefitId: string): string {
  let h = 0
  for (let i = 0; i < benefitId.length; i++) h = ((h << 5) - h) + benefitId.charCodeAt(i)
  return BENEFIT_COLORS[Math.abs(h) % BENEFIT_COLORS.length]
}

interface BenefitsOverviewProps {
  benefits: HierarchyBenefit[]
  zones: Record<string, ZoneData> | null
  sessionElapsed?: number
  selectedBenefitId: string | null
  onSelectBenefit: (benId: string | null) => void
  onToggleBenefit: (benId: string, active: boolean) => Promise<void>
  onEditBenefit: (benId: string) => void
}

export default function BenefitsOverview({
  benefits,
  zones: _zones,
  sessionElapsed = 0,
  selectedBenefitId,
  onSelectBenefit,
  onToggleBenefit,
  onEditBenefit,
}: BenefitsOverviewProps) {
  /* Optimistic update: évite le re-render immédiat qui coupe l'animation goo (comme vanilla) */
  const [optimisticActive, setOptimisticActive] = useState<Record<string, boolean>>({})

  useEffect(() => {
    setOptimisticActive((prev) => {
      const next = { ...prev }
      benefits.forEach((b) => {
        const v = b.active !== false
        if (b.benefit_id in next && next[b.benefit_id] === v) delete next[b.benefit_id]
      })
      return next
    })
  }, [benefits])

  if (benefits.length === 0) {
    return (
      <div className={styles.empty}>
        Aucun bénéfice pour cette caméra.
      </div>
    )
  }

  return (
    <div className={styles.list}>
      {benefits.map((ben) => {
        const timerText = formatTime(sessionElapsed)
        const enabled =
          ben.benefit_id in optimisticActive ? optimisticActive[ben.benefit_id] : ben.active !== false
        const isSelected = selectedBenefitId === ben.benefit_id

        return (
          <article
            key={ben.benefit_id}
            className={`${styles.row} ${isSelected ? styles.selected : ''}`}
            onClick={() => onSelectBenefit(isSelected ? null : ben.benefit_id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelectBenefit(isSelected ? null : ben.benefit_id)
              }
            }}
          >
            <div className={styles.skill}>
              <img
                src={getSkillLabelAndIcon(ben.skill).icon}
                alt=""
                className={styles.skillIcon}
                onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }}
              />
            </div>
            <div className={styles.main}>
              <div className={styles.header}>
                <span
                  className={styles.dot}
                  style={{ background: getBenefitColor(ben.benefit_id) }}
                  title={ben.name}
                />
                <span className={styles.title} title={ben.name}>
                  {(ben.name || ben.benefit_id).toUpperCase()}
                </span>
              </div>
              <div className={styles.chipsRow}>
                <span className={`${styles.skillChip} ${styles[`skillChip--${ben.skill ?? 'detection'}`] ?? styles['skillChip--detection']}`}>
                  {getSkillItemLabel(ben.skill, ben.skill_item)}
                </span>
                {getCategoryIcons(ben.skill ?? 'detection', ben.categories ?? []).map(({ key, icon }) => {
                  const accent = getAccentFromKey(key)
                  return (
                  <span
                    key={key}
                    className={`${styles.catChip} ${isSelected ? styles[`catChipAccent${accent.charAt(0).toUpperCase() + accent.slice(1)}`] : ''}`}
                    title={key}
                  >
                    <img src={icon} alt="" className={styles.chipIcon} onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />
                  </span>
                  )
                })}
              </div>
              <div className={styles.footer}>
                <span className={styles.timer}>{timerText}</span>
                <div className={styles.toggleWrap} onClick={(e) => e.stopPropagation()}>
                  <Toggle
                    checked={enabled}
                    variant="primary"
                    title={enabled ? 'Désactiver' : 'Activer'}
                    onCheckedChange={(next) => {
                      setOptimisticActive((prev) => ({ ...prev, [ben.benefit_id]: next }))
                      onToggleBenefit(ben.benefit_id, next).catch(() => {
                        setOptimisticActive((prev) => {
                          const n = { ...prev }
                          delete n[ben.benefit_id]
                          return n
                        })
                      })
                    }}
                  />
                </div>
                <button
                  type="button"
                  className={styles.editBtn}
                  onClick={(e) => {
                    e.stopPropagation()
                    onEditBenefit(ben.benefit_id)
                  }}
                >
                  Modifier
                </button>
              </div>
            </div>
          </article>
        )
      })}
    </div>
  )
}
