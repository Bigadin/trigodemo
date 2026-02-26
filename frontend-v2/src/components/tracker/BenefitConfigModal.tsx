import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { createPortal } from 'react-dom'
import type { HierarchyBenefit } from '@/types/hierarchy'
import {
  loadSkillsConfig,
  getSkillGroups,
  getCategoryGroupsBySkill,
} from '@/api/skills'
import { createBenefit, updateBenefit, deleteBenefit } from '@/api/benefits'
import { getFrameUrl, fetchVideoInfo } from '@/api/tracker'
import { optimisticMutation } from '@/api/request'
import { useHierarchy } from '@/context/HierarchyContext'
import { useBenefitEditorCanvas } from '@/hooks/useBenefitEditorCanvas'
import type { Polygon } from '@/hooks/useBenefitEditorCanvas'
import CardMenu from '@/components/ui/CardMenu'
import '@/styles/benefit-panel.css'

const ICON_SELECT = '/static/assets_youn/YrysUIPackage/selection.svg'
const ICON_INCLUDE = '/static/assets_youn/YrysUIPackage/drawzone.svg'
const ICON_EXCLUDE = '/static/assets_youn/SvIcons/intersect-svgrepo-com%20(1).svg'

const STEPS = ['skill', 'category', 'info'] as const
type StepId = (typeof STEPS)[number]

const FALLBACK_ICON = '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg'

function toLocalInputDateTime(): string {
  const d = new Date()
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function getDefaultBenefitName(camName: string): string {
  return `Nouveau bénéfice - ${camName || 'Caméra'}`
}


export interface BenefitConfigModalProps {
  open: boolean
  onClose: () => void
  cameraId: string
  cameraName: string
  videoPath: string
  benefit?: HierarchyBenefit | null
  /** Appelé après sauvegarde. En création, reçoit le benefit_id du nouveau bénéfice. */
  onSaved: (createdBenefitId?: string) => void
  /** Appelé si l'enregistrement en arrière-plan échoue */
  onSaveError?: (message: string) => void
}

export default function BenefitConfigModal({
  open,
  onClose,
  cameraId,
  cameraName,
  videoPath,
  benefit,
  onSaved,
  onSaveError,
}: BenefitConfigModalProps) {
  const isEdit = !!benefit
  const { optimisticMergeBenefit, refetch } = useHierarchy()

  const [skillsReady, setSkillsReady] = useState(false)
  const [step, setStep] = useState<StepId>(isEdit ? 'info' : 'skill')
  const [skill, setSkill] = useState(benefit?.skill ?? 'detection')
  const [skillItem, setSkillItem] = useState(benefit?.skill_item ?? 'detection_presence')
  const [selectedCategories, setSelectedCategories] = useState<string[]>(
    benefit?.categories ?? ['human::silhouette']
  )
  const [name, setName] = useState(benefit?.name ?? getDefaultBenefitName(cameraName))
  const [createdBy, setCreatedBy] = useState('Opérateur')
  const [createdAt, setCreatedAt] = useState(toLocalInputDateTime())
  const [comment, setComment] = useState('')
  const [scheduleEnabled, setScheduleEnabled] = useState(false)
  const [scheduleStart, setScheduleStart] = useState('')
  const [scheduleEnd, setScheduleEnd] = useState('')
  const [enabled, setEnabled] = useState(benefit?.active !== false)
  const [categoryOpen, setCategoryOpen] = useState<Record<string, boolean>>({})
  const [error, setError] = useState<string | null>(null)
  const [videoInfo, setVideoInfo] = useState<{ width: number; height: number } | null>(null)
  const [editorTool, setEditorTool] = useState<'select' | 'include' | 'exclude'>('select')
  const [polygonIdx, setPolygonIdx] = useState<number | null>(null)
  const [localZonePolygons, setLocalZonePolygons] = useState<number[][][]>([])
  const [localZonePolygonTypes, setLocalZonePolygonTypes] = useState<('include' | 'exclude')[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const initialZonePolygonsRef = useRef<number[][][]>([])

  useEffect(() => {
    if (open) {
      setSkillsReady(false)
      loadSkillsConfig().then(() => setSkillsReady(true))
    }
  }, [open])

  useEffect(() => {
    if (open) {
      const initial = (benefit?.zone_polygons ?? []).length > 0 ? JSON.parse(JSON.stringify(benefit!.zone_polygons!)) : []
      initialZonePolygonsRef.current = initial
    }
  }, [open, benefit?.zone_polygons])

  const zonesModified = useMemo(() => {
    return JSON.stringify(localZonePolygons) !== JSON.stringify(initialZonePolygonsRef.current)
  }, [localZonePolygons])

  useEffect(() => {
    if (open && videoPath) {
      fetchVideoInfo(videoPath)
        .then((info) => setVideoInfo({ width: info.width, height: info.height }))
        .catch(() => setVideoInfo(null))
    } else {
      setVideoInfo(null)
    }
  }, [open, videoPath])


  useEffect(() => {
    if (open) {
      const polys = benefit?.zone_polygons ?? []
      setLocalZonePolygons(polys.length > 0 ? [...polys] : [])
      const types = benefit?.zone_polygon_types
      setLocalZonePolygonTypes(
        types && types.length >= polys.length
          ? types.slice(0, polys.length)
          : polys.map(() => 'include' as const)
      )
    }
  }, [open, benefit?.zone_polygons, benefit?.zone_polygon_types])

  useEffect(() => {
    if (open && benefit) {
      setStep('info')
      setName(benefit.name || getDefaultBenefitName(cameraName))
      setSkill(benefit.skill ?? 'detection')
      setSkillItem(benefit.skill_item ?? 'detection_presence')
      setSelectedCategories(benefit.categories ?? ['human::silhouette'])
      setEnabled(benefit.active !== false)
    } else if (open && !benefit) {
      setStep('skill')
      setSkill('detection')
      setSkillItem('detection_presence')
      setSelectedCategories(['human::silhouette'])
      setName(getDefaultBenefitName(cameraName))
      setCreatedBy('Opérateur')
      setCreatedAt(toLocalInputDateTime())
      setComment('')
      setScheduleEnabled(false)
      setScheduleStart('')
      setScheduleEnd('')
      setEnabled(true)
    }
  }, [open, benefit, cameraName])

  const skillGroups = getSkillGroups()
  const categoryGroups = getCategoryGroupsBySkill(skill)

  const activeSkill = skillGroups.find((g) => g.key === skill)
  const activeSkillItems = activeSkill?.items ?? []
  const validSkillItem = activeSkillItems.some((i) => i.id === skillItem)
    ? skillItem
    : activeSkillItems[0]?.id ?? skillItem

  const toggleCategory = useCallback((key: string) => {
    setSelectedCategories((prev) => {
      const set = new Set(prev)
      if (set.has(key)) set.delete(key)
      else set.add(key)
      return Array.from(set)
    })
  }, [])

  const toggleCategoryGroup = useCallback((key: string) => {
    setCategoryOpen((prev) => ({ ...prev, [key]: !(prev[key] ?? true) }))
  }, [])

  const stepIndex = STEPS.indexOf(step)
  const stepSkillDone = !!skill && !!validSkillItem
  const stepCategoryDone = selectedCategories.length > 0
  const stepInfoDone = (name || '').trim().length > 0
  const canFinalize = stepSkillDone && stepCategoryDone && stepInfoDone

  const handlePrev = () => {
    const idx = Math.max(0, stepIndex - 1)
    setStep(STEPS[idx])
    setError(null)
  }

  const handleNext = () => {
    if (step === 'category' && selectedCategories.length === 0) {
      setError('Sélectionnez au moins une catégorie.')
      return
    }
    if (step === 'info' && !(name || '').trim()) {
      setError('Le nom est requis.')
      return
    }
    setError(null)
    if (stepIndex >= STEPS.length - 1) {
      handleSave()
      return
    }
    setStep(STEPS[stepIndex + 1])
  }

  const handleSave = () => {
    if (!(name || '').trim()) {
      setError('Le nom est requis.')
      return
    }
    if (selectedCategories.length === 0) {
      setError('Sélectionnez au moins une catégorie.')
      return
    }

    const zonePolygons: number[][][] =
      (localZonePolygons?.length ?? 0) > 0
        ? localZonePolygons!
        : (benefit?.zone_polygons ?? []).length > 0
          ? benefit!.zone_polygons!
          : []

    setError(null)

    const refW = canvasW
    const refH = canvasH

    const benefitId = isEdit && benefit ? benefit.benefit_id : `ben-${cameraId}-${Date.now()}`.replace(/\s/g, '_')
    const benefitData: HierarchyBenefit = {
      benefit_id: benefitId,
      name: (name || '').trim(),
      skill: skill ?? 'detection',
      skill_item: validSkillItem,
      categories: selectedCategories,
      camera_id: cameraId,
      zone_polygons: zonePolygons,
      zone_polygon_types: localZonePolygonTypes.length >= zonePolygons.length ? localZonePolygonTypes : zonePolygons.map(() => 'include' as const),
      zone_ref_width: refW,
      zone_ref_height: refH,
      active: enabled,
    }

    optimisticMutation({
      optimisticApply: () => {
        optimisticMergeBenefit(cameraId, benefitData)
        onClose()
      },
      mutate: () =>
        isEdit && benefit
          ? updateBenefit(benefit.benefit_id, {
              name: benefitData.name,
              skill: benefitData.skill,
              skill_item: benefitData.skill_item,
              categories: benefitData.categories,
              zone_polygons: benefitData.zone_polygons,
              zone_polygon_types: benefitData.zone_polygon_types,
              zone_ref_width: refW,
              zone_ref_height: refH,
              active: benefitData.active,
            })
          : createBenefit({
              benefit_id: benefitId,
              name: benefitData.name,
              skill: benefitData.skill ?? 'detection',
              skill_item: benefitData.skill_item ?? 'detection_presence',
              categories: benefitData.categories ?? ['human::silhouette'],
              camera_id: cameraId,
              zone_polygons: zonePolygons,
              zone_polygon_types: benefitData.zone_polygon_types,
              zone_ref_width: refW,
              zone_ref_height: refH,
              active: enabled,
            }),
      onSuccess: () => onSaved(isEdit ? undefined : benefitId),
      onError: (err) => {
        refetch()
        onSaveError?.(err.message)
      },
    })
  }

  const goToStep = (s: StepId) => {
    if (isEdit && s === 'skill') return
    setStep(s)
    setError(null)
  }

  const zonePolygons = localZonePolygons
  const hasEditorLeft = !!videoPath
  const frameUrl = videoPath ? getFrameUrl(videoPath) : ''
  const canvasW = (benefit?.zone_ref_width != null && benefit.zone_ref_width > 0) ? benefit.zone_ref_width : (videoInfo?.width ?? 1280)
  const canvasH = (benefit?.zone_ref_height != null && benefit.zone_ref_height > 0) ? benefit.zone_ref_height : (videoInfo?.height ?? 720)

  const {
    draftPoints,
    hoverBar,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handlePointerLeave,
    handleValidate,
    handleUndo,
    handleClear,
    handleDeletePoint,
    handleDeleteShape,
    resetUndoOnOpen,
  } = useBenefitEditorCanvas({
    polygons: localZonePolygons as Polygon[],
    setPolygons: setLocalZonePolygons as (p: Polygon[]) => void,
    polygonTypes: localZonePolygonTypes,
    setPolygonTypes: setLocalZonePolygonTypes,
    polygonIdx,
    setPolygonIdx,
    tool: editorTool,
    canvasRef,
    width: canvasW,
    height: canvasH,
  })

  const canValidateZones = draftPoints.length >= 3 || zonesModified

  useEffect(() => {
    if (open) resetUndoOnOpen()
  }, [open, resetUndoOnOpen])

  // Dessiner les polygones, draftPoints et poignées de sommets sur le canvas
  // scale = canvas coords par pixel affiché → pour garder traits/points à taille visuelle constante
  useEffect(() => {
    const canvas = canvasRef.current
    const polys = localZonePolygons
    const w = canvasW
    const h = canvasH
    if (!canvas || w <= 0 || h <= 0) return
    if (canvas.width !== w || canvas.height !== h) return
    const rect = canvas.getBoundingClientRect()
    const scale = Math.max(w / rect.width, h / rect.height)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, w, h)
    const getPolyColor = (idx: number) => (localZonePolygonTypes[idx] === 'exclude' ? 'rgba(240,131,33,0.35)' : 'rgba(34,197,94,0.35)')
    polys.forEach((poly, idx) => {
      if (poly.length < 2) return
      ctx.beginPath()
      ctx.moveTo(poly[0][0], poly[0][1])
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1])
      ctx.closePath()
      ctx.fillStyle = getPolyColor(idx)
      ctx.fill()
      ctx.strokeStyle = idx === polygonIdx ? 'rgba(29,91,255,0.9)' : 'rgba(255,255,255,0.5)'
      ctx.lineWidth = (idx === polygonIdx ? 3 : 1.5) * scale
      ctx.stroke()
    })
    // Draft points (include/exclude) — style moderne, trait plein
    if (draftPoints.length > 0) {
      const draftColor = editorTool === 'include' ? 'rgba(34,197,94,0.4)' : 'rgba(240,131,33,0.4)'
      const draftStroke = editorTool === 'include' ? 'rgba(34,197,94,0.95)' : 'rgba(240,131,33,0.95)'
      ctx.beginPath()
      ctx.moveTo(draftPoints[0][0], draftPoints[0][1])
      for (let i = 1; i < draftPoints.length; i++) ctx.lineTo(draftPoints[i][0], draftPoints[i][1])
      ctx.closePath()
      if (draftPoints.length >= 3) {
        ctx.fillStyle = draftColor
        ctx.fill()
      }
      ctx.strokeStyle = draftStroke
      ctx.lineWidth = 2.5 * scale
      ctx.lineJoin = 'round'
      ctx.lineCap = 'round'
      ctx.stroke()
      const r = 5 * scale
      draftPoints.forEach((pt) => {
        ctx.beginPath()
        ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2)
        ctx.fillStyle = editorTool === 'include' ? 'rgba(34,197,94,0.95)' : 'rgba(240,131,33,0.95)'
        ctx.fill()
        ctx.strokeStyle = 'rgba(255,255,255,0.9)'
        ctx.lineWidth = 1.5 * scale
        ctx.stroke()
      })
    }
    // Poignées de sommets en mode Sélect
    if (editorTool === 'select' && polys.length > 0 && polygonIdx !== null && polys[polygonIdx]) {
      const poly = polys[polygonIdx]
      const r = 6 * scale
      poly.forEach((pt) => {
        ctx.beginPath()
        ctx.arc(pt[0], pt[1], r, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(29,91,255,0.9)'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1.5 * scale
        ctx.stroke()
      })
    }
  }, [localZonePolygons, localZonePolygonTypes, polygonIdx, draftPoints, editorTool, canvasW, canvasH])

  if (!open) return null

  const benPanelContent = (
    <div className="ben-panel" style={{ minHeight: '100%' }}>
      <div className="ben-wizard">
        <section className="ben-stage">
          <div className="ben-stepper">
            {STEPS.map((s, i) => (
              <span key={s} className="ben-step-wrap">
                <button
                  type="button"
                  className={`ben-step-item ${isEdit && s === 'skill' ? 'is-locked' : ''}`}
                  data-state={i < stepIndex ? 'completed' : i === stepIndex ? 'active' : 'inactive'}
                  onClick={() => goToStep(s)}
                  disabled={isEdit && s === 'skill'}
                >
                  <span className="ben-step-indicator">
                    <span className="ben-step-num">{i + 1}</span>
                    <span className="ben-step-spin" />
                    <span className="ben-step-check">✓</span>
                  </span>
                  <span className="ben-step-title">
                    {s === 'skill' && 'Skill'}
                    {s === 'category' && 'Catégorie'}
                    {s === 'info' && 'Infos'}
                  </span>
                </button>
                {i < STEPS.length - 1 && (
                  <span className={`ben-step-sep ${i < stepIndex ? 'is-completed' : ''}`} />
                )}
              </span>
            ))}
          </div>

          {!skillsReady ? (
            <div style={{ padding: 12, color: 'rgba(234,242,255,0.7)' }}>Chargement…</div>
          ) : (
            <>
              {error && (
                <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 12 }}>{error}</div>
              )}

              <div className={`ben-step-panel ${step === 'skill' ? 'active' : ''}`} data-ben-step-panel="skill">
                <div className="ben-block">
                  <div className="ben-tree-shell">
                    <div className="ben-skill-cards">
                      {skillGroups.map((g) => (
                        <button
                          key={g.key}
                          type="button"
                          className={`ben-skill-card ${skill === g.key ? 'active' : ''}`}
                          onClick={() => { setSkill(g.key); setSkillItem(g.items[0]?.id ?? skillItem) }}
                        >
                          <img src={g.icon} alt="" className="ben-skill-card__ico" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />
                          <span className="ben-skill-card__label">{g.label}</span>
                        </button>
                      ))}
                    </div>
                    <div className="ben-tree">
                      {activeSkillItems.map((it) => (
                        <button
                          key={it.id}
                          type="button"
                          className={`ben-tree-leaf ${validSkillItem === it.id ? 'active' : ''}`}
                          onClick={() => setSkillItem(it.id)}
                        >
                          <img src={it.icon} alt="" className="ben-tree-ico" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />
                          <span>{it.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className={`ben-step-panel ${step === 'category' ? 'active' : ''}`} data-ben-step-panel="category">
                <div className="ben-block">
                  <div className="ben-tree">
                    {categoryGroups.map((grp) => (
                      <div key={grp.key} className={`ben-tree-group ${(categoryOpen[grp.key] ?? true) ? 'is-open' : ''} ${selectedCategories.some((k) => k.startsWith(`${grp.key}::`)) ? 'is-active' : ''}`}>
                        <button type="button" className={`ben-tree-toggle ${selectedCategories.some((k) => k.startsWith(`${grp.key}::`)) ? 'active' : ''}`} onClick={() => toggleCategoryGroup(grp.key)}>
                          <img src={grp.icon} alt="" className="ben-tree-ico" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />
                          <span className="ben-tree-label">{grp.label}</span>
                          <svg className="ben-tree-chev" width="10" height="10" viewBox="0 0 10 10"><path d="M3 2l4 3-4 3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
                        </button>
                        <div className={`ben-tree-leaves ${(categoryOpen[grp.key] ?? true) ? '' : 'collapsed'}`}>
                          {(grp.items ?? []).map((it) => {
                            const key = `${grp.key}::${it.id}`
                            return (
                              <button
                                key={key}
                                type="button"
                                className={`ben-tree-leaf ${selectedCategories.includes(key) ? 'active' : ''}`}
                                onClick={() => toggleCategory(key)}
                              >
                                <img src={it.icon} alt="" className="ben-tree-ico" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />
                                <span>{it.label}</span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className={`ben-step-panel ${step === 'info' ? 'active' : ''}`} data-ben-step-panel="info">
                <div className="ben-block">
                  <div className="ben-info-card">
                    <div className="ben-info-grid">
                      <label className="ben-field ben-field--full">
                        <span className="ben-field__label">Nom</span>
                        <input type="text" className="ben-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Ex: Waiting line - Entrance" />
                      </label>
                      <div className="ben-field ben-field--full">
                        <div className="ben-recap ben-recap--under-name">
                          {(() => {
                            const toneBySkill: Record<string, string> = { detection: 'success', counting: 'primary', heatmap: 'info', quality: 'warning' }
                            const tone = toneBySkill[skill] ?? 'primary'
                            const accentPalette = ['mint', 'sky', 'violet', 'peach', 'rose', 'sand', 'teal', 'slate', 'lime', 'coral', 'azure', 'amber']
                            const accentFromKey = (key: string) => accentPalette[Math.abs([...key].reduce((h, c) => ((h << 5) - h) + c.charCodeAt(0), 0)) % accentPalette.length]
                            const toWord = (txt: string) => String(txt || '').trim().split(/\s+/)[0] || ''
                            const chips: { icon: string; label: string; appearance: 'solid' | 'ghost'; tone: string; accent?: string; iconOnly: boolean }[] = []
                            if (activeSkill) {
                              chips.push({ icon: activeSkill.icon, label: toWord(activeSkill.label), appearance: 'solid', tone, iconOnly: false })
                            }
                            const catByKey: Record<string, { icon: string; label: string }> = {}
                            categoryGroups.forEach((g) => (g.items ?? []).forEach((it) => {
                              catByKey[`${g.key}::${it.id}`] = { icon: it.icon || g.icon, label: it.label || g.label || '' }
                            }))
                            const selectedCats = selectedCategories
                              .map((k) => {
                                const c = catByKey[k]
                                return c ? { key: k, icon: c.icon, label: c.label } : null
                              })
                              .filter((x): x is { key: string; icon: string; label: string } => !!x)
                            const skillSubItem = activeSkillItems.find((i) => i.id === validSkillItem)
                            if (selectedCats.length > 0) {
                              selectedCats.forEach(({ key, icon }) => chips.push({
                                icon,
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
                                accent: accentFromKey(validSkillItem),
                                iconOnly: true,
                              })
                            }
                            return chips.map((c, i) => (
                              <span
                                key={i}
                                className={`ben-recap-chip ben-recap-chip--${c.appearance} ben-recap-chip--${c.tone}${c.accent ? ` ben-recap-chip--accent-${c.accent}` : ''}${c.iconOnly ? ' ben-recap-chip--icon-only' : ''}`}
                              >
                                {c.icon && <img className="ben-tree-ico" src={c.icon} alt="" onError={(e) => { (e.target as HTMLImageElement).src = FALLBACK_ICON }} />}
                                {!c.iconOnly && <span>{c.label}</span>}
                              </span>
                            ))
                          })()}
                        </div>
                      </div>
                      <label className="ben-field ben-field--full">
                        <span className="ben-field__label">Créé par</span>
                        <input type="text" className="ben-input" value={createdBy} onChange={(e) => setCreatedBy(e.target.value)} placeholder="Nom opérateur" />
                      </label>
                      <label className="ben-field">
                        <span className="ben-field__label">Créé le</span>
                        <input type="datetime-local" className="ben-input" value={createdAt} onChange={(e) => setCreatedAt(e.target.value)} />
                      </label>
                      <label className="ben-field ben-field--full">
                        <span className="ben-field__label">Commentaire</span>
                        <textarea className="ben-textarea" value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Contexte, consignes, notes..." />
                      </label>
                      <div className="ben-toggle-row ben-field--full">
                        <span className="ben-toggle-row__label">Zone activée</span>
                        <label className="ben-toggle ben-toggle--success">
                          <input type="checkbox" className="ben-toggle__input" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
                          <span className="ben-toggle__track">
                            <span className="ben-toggle__dot ben-toggle__dot--left" />
                            <span className="ben-toggle__dot ben-toggle__dot--right" />
                          </span>
                        </label>
                      </div>
                      <div className="ben-toggle-row ben-field--full">
                        <span className="ben-toggle-row__label">Scheduling</span>
                        <label className="ben-toggle">
                          <input type="checkbox" className="ben-toggle__input" checked={scheduleEnabled} onChange={(e) => setScheduleEnabled(e.target.checked)} />
                          <span className="ben-toggle__track">
                            <span className="ben-toggle__dot ben-toggle__dot--left" />
                            <span className="ben-toggle__dot ben-toggle__dot--right" />
                          </span>
                        </label>
                      </div>
                      <div className={`ben-schedule-grid ben-field--full ${scheduleEnabled ? '' : 'hidden'}`}>
                        <label className="ben-field">
                          <span className="ben-field__label">Début</span>
                          <input type="datetime-local" className="ben-input" value={scheduleStart} onChange={(e) => setScheduleStart(e.target.value)} />
                        </label>
                        <label className="ben-field">
                          <span className="ben-field__label">Fin</span>
                          <input type="datetime-local" className="ben-input" value={scheduleEnd} onChange={(e) => setScheduleEnd(e.target.value)} />
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="ben-step-actions">
                <button type="button" className="tool-btn ben-step-btn" onClick={handlePrev} disabled={stepIndex <= 0}>Retour</button>
                {isEdit ? (
                  <button type="button" className="tool-btn ben-step-btn ben-step-btn--next save" onClick={handleSave} disabled={!canFinalize}>
                    SAUVEGARDER
                  </button>
                ) : (
                  <button
                    type="button"
                    className={`tool-btn ben-step-btn ben-step-btn--next ${stepIndex >= STEPS.length - 1 ? 'save' : ''}`}
                    onClick={handleNext}
                  >
                    {stepIndex >= STEPS.length - 1 ? 'Terminer' : 'Suivant'}
                  </button>
                )}
              </div>
            </>
          )}
        </section>
      </div>
      <div className="ben-watermark" aria-hidden="true">
        <img src="/static/assets_youn/YrysUIPackage/ArcyWhitelogo.svg" alt="" />
      </div>
    </div>
  )

  const modalContent = (
    <div
      className="editor-overlay"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.65)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 'var(--space-6)',
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label={isEdit ? 'Modifier le bénéfice' : 'Nouveau bénéfice'}
    >
      <div
        className={`editor ${hasEditorLeft ? 'benefit-mode-open' : ''}`}
        style={{
          width: hasEditorLeft ? 'min(1460px, 96vw)' : 'min(480px, 96vw)',
          height: hasEditorLeft ? 'min(90vh, 860px)' : 'auto',
          maxHeight: '90vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          background: '#111',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 12,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="editor-topbar" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 18px', background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'rgba(255,255,255,0.4)', marginTop: 2 }}>Zone Tracker</div>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#F5F7FF' }}>{isEdit ? 'Modifier le bénéfice' : 'Nouveau bénéfice'}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {isEdit && benefit && (
              <CardMenu
                className="benefit-modal-topbar-menu"
                title="Options"
                panelZIndex={10001}
                options={[
                  {
                    label: 'Supprimer le bénéfice',
                    danger: true,
                    onClick: async () => {
                      if (!confirm('Supprimer ce bénéfice ?')) return
                      try {
                        await deleteBenefit(benefit.benefit_id)
                        onSaved()
                        onClose()
                      } catch (err) {
                        onSaveError?.(err instanceof Error ? err.message : 'Erreur suppression')
                      }
                    },
                  },
                ]}
              />
            )}
            <button type="button" onClick={onClose} aria-label="Fermer" style={{ color: 'rgba(255,255,255,0.5)', fontSize: 18, border: 'none', background: 'transparent', cursor: 'pointer' }}>✕</button>
          </div>
        </div>

        <div className="benefit-modal-root" style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {hasEditorLeft ? (
            <div className="editor-body" style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
              <div className="editor-left">
                <div
                  className={`editor-canvas-wrap ${editorTool !== 'select' ? 'is-draw-mode' : ''}`}
                  style={canvasW > 0 && canvasH > 0 ? { aspectRatio: `${canvasW} / ${canvasH}` } : undefined}
                >
                  <img ref={imgRef} src={frameUrl} alt="Frame" />
                  <canvas
                    ref={canvasRef}
                    width={canvasW}
                    height={canvasH}
                    style={{ pointerEvents: 'auto' }}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={() => { handlePointerUp(); handlePointerLeave() }}
                  />
                  <div
                    className={`editor-hoverbar ${hoverBar.visible ? 'editor-hoverbar--visible' : 'editor-hoverbar--hidden'}`}
                    style={{
                      left: hoverBar.x,
                      top: hoverBar.y,
                    }}
                  >
                    {hoverBar.kind === 'vertex' && (
                      <button
                        type="button"
                        className="editor-hoverbtn danger"
                        title="Supprimer le point"
                        onClick={handleDeletePoint}
                      >
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 6l12 12M18 6L6 18" />
                        </svg>
                      </button>
                    )}
                    {hoverBar.kind === 'poly' && (
                      <button
                        type="button"
                        className="editor-hoverbtn danger"
                        title="Supprimer la forme"
                        onClick={handleDeleteShape}
                      >
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                  <div className={`editor-shape-labels ${zonePolygons.length === 0 ? 'hidden' : ''}`}>
                    {zonePolygons.map((_, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className={`editor-shape-label ${polygonIdx === idx ? 'active' : ''}`}
                        title="Sélectionner cette forme"
                        onClick={() => {
                          setEditorTool('select')
                          setPolygonIdx(idx)
                        }}
                      >
                        Forme {idx + 1} - {(localZonePolygonTypes[idx] ?? 'include') === 'include' ? 'Include' : 'Exclure'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="editor-tools">
                  <div className="tool-row">
                    <div className="tool-row-left">
                      <button
                        type="button"
                        className={`tool-btn icon primary ${editorTool === 'select' ? 'active' : ''}`}
                        title="Sélection"
                        onClick={() => setEditorTool('select')}
                      >
                        <img className="tool-ico" src={ICON_SELECT} alt="" />
                        <span className="tool-label">Sélect</span>
                      </button>
                      <button
                        type="button"
                        className={`tool-btn icon green ${editorTool === 'include' ? 'active' : ''}`}
                        title="Zone d'inclusion"
                        onClick={() => setEditorTool('include')}
                      >
                        <img className="tool-ico" src={ICON_INCLUDE} alt="" />
                        <span className="tool-label">Inclure</span>
                      </button>
                        <button
                        type="button"
                        className={`tool-btn icon yellow ${editorTool === 'exclude' ? 'active' : ''}`}
                        title="Zone d'exclusion"
                        onClick={() => setEditorTool('exclude')}
                      >
                        <img className="tool-ico" src={ICON_EXCLUDE} alt="" />
                        <span className="tool-label">Exclure</span>
                      </button>
                      <button type="button" className="tool-btn icon compact" title="Annuler (Ctrl+Z)" onClick={handleUndo}>↶</button>
                      <button type="button" className="tool-btn icon red compact" title="Tout effacer" onClick={handleClear}>✕</button>
                    </div>
                    <button
                      type="button"
                      className="tool-btn ben-step-btn ben-step-btn--next"
                      title="Valider la forme"
                      disabled={!canValidateZones}
                      onClick={() => {
                        handleValidate()
                        setEditorTool('select')
                      }}
                    >
                      VALIDER
                    </button>
                  </div>
                  <div className="editor-guide">
                    {editorTool === 'select' ? (
                      <>
                        Sélection : cliquez un point puis glissez pour déplacer.<br />
                        <span className="editor-guide-kbd">⇧</span> + clic près d&apos;une arête = ajouter un point.
                      </>
                    ) : editorTool === 'include' ? (
                      "Zone d'inclusion: cliquez pour placer des points (3+), puis Valider."
                    ) : editorTool === 'exclude' ? (
                      "Zone d'exclusion: cliquez pour placer des points (3+), puis Valider."
                    ) : (
                      "Choisissez un outil pour dessiner ou modifier les zones."
                    )}
                  </div>
                </div>
              </div>
              <div className="editor-right" style={{ overflow: 'auto', background: '#161616', borderLeft: '1px solid rgba(255,255,255,0.06)' }}>
                {benPanelContent}
              </div>
            </div>
          ) : (
            <div style={{ overflow: 'auto' }}>
              {benPanelContent}
            </div>
          )}
        </div>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
}
