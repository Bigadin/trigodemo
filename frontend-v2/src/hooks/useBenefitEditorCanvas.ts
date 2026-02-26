import type React from 'react'
import { useCallback, useRef, useState, useEffect } from 'react'

const HOVER_BAR_PERSIST_MS = 2500

export type Point = [number, number]
export type Polygon = Point[]

function clonePoints(pts: Point[]): Point[] {
  return pts.map((p) => [p[0], p[1]])
}

function dist2(a: Point, b: Point): number {
  const dx = a[0] - b[0]
  const dy = a[1] - b[1]
  return dx * dx + dy * dy
}

function nearestVertex(poly: Polygon, p: Point, radius = 32): number {
  let best = -1
  let bestD = Infinity
  for (let i = 0; i < poly.length; i++) {
    const d = dist2(poly[i], p)
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return bestD <= radius * radius ? best : -1
}

function pointInPoly(poly: Polygon, p: Point): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect =
      (yi > p[1]) !== (yj > p[1]) &&
      p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi + 1e-9) + xi
    if (intersect) inside = !inside
  }
  return inside
}

function pointToSegmentDistanceSquared(p: Point, a: Point, b: Point): number {
  const vx = b[0] - a[0]
  const vy = b[1] - a[1]
  const wx = p[0] - a[0]
  const wy = p[1] - a[1]
  const c1 = vx * wx + vy * wy
  if (c1 <= 0) return dist2(p, a)
  const c2 = vx * vx + vy * vy
  if (c2 <= c1) return dist2(p, b)
  const t = c1 / c2
  const proj: Point = [a[0] + t * vx, a[1] + t * vy]
  return dist2(p, proj)
}

const EDGE_INSERT_THRESHOLD_SQ = 32 * 32  // distance² max pour insérer sur une arête (32px)

function findNearestEdge(polygons: Polygon[], p: Point): { polyIdx: number; insertAfter: number } | null {
  let bestPolyIdx = -1
  let bestInsertAfter = -1
  let bestD = Infinity
  for (let idx = 0; idx < polygons.length; idx++) {
    const poly = polygons[idx]
    if (!poly || poly.length < 3) continue
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i]
      const b = poly[(i + 1) % poly.length]
      const d = pointToSegmentDistanceSquared(p, a, b)
      if (d < bestD) {
        bestD = d
        bestPolyIdx = idx
        bestInsertAfter = i
      }
    }
  }
  if (bestPolyIdx < 0 || bestD > EDGE_INSERT_THRESHOLD_SQ) return null
  return { polyIdx: bestPolyIdx, insertAfter: bestInsertAfter }
}

function insertPointOnEdge(poly: Polygon, p: Point): boolean {
  if (!poly || poly.length < 3) return false
  let bestIdx = -1
  let bestD = Infinity
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]
    const b = poly[(i + 1) % poly.length]
    const d = pointToSegmentDistanceSquared(p, a, b)
    if (d < bestD) {
      bestD = d
      bestIdx = i
    }
  }
  if (bestIdx < 0 || bestD > EDGE_INSERT_THRESHOLD_SQ) return false
  poly.splice(bestIdx + 1, 0, [p[0], p[1]])
  return true
}

export type PolygonType = 'include' | 'exclude'

export interface UseBenefitEditorCanvasProps {
  polygons: Polygon[]
  setPolygons: (p: Polygon[]) => void
  polygonTypes?: PolygonType[]
  setPolygonTypes?: (t: PolygonType[] | ((prev: PolygonType[]) => PolygonType[])) => void
  polygonIdx: number | null
  setPolygonIdx: React.Dispatch<React.SetStateAction<number | null>>
  tool: 'select' | 'include' | 'exclude'
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  width: number
  height: number
}

export interface HoverBarState {
  visible: boolean
  x: number
  y: number
  kind: 'vertex' | 'poly'
  vIdx: number | null
  polyIdx: number | null
}

export function useBenefitEditorCanvas({
  polygons,
  setPolygons,
  polygonTypes = [],
  setPolygonTypes,
  polygonIdx,
  setPolygonIdx,
  tool,
  canvasRef,
  width,
  height,
}: UseBenefitEditorCanvasProps) {
  const [draftPoints, setDraftPoints] = useState<Point[]>([])
  const [undoStack, setUndoStack] = useState<{ polygons: Polygon[]; types: PolygonType[] }[]>([])
  const [hoverBar, setHoverBar] = useState<HoverBarState>({
    visible: false,
    x: 0,
    y: 0,
    kind: 'vertex',
    vIdx: null,
    polyIdx: null,
  })
  const [hoverBarVisible, setHoverBarVisible] = useState(false)
  const lastTargetRef = useRef<{ kind: 'vertex' | 'poly'; polyIdx: number; vIdx: number | null } | null>(null)
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const dragRef = useRef<{ kind: 'vertex'; vIdx: number } | { kind: 'poly'; start: Point } | null>(null)
  const didDragRef = useRef(false)

  const pushUndo = useCallback(() => {
    const types = setPolygonTypes ? polygonTypes : polygons.map((_, i) => (i % 2 === 0 ? 'include' : 'exclude') as PolygonType)
    setUndoStack((s) => [...s.slice(-49), { polygons: polygons.map((p) => clonePoints(p)), types: [...types] }])
  }, [polygons, polygonTypes, setPolygonTypes])

  const getEventPoint = useCallback(
    (e: React.PointerEvent): Point => {
      const canvas = canvasRef.current
      if (!canvas) return [0, 0]
      const rect = canvas.getBoundingClientRect()
      const scaleX = width / rect.width
      const scaleY = height / rect.height
      return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY]
    },
    [canvasRef, width, height]
  )

  const pickTarget = useCallback(
    (p: Point): { idx: number; kind: 'vertex'; vIdx: number } | { idx: number; kind: 'poly' } | null => {
      for (let idx = polygons.length - 1; idx >= 0; idx--) {
        const poly = polygons[idx]
        if (!poly || poly.length < 3) continue
        const vIdx = nearestVertex(poly, p, 34)
        if (vIdx >= 0) return { idx, kind: 'vertex', vIdx }
        if (pointInPoly(poly, p)) return { idx, kind: 'poly' }
      }
      return null
    },
    [polygons]
  )

  const hideHoverBar = useCallback(() => {
    if (persistTimerRef.current) {
      clearTimeout(persistTimerRef.current)
      persistTimerRef.current = null
    }
    lastTargetRef.current = null
    setHoverBarVisible(false)
    setHoverBar((h) => ({ ...h, visible: false }))
  }, [])

  const scheduleHide = useCallback(() => {
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current)
    persistTimerRef.current = setTimeout(() => {
      persistTimerRef.current = null
      lastTargetRef.current = null
      setHoverBarVisible(false)
      setHoverBar((h) => ({ ...h, visible: false }))
    }, HOVER_BAR_PERSIST_MS)
  }, [])

  useEffect(() => () => {
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current)
  }, [])

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      const p = getEventPoint(e)
      if (tool === 'include' || tool === 'exclude') {
        setDraftPoints((d) => [...d, p])
        return
      }
      if (tool === 'select') {
        // Shift+clic : insérer un point sur l'arête la plus proche (même si pas près d'un sommet)
        if (e.shiftKey && polygons.length > 0) {
          const edge = findNearestEdge(polygons, p)
          if (edge) {
            const polyCopy = [...polygons[edge.polyIdx]]
            polyCopy.splice(edge.insertAfter + 1, 0, [p[0], p[1]])
            pushUndo()
            const next = polygons.map((poly, i) => (i === edge.polyIdx ? polyCopy : poly))
            setPolygons(next)
            setPolygonIdx(edge.polyIdx)
            hideHoverBar()
            return
          }
        }

        const target = pickTarget(p)
        if (!target) {
          setPolygonIdx(null)
          hideHoverBar()
          return
        }
        setPolygonIdx(target.idx)
        hideHoverBar()
        if (target.kind === 'vertex') {
          const polyCopy = [...polygons[target.idx]]
          const inserted = e.shiftKey && insertPointOnEdge(polyCopy, p)
          if (inserted) {
            pushUndo()
            const next = polygons.map((poly, i) => (i === target.idx ? polyCopy : poly))
            setPolygons(next)
          } else {
            dragRef.current = { kind: 'vertex', vIdx: target.vIdx }
            didDragRef.current = false
            canvasRef.current?.setPointerCapture(e.pointerId)
          }
        } else {
          dragRef.current = { kind: 'poly', start: p }
          didDragRef.current = false
          canvasRef.current?.setPointerCapture(e.pointerId)
        }
      }
    },
    [tool, getEventPoint, pushUndo, pickTarget, polygons, setPolygons, setPolygonIdx, hideHoverBar]
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      const p = getEventPoint(e)
      const canvas = canvasRef.current
      if (!canvas) return

      const rect = canvas.getBoundingClientRect()
      const xCss = (e.clientX - rect.left)
      const yCss = (e.clientY - rect.top)

      if (dragRef.current) {
        const drag = dragRef.current
        if (drag.kind === 'vertex' && polygonIdx !== null && polygons[polygonIdx]) {
          const poly = polygons[polygonIdx]
          const next = poly.map((pt, i) => (i === drag.vIdx ? p : pt))
          setPolygons(polygons.map((p, i) => (i === polygonIdx ? next : p)))
          didDragRef.current = true
        } else if (drag.kind === 'poly' && polygonIdx !== null && polygons[polygonIdx]) {
          const dx = p[0] - drag.start[0]
          const dy = p[1] - drag.start[1]
          const next = polygons[polygonIdx].map((pt) => [pt[0] + dx, pt[1] + dy] as Point)
          setPolygons(polygons.map((p, i) => (i === polygonIdx ? next : p)))
          dragRef.current = { kind: 'poly', start: p }
          didDragRef.current = true
        }
        return
      }

      if (tool === 'select' && polygons.length > 0) {
        const target = pickTarget(p)
        const wrap = canvas.parentElement
        const wrapRect = wrap?.getBoundingClientRect()
        const offsetX = wrapRect ? rect.left - wrapRect.left : 0
        const offsetY = wrapRect ? rect.top - wrapRect.top : 0
        const xCssClamp = Math.min(rect.width - 92, Math.max(0, xCss + 12))
        const yCssClamp = Math.min(rect.height - 46, Math.max(0, yCss - 46 - 12))
        const xInWrap = offsetX + xCssClamp
        const yInWrap = offsetY + yCssClamp
        if (target) {
          if (persistTimerRef.current) {
            clearTimeout(persistTimerRef.current)
            persistTimerRef.current = null
          }
          const newTarget = target.kind === 'vertex'
            ? { kind: 'vertex' as const, polyIdx: target.idx, vIdx: target.vIdx }
            : { kind: 'poly' as const, polyIdx: target.idx, vIdx: null }
          const last = lastTargetRef.current
          const targetChanged = !last || last.kind !== newTarget.kind || last.polyIdx !== newTarget.polyIdx || last.vIdx !== newTarget.vIdx
          if (targetChanged) {
            lastTargetRef.current = newTarget
            setHoverBar({
              visible: true,
              x: xInWrap,
              y: yInWrap,
              kind: target.kind,
              vIdx: target.kind === 'vertex' ? target.vIdx : null,
              polyIdx: target.idx,
            })
            setHoverBarVisible(true)
          }
        } else {
          if (lastTargetRef.current) {
            scheduleHide()
          } else {
            hideHoverBar()
          }
        }
      }
    },
    [tool, getEventPoint, polygons, setPolygons, polygonIdx, pickTarget, canvasRef, width, height, hideHoverBar]
  )

  const handlePointerUp = useCallback(() => {
    dragRef.current = null
  }, [])

  const handlePointerLeave = useCallback(() => {
    if (lastTargetRef.current) scheduleHide()
  }, [scheduleHide])

  const handleValidate = useCallback(() => {
    if ((tool === 'include' || tool === 'exclude') && draftPoints.length >= 3) {
      pushUndo()
      setPolygons([...polygons, clonePoints(draftPoints)])
      if (setPolygonTypes) setPolygonTypes((t) => [...t, tool])
      setDraftPoints([])
      setPolygonIdx(polygons.length)
    }
  }, [tool, draftPoints, polygons, setPolygons, setPolygonTypes, setPolygonIdx, pushUndo])

  const handleUndo = useCallback(() => {
    if (draftPoints.length > 0) {
      setDraftPoints((d) => d.slice(0, -1))
      return
    }
    if (undoStack.length === 0) return
    const prev = undoStack[undoStack.length - 1]
    setUndoStack((s) => s.slice(0, -1))
    setPolygons(prev.polygons)
    if (setPolygonTypes) setPolygonTypes(prev.types)
    setPolygonIdx(prev.polygons.length > 0 ? prev.polygons.length - 1 : null)
  }, [draftPoints, undoStack, setPolygons, setPolygonTypes, setPolygonIdx])

  const handleClear = useCallback(() => {
    if (draftPoints.length > 0) {
      setDraftPoints([])
      return
    }
    if (polygonIdx !== null && polygons.length > 0 && polygons[polygonIdx]) {
      pushUndo()
      const idxToRemove = polygonIdx
      setPolygons(polygons.filter((_, i) => i !== idxToRemove))
      if (setPolygonTypes) setPolygonTypes((t) => t.filter((_, i) => i !== idxToRemove))
      setPolygonIdx((prev) => (prev !== null && prev >= idxToRemove ? Math.max(0, prev - 1) : prev))
      hideHoverBar()
      return
    }
    pushUndo()
    setPolygons([])
    if (setPolygonTypes) setPolygonTypes([])
    setPolygonIdx(null)
    hideHoverBar()
  }, [draftPoints, polygonIdx, polygons, pushUndo, setPolygons, setPolygonTypes, setPolygonIdx, hideHoverBar])

  const handleDeletePoint = useCallback(() => {
    if (hoverBar.vIdx === null || hoverBar.polyIdx === null) return
    const poly = polygons[hoverBar.polyIdx]
    if (poly.length <= 3) return
    pushUndo()
    const next = poly.filter((_, i) => i !== hoverBar.vIdx)
    setPolygons(polygons.map((p, i) => (i === hoverBar.polyIdx! ? next : p)))
    setPolygonIdx(hoverBar.polyIdx)
    hideHoverBar()
  }, [hoverBar, polygons, setPolygons, setPolygonIdx, pushUndo, hideHoverBar])

  const handleDeleteShape = useCallback(() => {
    if (hoverBar.polyIdx === null) return
    pushUndo()
    setPolygons(polygons.filter((_, i) => i !== hoverBar.polyIdx))
    if (setPolygonTypes) setPolygonTypes((t) => t.filter((_, i) => i !== hoverBar.polyIdx))
    setPolygonIdx(prev => (prev !== null && prev >= hoverBar.polyIdx! ? Math.max(0, prev - 1) : prev))
    hideHoverBar()
  }, [hoverBar, polygons, setPolygons, setPolygonTypes, setPolygonIdx, pushUndo, hideHoverBar])

  const resetUndoOnOpen = useCallback(() => {
    setUndoStack([])
    setDraftPoints([])
  }, [])

  return {
    draftPoints,
    hoverBar: { ...hoverBar, visible: hoverBar.visible && hoverBarVisible },
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
  }
}
