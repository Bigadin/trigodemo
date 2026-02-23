import { useRef, useEffect, useCallback } from 'react'
import { useEditorStore, type Point } from '@/stores/editorStore'
import styles from './ZoneEditor.module.css'

function dist2(a: Point, b: Point) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
}

function pointInPoly(poly: Point[], p: Point): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect = ((yi > p[1]) !== (yj > p[1])) &&
      (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-9) + xi)
    if (intersect) inside = !inside
  }
  return inside
}

function nearestVertex(poly: Point[], p: Point, radius = 32): number {
  let best = -1, bestD = Infinity
  for (let i = 0; i < poly.length; i++) {
    const d = dist2(poly[i], p)
    if (d < bestD) { bestD = d; best = i }
  }
  return bestD <= radius * radius ? best : -1
}

function colorsForType(type: string, isActive: boolean) {
  const m: Record<string, { fill: string; stroke: string }> = {
    include: { fill: 'rgba(34,197,94,0.18)', stroke: isActive ? '#22c55e' : 'rgba(34,197,94,0.6)' },
    exclude: { fill: 'rgba(239,68,68,0.18)', stroke: isActive ? '#ef4444' : 'rgba(239,68,68,0.6)' },
    countingROI: { fill: 'rgba(29,91,255,0.18)', stroke: isActive ? '#1d5bff' : 'rgba(29,91,255,0.6)' },
  }
  return m[type] || m.include
}

function getEventPoint(e: React.PointerEvent, canvas: HTMLCanvasElement): Point {
  const rect = canvas.getBoundingClientRect()
  const scaleX = canvas.width / rect.width
  const scaleY = canvas.height / rect.height
  return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY]
}

export default function EditorCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<HTMLImageElement>(null)

  const open = useEditorStore((s) => s.open)
  const w = useEditorStore((s) => s.w)
  const h = useEditorStore((s) => s.h)
  const tool = useEditorStore((s) => s.tool)
  const zone = useEditorStore((s) => s.zone)
  const polygonIdx = useEditorStore((s) => s.polygonIdx)
  const points = useEditorStore((s) => s.points)
  const zones = useEditorStore((s) => s.zones)
  const drag = useEditorStore((s) => s.drag)
  const frameUrl = useEditorStore((s) => s.frameUrl)

  const addPoint = useEditorStore((s) => s.addPoint)
  const pushUndo = useEditorStore((s) => s.pushUndo)
  const setDrag = useEditorStore((s) => s.setDrag)
  const updateVertex = useEditorStore((s) => s.updateVertex)
  const movePolygon = useEditorStore((s) => s.movePolygon)
  const endDrag = useEditorStore((s) => s.endDrag)
  const selectPolygon = useEditorStore((s) => s.selectPolygon)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const selectedZone = zone
    const zoneKeys = Object.keys(zones).sort()

    zoneKeys.forEach((zoneName) => {
      const polys = zones[zoneName]?.polygons || []
      const isGhost = !!(selectedZone && zoneName !== selectedZone)
      const prevAlpha = ctx.globalAlpha
      ctx.globalAlpha = isGhost ? 0.22 : 1

      polys.forEach((poly, idx) => {
        if (!poly || poly.length < 3) return
        const isActive = !isGhost && polygonIdx === idx
        const c = colorsForType('include', isActive)

        ctx.beginPath()
        ctx.moveTo(poly[0][0], poly[0][1])
        for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1])
        ctx.closePath()
        ctx.fillStyle = c.fill
        ctx.fill()
        ctx.strokeStyle = c.stroke
        ctx.lineWidth = isActive ? 4 : isGhost ? 1.6 : 2
        ctx.stroke()
      })

      ctx.globalAlpha = prevAlpha
    })

    if (points.length > 0 && (tool === 'include' || tool === 'exclude' || tool === 'countingROI')) {
      const t = tool as string
      const c = colorsForType(t, false)
      ctx.beginPath()
      ctx.moveTo(points[0][0], points[0][1])
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1])
      if (points.length >= 3) {
        ctx.closePath()
        ctx.fillStyle = c.fill
        ctx.fill()
      }
      ctx.strokeStyle = c.stroke
      ctx.lineWidth = 3
      ctx.setLineDash([6, 4])
      ctx.stroke()
      ctx.setLineDash([])

      points.forEach((p, i) => {
        ctx.beginPath()
        ctx.arc(p[0], p[1], 8, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? '#1d5bff' : '#22c55e'
        ctx.fill()
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }

    if (tool === 'select' && zone && typeof polygonIdx === 'number') {
      const poly = zones[zone]?.polygons?.[polygonIdx]
      if (poly?.length) {
        poly.forEach((p) => {
          ctx.beginPath()
          ctx.arc(p[0], p[1], 9, 0, Math.PI * 2)
          ctx.fillStyle = '#1d5bff'
          ctx.fill()
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 2
          ctx.stroke()
        })
      }
    }
  }, [zone, zones, polygonIdx, points, tool])

  useEffect(() => { render() }, [render])

  useEffect(() => {
    if (!canvasRef.current) return
    canvasRef.current.width = w
    canvasRef.current.height = h
  }, [w, h])

  useEffect(() => {
    const frame = frameRef.current
    const canvas = canvasRef.current
    if (!frame || !canvas) return
    const syncSize = () => {
      const r = frame.getBoundingClientRect()
      canvas.style.width = r.width + 'px'
      canvas.style.height = r.height + 'px'
      render()
    }
    frame.addEventListener('load', syncSize)
    window.addEventListener('resize', syncSize)
    return () => {
      frame.removeEventListener('load', syncSize)
      window.removeEventListener('resize', syncSize)
    }
  }, [render, frameUrl])

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!open) return
    const canvas = canvasRef.current
    if (!canvas) return
    const p = getEventPoint(e, canvas)

    if (tool === 'include' || tool === 'exclude' || tool === 'countingROI') {
      addPoint(p)
      return
    }

    if (!zone) return
    const polys = zones[zone]?.polygons || []
    let found: { idx: number; vIdx: number } | null = null
    for (let idx = polys.length - 1; idx >= 0; idx--) {
      const poly = polys[idx]
      if (!poly || poly.length < 3) continue
      const vIdx = nearestVertex(poly, p, 34)
      if (vIdx >= 0) { found = { idx, vIdx }; break }
    }

    if (found) {
      selectPolygon(found.idx)
      pushUndo()
      setDrag({ kind: 'vertex', vIdx: found.vIdx })
      canvas.setPointerCapture(e.pointerId)
      render()
      return
    }

    let polyHit: number | null = null
    for (let idx = polys.length - 1; idx >= 0; idx--) {
      const poly = polys[idx]
      if (poly?.length >= 3 && pointInPoly(poly, p)) { polyHit = idx; break }
    }

    if (polyHit !== null) {
      selectPolygon(polyHit)
      pushUndo()
      setDrag({ kind: 'poly', start: p })
      canvas.setPointerCapture(e.pointerId)
    } else {
      selectPolygon(null)
    }
    render()
  }, [open, tool, zone, zones, addPoint, pushUndo, setDrag, selectPolygon, render])

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!open || !drag) return
    const canvas = canvasRef.current
    if (!canvas) return
    const p = getEventPoint(e, canvas)

    if (drag.kind === 'vertex' && typeof drag.vIdx === 'number') {
      updateVertex(drag.vIdx, p)
    } else if (drag.kind === 'poly' && drag.start) {
      const dx = p[0] - drag.start[0]
      const dy = p[1] - drag.start[1]
      movePolygon(dx, dy)
      useEditorStore.setState((s) => ({
        drag: s.drag ? { ...s.drag, start: p } : null,
      }))
    }
    render()
  }, [open, drag, updateVertex, movePolygon, render])

  const handlePointerUp = useCallback(() => {
    if (!open) return
    endDrag()
    render()
  }, [open, endDrag, render])

  const isDrawMode = tool === 'include' || tool === 'exclude' || tool === 'countingROI'

  if (!open) return null

  return (
    <div className={`${styles.canvasWrap} ${isDrawMode ? styles.canvasDrawMode : ''}`}>
      {frameUrl && (
        <img
          ref={frameRef}
          src={frameUrl}
          alt="Video frame"
          className={styles.canvasFrame}
          draggable={false}
        />
      )}
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={(e) => e.preventDefault()}
      />
    </div>
  )
}
