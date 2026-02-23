import { create } from 'zustand'
import { api } from '@/api/client'

export type EditorTool = 'select' | 'include' | 'exclude' | 'countingROI'
export type Point = [number, number]
export type Polygon = Point[]

export interface ZoneData {
  polygons: Polygon[]
}

export interface PresenceInfo {
  occupied: boolean
  total_time: number
  formatted_time?: string
}

interface DragState {
  kind: 'vertex' | 'poly'
  vIdx?: number
  start?: Point
}

interface UndoSnapshot {
  tool: EditorTool
  zone: string | null
  polygonIdx: number | null
  mode: string
  points: Point[]
  zonePolys: Polygon[] | null
}

interface EditorState {
  open: boolean
  tool: EditorTool
  zone: string | null
  polygonIdx: number | null
  mode: 'idle' | 'creating' | 'editing'
  points: Point[]
  drag: DragState | null
  undo: UndoSnapshot[]
  zones: Record<string, ZoneData>
  presence: Record<string, PresenceInfo>
  w: number
  h: number
  dirty: boolean
  didDrag: boolean
  video: string | null
  frameUrl: string | null

  openEditor: (video: string) => Promise<void>
  closeEditor: () => void
  setTool: (tool: EditorTool) => void
  selectZone: (name: string | null) => void
  selectPolygon: (idx: number | null) => void
  addPoint: (p: Point) => void
  pushUndo: () => void
  popUndo: () => void
  clearTemp: () => void
  setDrag: (d: DragState | null) => void
  updateVertex: (vIdx: number, p: Point) => void
  movePolygon: (dx: number, dy: number) => void
  endDrag: () => void
  deleteVertex: (vIdx: number) => void
  deleteShape: (idx: number) => Promise<void>
  commitDraft: () => Promise<void>
  saveAll: () => Promise<void>
  addZone: (name: string) => Promise<void>
  deleteZone: () => Promise<void>
  refreshFromServer: () => Promise<void>
}

function clonePoints(pts: Point[]): Point[] {
  return pts.map(p => [p[0], p[1]])
}
function clonePolygons(polys: Polygon[]): Polygon[] {
  return polys.map(poly => clonePoints(poly))
}

export const useEditorStore = create<EditorState>((set, get) => ({
  open: false,
  tool: 'select',
  zone: null,
  polygonIdx: null,
  mode: 'idle',
  points: [],
  drag: null,
  undo: [],
  zones: {},
  presence: {},
  w: 0,
  h: 0,
  dirty: false,
  didDrag: false,
  video: null,
  frameUrl: null,

  openEditor: async (video) => {
    const [infoRes, zonesRes, presenceRes] = await Promise.all([
      api.get<{ width: number; height: number }>(`/api/videos/${encodeURIComponent(video)}/info`),
      api.get<{ zones: Record<string, ZoneData> }>(`/api/zones/${encodeURIComponent(video)}`),
      api.get<{ zones: Record<string, PresenceInfo> }>(`/api/presence/${encodeURIComponent(video)}`),
    ])
    const firstZone = Object.keys(zonesRes.zones)[0] || null
    set({
      open: true,
      video,
      w: infoRes.width,
      h: infoRes.height,
      zones: zonesRes.zones || {},
      presence: presenceRes.zones || {},
      zone: firstZone,
      polygonIdx: null,
      tool: 'select',
      mode: 'idle',
      points: [],
      drag: null,
      undo: [],
      dirty: false,
      didDrag: false,
      frameUrl: `/api/videos/${encodeURIComponent(video)}/frame?t=${Date.now()}`,
    })
  },

  closeEditor: () => {
    const { video, zone, dirty } = get()
    if (video && zone && dirty) {
      const zones = get().zones
      const polygons = zones[zone]?.polygons || []
      api.put(`/api/zones/${encodeURIComponent(video)}/${encodeURIComponent(zone)}`, { polygons }).catch(() => {})
    }
    set({
      open: false,
      video: null,
      frameUrl: null,
      zone: null,
      polygonIdx: null,
      points: [],
      drag: null,
      undo: [],
      dirty: false,
    })
  },

  setTool: (tool) => {
    set({ tool, points: [], mode: 'idle' })
  },

  selectZone: (name) => {
    set({ zone: name, polygonIdx: null, mode: 'idle', points: [] })
  },

  selectPolygon: (idx) => set({ polygonIdx: idx }),

  addPoint: (p) => {
    get().pushUndo()
    set((s) => ({ points: [...s.points, p] }))
  },

  pushUndo: () => {
    const s = get()
    const snapshot: UndoSnapshot = {
      tool: s.tool,
      zone: s.zone,
      polygonIdx: s.polygonIdx,
      mode: s.mode,
      points: clonePoints(s.points),
      zonePolys: s.zone ? clonePolygons(s.zones[s.zone]?.polygons || []) : null,
    }
    const undo = [...s.undo, snapshot]
    if (undo.length > 50) undo.shift()
    set({ undo })
  },

  popUndo: () => {
    const s = get()
    if ((s.tool === 'include' || s.tool === 'exclude' || s.tool === 'countingROI') && s.points.length > 0) {
      set((st) => ({ points: st.points.slice(0, -1) }))
      return
    }
    const undo = [...s.undo]
    const snap = undo.pop()
    if (!snap) return
    const zones = { ...s.zones }
    if (snap.zone && snap.zonePolys && zones[snap.zone]) {
      zones[snap.zone] = { ...zones[snap.zone], polygons: clonePolygons(snap.zonePolys) }
    }
    set({
      undo,
      tool: snap.tool,
      zone: snap.zone,
      polygonIdx: snap.polygonIdx,
      mode: snap.mode as EditorState['mode'],
      points: clonePoints(snap.points),
      zones,
    })
  },

  clearTemp: () => {
    set({ mode: 'idle', points: [], drag: null, polygonIdx: null })
  },

  setDrag: (d) => set({ drag: d, didDrag: false }),

  updateVertex: (vIdx, p) => {
    const s = get()
    if (!s.zone || s.polygonIdx === null) return
    const zones = { ...s.zones }
    const zoneData = zones[s.zone]
    if (!zoneData) return
    const polys = clonePolygons(zoneData.polygons)
    const poly = polys[s.polygonIdx]
    if (!poly || !poly[vIdx]) return
    poly[vIdx] = [p[0], p[1]]
    zones[s.zone] = { ...zoneData, polygons: polys }
    set({ zones, didDrag: true, dirty: true })
  },

  movePolygon: (dx, dy) => {
    const s = get()
    if (!s.zone || s.polygonIdx === null) return
    const zones = { ...s.zones }
    const zoneData = zones[s.zone]
    if (!zoneData) return
    const polys = clonePolygons(zoneData.polygons)
    const poly = polys[s.polygonIdx]
    if (!poly) return
    for (let i = 0; i < poly.length; i++) {
      poly[i] = [poly[i][0] + dx, poly[i][1] + dy]
    }
    zones[s.zone] = { ...zoneData, polygons: polys }
    set({ zones, didDrag: true, dirty: true })
  },

  endDrag: () => {
    const s = get()
    set({ drag: null })
    if (s.didDrag) {
      set({ didDrag: false })
      const { video, zone, zones } = get()
      if (video && zone) {
        const polygons = zones[zone]?.polygons || []
        api.put(`/api/zones/${encodeURIComponent(video)}/${encodeURIComponent(zone)}`, { polygons }).catch(() => {})
      }
    }
  },

  deleteVertex: (vIdx) => {
    const s = get()
    if (!s.zone || s.polygonIdx === null) return
    const zones = { ...s.zones }
    const zoneData = zones[s.zone]
    if (!zoneData) return
    const polys = clonePolygons(zoneData.polygons)
    const poly = polys[s.polygonIdx]
    if (!poly || poly.length <= 3) return
    s.pushUndo()
    poly.splice(vIdx, 1)
    zones[s.zone] = { ...zoneData, polygons: polys }
    set({ zones, dirty: true })
  },

  deleteShape: async (idx) => {
    const s = get()
    if (!s.zone) return
    s.pushUndo()
    const zones = { ...s.zones }
    const zoneData = zones[s.zone]
    if (!zoneData) return
    const polys = [...zoneData.polygons]
    polys.splice(idx, 1)
    zones[s.zone] = { ...zoneData, polygons: polys }
    set({ zones, polygonIdx: null, dirty: true })
    if (s.video) {
      await api.put(`/api/zones/${encodeURIComponent(s.video)}/${encodeURIComponent(s.zone)}`, { polygons: polys }).catch(() => {})
    }
  },

  commitDraft: async () => {
    const s = get()
    if (!s.video) return

    if (s.tool === 'select') {
      if (!s.dirty) return
      if (s.zone) {
        const polygons = s.zones[s.zone]?.polygons || []
        await api.put(`/api/zones/${encodeURIComponent(s.video)}/${encodeURIComponent(s.zone)}`, { polygons })
        set({ dirty: false })
      }
      return
    }

    if ((s.tool === 'include' || s.tool === 'exclude' || s.tool === 'countingROI') && s.points.length >= 3) {
      let zone = s.zone
      if (!zone) {
        if (s.tool === 'countingROI') {
          zone = 'ROI Comptage'
          if (!s.zones[zone]) {
            set((st) => ({ zones: { ...st.zones, [zone!]: { polygons: [] } }, zone }))
          }
        } else {
          const keys = Object.keys(s.zones).sort()
          if (keys.length) {
            zone = keys[0]
            set({ zone })
          } else return
        }
      }
      const poly = clonePoints(s.points)
      await api.post('/api/zones', { name: zone, polygons: [poly], video: s.video })
      if (s.tool === 'countingROI' && zone) {
        await api.post(`/api/counting/${encodeURIComponent(s.video)}/config`, { zone_name: zone }).catch(() => {})
      }
      await get().refreshFromServer()
      set({ points: [], tool: 'select' })
    }
  },

  saveAll: async () => {
    const s = get()
    if (!s.video || !s.zone) return
    const polygons = s.zones[s.zone]?.polygons || []
    await api.put(`/api/zones/${encodeURIComponent(s.video)}/${encodeURIComponent(s.zone)}`, { polygons })
    set({ dirty: false })
    await get().refreshFromServer()
  },

  addZone: async (name) => {
    const s = get()
    if (!s.video || !name.trim()) return
    await api.post('/api/zones', { name: name.trim(), polygons: [], video: s.video })
    await get().refreshFromServer()
    set({ zone: name.trim() })
  },

  deleteZone: async () => {
    const s = get()
    if (!s.video || !s.zone) return
    await api.del(`/api/zones/${encodeURIComponent(s.video)}/${encodeURIComponent(s.zone)}`)
    set({ zone: null, polygonIdx: null })
    await get().refreshFromServer()
  },

  refreshFromServer: async () => {
    const { video } = get()
    if (!video) return
    const [zonesRes, presenceRes] = await Promise.all([
      api.get<{ zones: Record<string, ZoneData> }>(`/api/zones/${encodeURIComponent(video)}`),
      api.get<{ zones: Record<string, PresenceInfo> }>(`/api/presence/${encodeURIComponent(video)}`),
    ])
    set({ zones: zonesRes.zones || {}, presence: presenceRes.zones || {} })
  },
}))
