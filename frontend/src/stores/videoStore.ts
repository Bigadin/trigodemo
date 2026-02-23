import { create } from 'zustand'
import { startStream, stopStream, stopAllStreams, fetchStreams } from '@/api/streams'
import { fetchZones, fetchPresence, type ZonesApiResponse } from '@/api/zones'

export interface PresenceData {
  occupied: boolean
  total_time: number
}

interface VideoState {
  currentVideo: string | null
  currentCameraId: string | null

  activeStreams: Set<string>
  zones: Record<string, ZonesApiResponse['zones']>
  presence: Record<string, Record<string, PresenceData>>
  blurEnabled: boolean

  setCurrentVideo: (video: string | null) => void
  setCurrentCamera: (id: string | null) => void

  refreshStreams: () => Promise<void>
  refreshZones: (video: string) => Promise<void>
  refreshPresence: (video: string) => Promise<void>

  startDetection: (video: string) => Promise<void>
  stopDetection: (video: string) => Promise<void>
  stopAll: () => Promise<void>

  isStreaming: (video: string) => boolean
}

export const useVideoStore = create<VideoState>((set, get) => ({
  currentVideo: null,
  currentCameraId: null,
  activeStreams: new Set(),
  zones: {},
  presence: {},
  blurEnabled: false,

  setCurrentVideo: (video) => set({ currentVideo: video }),
  setCurrentCamera: (id) => set({ currentCameraId: id }),

  refreshStreams: async () => {
    try {
      const res = await fetchStreams()
      set({ activeStreams: new Set(res.streams.map((s) => s.video)) })
    } catch { /* ignore */ }
  },

  refreshZones: async (video) => {
    try {
      const res = await fetchZones(video)
      set((s) => ({ zones: { ...s.zones, [video]: res.zones } }))
    } catch { /* ignore */ }
  },

  refreshPresence: async (video) => {
    try {
      const res = await fetchPresence(video)
      const mapped: Record<string, PresenceData> = {}
      for (const [k, v] of Object.entries(res.zones)) {
        mapped[k] = { occupied: v.occupied, total_time: v.total_time }
      }
      set((s) => ({ presence: { ...s.presence, [video]: mapped } }))
    } catch { /* ignore */ }
  },

  startDetection: async (video) => {
    await startStream(video)
    set((s) => {
      const next = new Set(s.activeStreams)
      next.add(video)
      return { activeStreams: next }
    })
  },

  stopDetection: async (video) => {
    await stopStream(video)
    set((s) => {
      const next = new Set(s.activeStreams)
      next.delete(video)
      return { activeStreams: next }
    })
  },

  stopAll: async () => {
    await stopAllStreams()
    set({ activeStreams: new Set() })
  },

  isStreaming: (video) => get().activeStreams.has(video),
}))
