import { create } from 'zustand'
import {
  fetchCounting,
  resetCounting,
  configCounting,
  toggleCounting,
  flipCounting,
  fetchCountingParams,
  updateCountingParams,
} from '@/api/counting'

export interface CountingConfig {
  configured: boolean
  zone_name: string | null
  flip_count: number
  mode: string
  angle: number | null
  zone_settings: Record<string, { mode: string; flip_count: number }>
  enabled: boolean
  count: number
}

interface CountingParamsData {
  simple_gradient_threshold: number
  simple_cooldown_frames: number
}

interface CountingStore {
  configs: Record<string, CountingConfig>
  params: CountingParamsData | null

  refresh: (video: string) => Promise<void>
  toggle: (video: string) => Promise<void>
  flip: (video: string) => Promise<void>
  reset: (video: string) => Promise<void>
  configure: (video: string, zone_name: string, mode: string) => Promise<void>

  loadParams: () => Promise<void>
  saveParams: (p: Partial<CountingParamsData>) => Promise<void>
}

export const useCountingStore = create<CountingStore>((set, get) => ({
  configs: {},
  params: null,

  refresh: async (video) => {
    try {
      const res = await fetchCounting(video)
      set((s) => ({ configs: { ...s.configs, [video]: res as unknown as CountingConfig } }))
    } catch { /* ignore */ }
  },

  toggle: async (video) => {
    await toggleCounting(video)
    await get().refresh(video)
  },

  flip: async (video) => {
    await flipCounting(video)
    await get().refresh(video)
  },

  reset: async (video) => {
    await resetCounting(video)
    await get().refresh(video)
  },

  configure: async (video, zone_name, mode) => {
    await configCounting(video, { zone_name, mode })
    await get().refresh(video)
  },

  loadParams: async () => {
    try {
      const res = await fetchCountingParams()
      set({ params: res })
    } catch { /* ignore */ }
  },

  saveParams: async (p) => {
    await updateCountingParams(p)
    await get().loadParams()
  },
}))
