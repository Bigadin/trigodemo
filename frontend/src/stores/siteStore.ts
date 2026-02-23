import { create } from 'zustand'
import type {
  HierarchyLieu,
  HierarchySite,
  HierarchyCamera,
} from '@/types/site'
import { fetchHierarchy, createLieu, createSite, deleteSite as apiDeleteSite } from '@/api/hierarchy'

interface SiteState {
  lieux: HierarchyLieu[]
  loading: boolean
  error: string | null

  selectedLieuId: string | null
  selectedSiteId: string | null
  selectedCameraId: string | null

  loadHierarchy: () => Promise<void>

  selectLieu: (id: string | null) => void
  selectSite: (id: string | null) => void
  selectCamera: (id: string | null) => void

  addLieu: (id: string, name: string) => Promise<void>
  addSite: (id: string, name: string, lieuId: string) => Promise<void>
  removeSite: (id: string) => Promise<void>

  getLieu: (id: string) => HierarchyLieu | undefined
  getSite: (id: string) => HierarchySite | undefined
  getCamera: (id: string) => HierarchyCamera | undefined
  getSiteForCamera: (cameraId: string) => HierarchySite | undefined
  getLieuForSite: (siteId: string) => HierarchyLieu | undefined
}

export const useSiteStore = create<SiteState>((set, get) => ({
  lieux: [],
  loading: false,
  error: null,

  selectedLieuId: null,
  selectedSiteId: null,
  selectedCameraId: null,

  loadHierarchy: async () => {
    set({ loading: true, error: null })
    try {
      const lieux = await fetchHierarchy()
      set({ lieux, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  selectLieu: (id) => set({ selectedLieuId: id }),
  selectSite: (id) => set({ selectedSiteId: id }),
  selectCamera: (id) => set({ selectedCameraId: id }),

  addLieu: async (id, name) => {
    await createLieu(id, name)
    await get().loadHierarchy()
  },

  addSite: async (id, name, lieuId) => {
    await createSite(id, name, lieuId)
    await get().loadHierarchy()
  },

  removeSite: async (id) => {
    await apiDeleteSite(id)
    const s = get()
    if (s.selectedSiteId === id) set({ selectedSiteId: null, selectedCameraId: null })
    await s.loadHierarchy()
  },

  getLieu: (id) => get().lieux.find((l) => l.id === id),
  getSite: (id) => {
    for (const l of get().lieux)
      for (const s of l.sites)
        if (s.id === id) return s
    return undefined
  },
  getCamera: (id) => {
    for (const l of get().lieux)
      for (const s of l.sites)
        for (const c of s.cameras)
          if (c.id === id) return c
    return undefined
  },
  getSiteForCamera: (cameraId) => {
    for (const l of get().lieux)
      for (const s of l.sites)
        for (const c of s.cameras)
          if (c.id === cameraId) return s
    return undefined
  },
  getLieuForSite: (siteId) => {
    for (const l of get().lieux)
      for (const s of l.sites)
        if (s.id === siteId) return l
    return undefined
  },
}))
