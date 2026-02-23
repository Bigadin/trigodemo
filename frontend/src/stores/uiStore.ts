import { create } from 'zustand'

type ViewMode = 'grid' | 'list'

interface CollapsedState {
  [nodeId: string]: boolean
}

interface UiState {
  siteViewMode: ViewMode
  setSiteViewMode: (m: ViewMode) => void

  collapsed: CollapsedState
  toggleCollapsed: (nodeId: string) => void
  isCollapsed: (nodeId: string) => boolean

  sidebarWidth: number
  setSidebarWidth: (w: number) => void
}

export const useUiStore = create<UiState>((set, get) => ({
  siteViewMode: 'grid',
  setSiteViewMode: (m) => set({ siteViewMode: m }),

  collapsed: {},
  toggleCollapsed: (nodeId) =>
    set((s) => ({
      collapsed: { ...s.collapsed, [nodeId]: !s.collapsed[nodeId] },
    })),
  isCollapsed: (nodeId) => !!get().collapsed[nodeId],

  sidebarWidth: 260,
  setSidebarWidth: (w) => set({ sidebarWidth: w }),
}))
