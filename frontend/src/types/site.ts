export interface Lieu {
  name: string
  address: string
  description: string
  icon: string
  created_at: string
}

export interface Site {
  name: string
  lieu_id: string
  address: string
  description: string
  icon: string
  created_at: string
}

export interface Camera {
  type: 'video' | 'webcam' | 'rtsp'
  name: string
  path: string
  site_id: string
  device_id?: number
  url?: string
}

export interface Benefit {
  name: string
  skill: 'detection' | 'counting' | 'heatmap' | 'quality'
  skill_item: string
  categories: string[]
  camera_id: string
  zone_polygons: number[][][]
  active: boolean
  canvas: Record<string, unknown>
  created_at: string
}

export interface HierarchyLieu {
  id: string
  name: string
  address: string
  description: string
  sites: HierarchySite[]
}

export interface HierarchySite {
  id: string
  name: string
  lieu_id: string
  address: string
  description: string
  cameras: HierarchyCamera[]
}

export interface HierarchyCamera {
  id: string
  name: string
  type: string
  path: string
  site_id: string
  benefits: HierarchyBenefit[]
}

export interface HierarchyBenefit {
  id: string
  name: string
  skill: string
  skill_item: string
  categories: string[]
  camera_id: string
  active: boolean
  zone_polygons: number[][][]
}

export interface HierarchyResponse {
  hierarchy: HierarchyLieu[]
}
