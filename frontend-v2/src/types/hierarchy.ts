export interface HierarchyBenefit {
  benefit_id: string
  name: string
  skill?: string
  skill_item?: string
  categories?: string[]
  camera_id: string
  zone_polygons?: number[][][]
  /** Type de chaque polygone : 'include' | 'exclude' (sinon dérivé de idx % 2) */
  zone_polygon_types?: ('include' | 'exclude')[]
  /** Dimensions vidéo au moment du dessin (pour ratio cohérent) */
  zone_ref_width?: number
  zone_ref_height?: number
  active?: boolean
}

export interface HierarchyCamera {
  camera_id: string
  name: string
  type?: string
  path?: string
  site_id?: string
  benefits: Record<string, HierarchyBenefit>
}

export interface HierarchySite {
  site_id: string
  name: string
  lieu_id: string
  address?: string
  description?: string
  cameras: Record<string, HierarchyCamera>
}

export interface HierarchyLieu {
  lieu_id: string
  name: string
  address?: string
  description?: string
  icon?: string
  sites: Record<string, HierarchySite>
}

export type Hierarchy = Record<string, HierarchyLieu>
