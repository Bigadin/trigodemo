import { api } from './client'
import type {
  HierarchyLieu,
  Lieu,
  Site,
} from '@/types/site'

interface RawBenefit {
  benefit_id: string
  name: string
  skill: string
  skill_item: string
  categories: string[]
  camera_id: string
  active: boolean
  zone_polygons: number[][][]
  [k: string]: unknown
}

interface RawCamera {
  camera_id: string
  name: string
  type: string
  path: string
  site_id: string
  benefits: Record<string, RawBenefit>
  [k: string]: unknown
}

interface RawSite {
  site_id: string
  name: string
  lieu_id: string
  address: string
  description: string
  cameras: Record<string, RawCamera>
  [k: string]: unknown
}

interface RawLieu {
  lieu_id: string
  name: string
  address: string
  description: string
  sites: Record<string, RawSite>
  [k: string]: unknown
}

interface RawHierarchyResponse {
  hierarchy: Record<string, RawLieu>
}

function normalize(raw: RawHierarchyResponse): HierarchyLieu[] {
  return Object.entries(raw.hierarchy).map(([id, lieu]) => ({
    id,
    name: lieu.name,
    address: lieu.address,
    description: lieu.description,
    sites: Object.entries(lieu.sites).map(([sid, site]) => ({
      id: sid,
      name: site.name,
      lieu_id: site.lieu_id,
      address: site.address,
      description: site.description,
      cameras: Object.entries(site.cameras).map(([cid, cam]) => ({
        id: cid,
        name: cam.name,
        type: cam.type,
        path: cam.path,
        site_id: cam.site_id,
        benefits: Object.entries(cam.benefits).map(([bid, ben]) => ({
          id: bid,
          name: ben.name,
          skill: ben.skill,
          skill_item: ben.skill_item,
          categories: ben.categories,
          camera_id: ben.camera_id,
          active: ben.active,
          zone_polygons: ben.zone_polygons,
        })),
      })),
    })),
  }))
}

export async function fetchHierarchy(): Promise<HierarchyLieu[]> {
  const raw = await api.get<RawHierarchyResponse>('/api/hierarchy')
  return normalize(raw)
}

export async function createLieu(id: string, name: string) {
  return api.post('/api/lieux', { lieu_id: id, name })
}

export async function createSite(id: string, name: string, lieuId: string) {
  return api.post('/api/sites', { site_id: id, name, lieu_id: lieuId })
}

export async function updateSite(id: string, data: Partial<Site>) {
  return api.put(`/api/sites/${id}`, data)
}

export async function deleteSite(id: string) {
  return api.del(`/api/sites/${id}`)
}

export async function updateLieu(id: string, data: Partial<Lieu>) {
  return api.put(`/api/lieux/${id}`, data)
}

export async function deleteLieu(id: string) {
  return api.del(`/api/lieux/${id}`)
}
