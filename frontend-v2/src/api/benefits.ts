import { apiPut, apiPost, apiDelete } from './client'

/** Synchronise les zones des bénéfices (detection + counting) vers le backend pour zones_by_video */
export async function syncBenefitZones(): Promise<{ synced: number }> {
  return apiPost<{ synced: number }>('/benefits/sync-zones')
}

export async function toggleBenefit(benefitId: string, active: boolean) {
  return apiPut(`/benefits/${encodeURIComponent(benefitId)}`, { active })
}

export interface BenefitCreatePayload {
  benefit_id: string
  name: string
  skill: string
  skill_item: string
  categories: string[]
  camera_id: string
  zone_polygons?: number[][][]
  zone_polygon_types?: ('include' | 'exclude')[]
  zone_ref_width?: number
  zone_ref_height?: number
  active?: boolean
}

export interface BenefitUpdatePayload {
  name?: string
  skill?: string
  skill_item?: string
  categories?: string[]
  zone_polygons?: number[][][]
  zone_polygon_types?: ('include' | 'exclude')[]
  zone_ref_width?: number
  zone_ref_height?: number
  active?: boolean
}

export async function createBenefit(payload: BenefitCreatePayload) {
  return apiPost<{ message: string; benefit_id: string }>('/benefits', payload)
}

export async function updateBenefit(benefitId: string, payload: BenefitUpdatePayload) {
  return apiPut<{ message: string }>(`/benefits/${encodeURIComponent(benefitId)}`, payload)
}

export async function deleteBenefit(benefitId: string) {
  return apiDelete<{ message: string }>(`/benefits/${encodeURIComponent(benefitId)}`)
}
