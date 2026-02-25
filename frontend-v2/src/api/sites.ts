import { apiPost } from './client'

export interface SiteCreatePayload {
  site_id: string
  name: string
  lieu_id: string
  address?: string
  description?: string
}

export async function createSite(payload: SiteCreatePayload): Promise<{ message: string; site_id: string }> {
  return apiPost<{ message: string; site_id: string }>('/sites', payload)
}
