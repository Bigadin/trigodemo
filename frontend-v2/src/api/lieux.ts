import { apiPost } from './client'

export interface LieuCreatePayload {
  lieu_id: string
  name: string
  address?: string
  description?: string
  icon?: string
}

export async function createLieu(payload: LieuCreatePayload): Promise<{ message: string; lieu_id: string }> {
  return apiPost<{ message: string; lieu_id: string }>('/lieux', payload)
}
