import { api } from './client'
import type { Benefit } from '@/types/site'

export async function fetchBenefits() {
  return api.get<{ benefits: Record<string, Benefit> }>('/api/benefits')
}

export async function createBenefit(id: string, data: Partial<Benefit>) {
  return api.post('/api/benefits', { benefit_id: id, ...data })
}

export async function updateBenefit(id: string, data: Partial<Benefit>) {
  return api.put(`/api/benefits/${id}`, data)
}

export async function deleteBenefit(id: string) {
  return api.del(`/api/benefits/${id}`)
}
