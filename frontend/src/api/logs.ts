import { api } from './client'
import type { LogsResponse } from '@/types/api'

export const fetchLogs = (limit = 500, category?: string) => {
  const params = new URLSearchParams({ limit: String(limit) })
  if (category) params.set('category', category)
  return api.get<LogsResponse>(`/api/logs?${params}`)
}

export const clearLogs = () =>
  api.del('/api/logs')
