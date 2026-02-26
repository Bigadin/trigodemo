import { apiGet, apiDelete } from './client'

export interface AuditLogEntry {
  ts: string
  category?: string
  action?: string
  detail?: string
  level?: string
  meta?: Record<string, unknown>
}

export interface LogsResponse {
  logs: AuditLogEntry[]
  total: number
}

export async function fetchLogs(limit = 500, category = ''): Promise<LogsResponse> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (category) params.set('category', category)
  return apiGet<LogsResponse>(`/logs?${params}`)
}

export async function clearLogs(): Promise<{ message: string }> {
  return apiDelete<{ message: string }>('/logs')
}
