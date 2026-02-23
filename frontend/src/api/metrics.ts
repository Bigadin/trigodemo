import { api } from './client'

export interface MetricsPoint {
  timestamp: number
  fps: number
  inference_ms: number
  cpu_percent: number
  memory_mb: number
}

export interface MetricsResponse {
  metrics: MetricsPoint[]
}

export const fetchMetrics = (points = 120) =>
  api.get<MetricsResponse>(`/api/metrics?points=${points}`)
