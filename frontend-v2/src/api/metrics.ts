import { apiGet } from './client'

export interface MetricSeries {
  unit?: string
  color: string
  min?: number
  max?: number
  data: { v: number }[]
}

export type MetricsData = Record<string, MetricSeries>

export async function fetchMetrics(points = 120): Promise<MetricsData> {
  return apiGet<MetricsData>(`/metrics?points=${points}`)
}
