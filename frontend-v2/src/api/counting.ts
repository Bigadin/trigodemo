import { apiGet, apiPost, apiPut } from './client'

export interface CountingConfigResponse {
  configured: boolean
  zone_name: string | null
  flip_count: number
  mode: string
  angle: number | null
  zone_settings: Record<string, { mode: string; flip_count: number }>
  enabled: boolean
  count: number
}

export interface CountingParamsResponse {
  simple_gradient_threshold: number
  simple_cooldown_frames: number
  min_blob_area?: number
  mog2_history?: number
  mog2_var_threshold?: number
  mog2_detect_shadows?: boolean
  [key: string]: unknown
}

function countingPath(videoPath: string): string {
  return `/counting/${encodeURIComponent(videoPath)}`
}

export async function fetchCounting(videoPath: string): Promise<CountingConfigResponse> {
  return apiGet<CountingConfigResponse>(countingPath(videoPath))
}

export async function configCounting(
  videoPath: string,
  body: { zone_name: string; mode: string }
): Promise<{ message: string; zone_name: string; mode: string; flip_count: number }> {
  return apiPost(countingPath(videoPath) + '/config', body)
}

export async function toggleCounting(
  videoPath: string
): Promise<{ enabled: boolean; count: number; mode: string }> {
  return apiPost(countingPath(videoPath) + '/toggle')
}

export async function flipCounting(videoPath: string): Promise<{ flip_count: number }> {
  return apiPost(countingPath(videoPath) + '/flip')
}

export async function resetCounting(videoPath: string): Promise<{ count: number }> {
  return apiPost(countingPath(videoPath) + '/reset')
}

export async function fetchCountingParams(): Promise<CountingParamsResponse> {
  return apiGet<CountingParamsResponse>('/counting/params')
}

export async function updateCountingParams(
  params: Partial<{ simple_gradient_threshold: number; simple_cooldown_frames: number }>
): Promise<CountingParamsResponse> {
  return apiPut<CountingParamsResponse>('/counting/params', params)
}
