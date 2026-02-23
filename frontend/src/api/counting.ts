import { api } from './client'
import type { CountingState } from '@/types/api'

export const fetchCounting = (video: string) =>
  api.get<CountingState>(`/api/counting/${encodeURIComponent(video)}`)

export const resetCounting = (video: string) =>
  api.post(`/api/counting/${encodeURIComponent(video)}/reset`)

export const configCounting = (video: string, body: { zone_name?: string; mode?: string }) =>
  api.post(`/api/counting/${encodeURIComponent(video)}/config`, body)

export const toggleCounting = (video: string) =>
  api.post(`/api/counting/${encodeURIComponent(video)}/toggle`)

export const flipCounting = (video: string) =>
  api.post(`/api/counting/${encodeURIComponent(video)}/flip`)

export const fetchCountingParams = () =>
  api.get<{ simple_gradient_threshold: number; simple_cooldown_frames: number }>('/api/counting/params')

export const updateCountingParams = (body: { simple_gradient_threshold?: number; simple_cooldown_frames?: number }) =>
  api.put('/api/counting/params', body)
