import { api } from './client'
import type { PresenceResponse } from '@/types/zone'

export interface ZonesApiResponse {
  zones: Record<string, {
    polygons: number[][][]
    type?: string
    line_meta?: { direction: number; flip: boolean }
  }>
}

export const fetchZones = (video: string) =>
  api.get<ZonesApiResponse>(`/api/zones/${encodeURIComponent(video)}`)

export const fetchPresence = (video: string) =>
  api.get<PresenceResponse>(`/api/presence/${encodeURIComponent(video)}`)
