import { apiGet, apiPost } from './client'

export interface ZoneData {
  polygons: number[][][]
  total_time?: number
  is_occupied?: boolean
}

export interface ZonesResponse {
  zones: Record<string, ZoneData>
}

export interface PresenceZone {
  is_occupied: boolean
  total_time?: number
}

export interface PresenceResponse {
  zones: Record<string, PresenceZone>
}

export interface StreamInfo {
  video: string
  active: boolean
}

export interface StreamsResponse {
  streams: StreamInfo[]
}

export function getZonesUrl(videoPath: string): string {
  return `/api/zones/${encodeURIComponent(videoPath)}`
}

export function getPresenceUrl(videoPath: string): string {
  return `/api/presence/${encodeURIComponent(videoPath)}`
}

export function getFrameUrl(videoPath: string): string {
  return `/api/videos/${encodeURIComponent(videoPath)}/frame?t=${Date.now()}`
}

export interface VideoInfo {
  width: number
  height: number
  fps?: number
  frame_count?: number
  duration?: number
  is_live?: boolean
}

export async function fetchVideoInfo(videoPath: string): Promise<VideoInfo> {
  return apiGet<VideoInfo>(`/videos/${encodeURIComponent(videoPath)}/info`)
}

export function getStreamUrl(videoPath: string, overlay = true): string {
  return `/api/stream/${encodeURIComponent(videoPath)}${overlay ? '?overlay=true' : ''}`
}

export async function fetchZones(videoPath: string): Promise<ZonesResponse> {
  return apiGet<ZonesResponse>(`/zones/${encodeURIComponent(videoPath)}`)
}

export async function fetchPresence(videoPath: string): Promise<PresenceResponse> {
  return apiGet<PresenceResponse>(`/presence/${encodeURIComponent(videoPath)}`)
}

export async function fetchStreams(): Promise<StreamsResponse> {
  return apiGet<StreamsResponse>('/streams')
}

export async function startStream(videoPath: string): Promise<{ message: string }> {
  return apiPost<{ message: string }>(`/stream/${encodeURIComponent(videoPath)}/start`)
}

export async function stopStream(videoPath: string): Promise<{ message: string }> {
  return apiPost<{ message: string }>(`/stream/${encodeURIComponent(videoPath)}/stop`)
}

export async function stopAllStreams(): Promise<{ stopped: number }> {
  return apiPost<{ stopped: number }>('/streams/stop')
}

export type { CountingConfigResponse as CountingResponse } from './counting'
export { fetchCounting } from './counting'

export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  track_id: number | null
}

export interface DetectionsResponse {
  detections: Detection[]
  active: boolean
}

export async function fetchDetections(videoPath: string): Promise<DetectionsResponse> {
  return apiGet<DetectionsResponse>(`/detections/${encodeURIComponent(videoPath)}`)
}

/** Réinitialise le timer de présence d'une zone (carte détection) */
export async function resetZoneTimer(zoneName: string): Promise<{ message: string }> {
  return apiPost<{ message: string }>(`/zones/reset/${encodeURIComponent(zoneName)}`)
}
