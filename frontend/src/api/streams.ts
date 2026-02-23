import { api } from './client'
import type { StreamsResponse } from '@/types/api'

export const fetchStreams = () =>
  api.get<StreamsResponse>('/api/streams')

export const startStream = (video: string) =>
  api.post(`/api/stream/${encodeURIComponent(video)}/start`)

export const stopStream = (video: string) =>
  api.post(`/api/stream/${encodeURIComponent(video)}/stop`)

export const stopAllStreams = () =>
  api.post('/api/streams/stop')
