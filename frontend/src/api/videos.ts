import { api } from './client'
import type { VideosResponse, VideoInfo } from '@/types/api'

export const fetchVideos = () =>
  api.get<VideosResponse>('/api/videos')

export const fetchVideoInfo = (video: string) =>
  api.get<VideoInfo>(`/api/videos/${encodeURIComponent(video)}/info`)
