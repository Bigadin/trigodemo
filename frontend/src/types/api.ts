export interface ApiError {
  detail: string
}

export interface StreamInfo {
  video: string
}

export interface StreamsResponse {
  streams: StreamInfo[]
}

export interface VideosResponse {
  videos: string[]
}

export interface VideoInfo {
  width: number
  height: number
  fps?: number
  duration?: number
}

export interface LogEntry {
  timestamp: string
  level: string
  category: string
  message: string
}

export interface LogsResponse {
  logs: LogEntry[]
  total: number
}

export interface BlurState {
  enabled: boolean
}

export interface CountingState {
  count: number
  zone_name: string
  mode: string
  active: boolean
  direction?: number
}
