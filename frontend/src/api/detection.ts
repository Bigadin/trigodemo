import { api } from './client'
import type { BlurState } from '@/types/api'

export const fetchBlurState = () =>
  api.get<BlurState>('/api/blur')

export const toggleBlur = () =>
  api.post('/api/blur/toggle')
