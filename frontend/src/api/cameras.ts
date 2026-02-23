import { api } from './client'
import type { Camera } from '@/types/site'

export async function fetchCameras() {
  return api.get<{ cameras: Record<string, Camera> }>('/api/cameras')
}

export async function addCamera(id: string, data: Partial<Camera>) {
  return api.post('/api/cameras', { camera_id: id, ...data })
}

export async function deleteCamera(id: string) {
  return api.del(`/api/cameras/${id}`)
}
