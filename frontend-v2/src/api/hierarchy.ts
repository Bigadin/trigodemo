import { apiGet } from './client'
import type { Hierarchy } from '@/types/hierarchy'

export async function fetchHierarchy(): Promise<Hierarchy> {
  const { hierarchy } = await apiGet<{ hierarchy: Hierarchy }>('/hierarchy')
  return hierarchy || {}
}
