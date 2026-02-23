export type Point = [number, number]
export type Polygon = Point[]

export interface ZoneDefinition {
  polygons: Polygon[]
  type?: 'include' | 'exclude'
  line_meta?: LineMeta
}

export interface LineMeta {
  direction: number
  flip: boolean
}

export interface ZonesResponse {
  zones: Record<string, ZoneDefinition>
}

export interface PresenceZone {
  occupied: boolean
  total_time: number
  last_occupied: string | null
}

export interface PresenceResponse {
  zones: Record<string, PresenceZone>
}
