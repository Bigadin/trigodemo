import type { Detection } from '@/api/tracker'

/**
 * Point-in-polygon test (ray casting algorithm).
 * Returns true if point (px, py) is inside the polygon.
 */
function pointInPolygon(px: number, py: number, polygon: number[][]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1]
    const xj = polygon[j][0], yj = polygon[j][1]
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }
  return inside
}

/**
 * Compute the area of intersection between a bbox and a polygon
 * using a sampling approach (grid of points).
 * Returns a ratio [0..1] of how much of the bbox is inside the polygon.
 */
function bboxPolygonOverlap(
  x1: number, y1: number, x2: number, y2: number,
  polygon: number[][],
  gridSize = 5,
): number {
  const w = x2 - x1
  const h = y2 - y1
  if (w <= 0 || h <= 0) return 0

  let inside = 0
  const total = gridSize * gridSize
  for (let gx = 0; gx < gridSize; gx++) {
    for (let gy = 0; gy < gridSize; gy++) {
      const px = x1 + (w * (gx + 0.5)) / gridSize
      const py = y1 + (h * (gy + 0.5)) / gridSize
      if (pointInPolygon(px, py, polygon)) inside++
    }
  }
  return inside / total
}

export interface ZoneOccupancyResult {
  /** Is at least one bbox mostly inside this zone polygon? */
  occupied: boolean
  /** Number of bboxes overlapping this zone */
  count: number
}

/**
 * For each zone polygon, check if at least one detection bbox is mostly inside.
 *
 * @param polygons   - Array of zone polygons (each polygon = array of [x,y] points)
 * @param types      - 'include' or 'exclude' for each polygon (only include zones trigger occupation)
 * @param detections - Current YOLO detection bboxes
 * @param scaleX     - Scale factor to convert detection coords → zone coords (detectionX * scaleX = zoneX)
 * @param scaleY     - Scale factor Y
 * @param threshold  - Min overlap ratio to consider a bbox "inside" (default 0.4 = 40%)
 */
export function computeZoneOccupancy(
  polygons: number[][][],
  types: ('include' | 'exclude')[] | undefined,
  detections: Detection[],
  scaleX: number,
  scaleY: number,
  threshold = 0.4,
): ZoneOccupancyResult[] {
  return polygons.map((poly, idx) => {
    const isInclude = (types?.[idx] ?? 'include') === 'include'
    if (!isInclude || !poly || poly.length < 3) {
      return { occupied: false, count: 0 }
    }
    let count = 0
    for (const det of detections) {
      const bx1 = det.x1 * scaleX
      const by1 = det.y1 * scaleY
      const bx2 = det.x2 * scaleX
      const by2 = det.y2 * scaleY
      const overlap = bboxPolygonOverlap(bx1, by1, bx2, by2, poly)
      if (overlap >= threshold) count++
    }
    return { occupied: count > 0, count }
  })
}
