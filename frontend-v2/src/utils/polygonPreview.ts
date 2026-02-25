/** Génère un SVG de prévisualisation pour un polygone (zone) */
export function buildPolyPreviewSvg(
  poly: number[][],
  type: 'include' | 'exclude' = 'include'
): string {
  if (!poly || poly.length < 3) return ''
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of poly) {
    minX = Math.min(minX, p[0])
    minY = Math.min(minY, p[1])
    maxX = Math.max(maxX, p[0])
    maxY = Math.max(maxY, p[1])
  }
  const w = Math.max(1, maxX - minX)
  const h = Math.max(1, maxY - minY)
  const pad = 6
  const vw = 100, vh = 64
  const sx = (vw - pad * 2) / w
  const sy = (vh - pad * 2) / h
  const s = Math.min(sx, sy)
  const pts = poly.map((p) => {
    const x = (p[0] - minX) * s + pad
    const y = (p[1] - minY) * s + pad
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const colors = type === 'include'
    ? { fill: 'rgba(34,197,94,0.25)', stroke: '#22c55e' }
    : { fill: 'rgba(240,131,33,0.25)', stroke: '#f08321' }
  return `<svg viewBox="0 0 ${vw} ${vh}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"><polygon points="${pts}" fill="${colors.fill}" stroke="${colors.stroke}" stroke-width="2"/></svg>`
}
