/** Icônes SVG UI — /static/icons/ */
const ICONS_BASE = '/static/icons'

/** Mapping icônes (alias → fichier .svg) */
const ICON_MAP: Record<string, string> = {
  location: 'location.svg',
  site: 'site.svg',
  camera: 'camera.svg',
  folder: 'folder.svg',
  desktop: 'desktop.svg',
  chart: 'chart.svg',
  terminal: 'terminal.svg',
  zone: 'zone.svg',
  play: '/static/assets_youn/SvIcons/play-svgrepo-com.svg',
  stop: '/static/assets_youn/SvIcons/stop-svgrepo-com.svg',
  check: '/static/assets_youn/SvIcons/checkmark-svgrepo-com.svg',
  // Rétrocompat
  'Lieux.svg': 'location.svg',
  'Site.svg': 'site.svg',
  'camera.svg': 'camera.svg',
  'folder-svgrepo-com.svg': 'folder.svg',
  'desktop-svgrepo-com.svg': 'desktop.svg',
  'chart-line-svgrepo-com.svg': 'chart.svg',
  'terminal-svgrepo-com.svg': 'terminal.svg',
  'zone.svg': 'zone.svg',
}

export const LOC_COLORS: Record<string, string> = {
  'og logistics': '#5a8fb8',
  'galerie westfield': '#8b7fb5',
  default: '#6b9b7a',
}

export const getLocColor = (loc: string) =>
  LOC_COLORS[loc.toLowerCase()] ?? LOC_COLORS.default

/** Images miniatures des lieux (lieu_id → chemin) — fallback si icon vide dans l'API */
export const LIEU_ICONS: Record<string, string> = {
  usine: '/static/assets_youn/OG.jpg',
  mall: '/static/assets_youn/westfield.jpg',
}

export const getLieuIcon = (lieuId: string, iconFromApi?: string) =>
  (iconFromApi && iconFromApi.trim()) || LIEU_ICONS[lieuId] || ''

export const icon = (name: string) => {
  const mapped = ICON_MAP[name]
  if (mapped?.startsWith('/')) return mapped
  const file = mapped ?? (name.endsWith('.svg') ? name : `${name}.svg`)
  return `${ICONS_BASE}/${file}`
}

/** Images (logos, textures) — noms avec contexte (ex. arcy-logo-white) */
const IMAGES_BASE = '/static/images'
const IMAGE_MAP: Record<string, string> = {
  'arcy-logo': 'arcy-logo.png',
  'arcy-logo-white': 'arcy-logo-white.svg',
  'bg-texture-light': 'bg-texture-light.png',
}

export const img = (name: string) => {
  const file = IMAGE_MAP[name] ?? name
  return `${IMAGES_BASE}/${file}`
}
