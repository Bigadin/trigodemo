import { apiGet } from './client'

const DEFAULT_SKILLS_CONFIG: SkillsConfig = {
  skills: [
    {
      key: 'detection',
      label: 'Detection',
      icon: '/static/assets_youn/SvIcons/SVGnew/Yclassify.svg',
      items: [
        { id: 'detection_presence', label: 'Presence / Absence', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' },
        { id: 'detection_linecross', label: 'Franchissement ligne', icon: '/static/assets_youn/SvIcons/SVGnew/Ylinecross.svg' },
        { id: 'detection_zone', label: 'Detection zone', icon: '/static/assets_youn/SvIcons/SVGnew/Yzonedetect.svg' },
      ],
    },
    {
      key: 'counting',
      label: 'Comptage',
      icon: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg',
      items: [
        { id: 'counting_people', label: 'Comptage personnes', icon: '/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg' },
        { id: 'counting_objects', label: 'Comptage objets', icon: '/static/assets_youn/SvIcons/SVGnew/Ycounting.svg' },
        { id: 'counting_zone', label: 'Comptage zone', icon: '/static/assets_youn/SvIcons/SVGnew/square-area-svgrepo-com.svg' },
      ],
    },
    {
      key: 'heatmap',
      label: 'Heatmap',
      icon: '/static/assets_youn/SvIcons/SVGnew/Yheatmap.svg',
      items: [
        { id: 'heatmap_density', label: 'Densite de flux', icon: '/static/assets_youn/SvIcons/SVGnew/grid-svgrepo-com.svg' },
        { id: 'heatmap_presence', label: 'Heatmap presence', icon: '/static/assets_youn/SvIcons/SVGnew/Yheatmapdense.svg' },
        { id: 'heatmap_trajectory', label: 'Heatmap trajectoires', icon: '/static/assets_youn/SvIcons/SVGnew/Ytraj.svg' },
      ],
    },
    {
      key: 'quality',
      label: 'Qualite',
      icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg',
      items: [
        { id: 'quality_fissure', label: 'Fissure', icon: '/static/assets_youn/SvIcons/SVGnew/Yfissure.svg' },
        { id: 'quality_humidity', label: 'Humidite', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg' },
        { id: 'quality_check', label: 'Qualite generale', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg' },
      ],
    },
  ],
  categories_by_skill: {
    detection: [
      {
        key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
        items: [
          { id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' },
          { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' },
          { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' },
        ],
      },
      {
        key: 'transport', label: 'Transport', icon: '/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg',
        items: [
          { id: 'voiture', label: 'Voiture', icon: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg' },
          { id: 'velo', label: 'Velo', icon: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg' },
          { id: 'public_transport', label: 'Transport public', icon: '/static/assets_youn/SvIcons/SVGnew/Ypublic transport.svg' },
          { id: 'avion', label: 'Avion', icon: '/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg' },
          { id: 'moto', label: 'Moto', icon: '/static/assets_youn/SvIcons/SVGnew/motorcycle.svg' },
        ],
      },
    ],
    counting: [
      {
        key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
        items: [
          { id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' },
          { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' },
          { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' },
        ],
      },
      {
        key: 'transport', label: 'Transport', icon: '/static/assets_youn/SvIcons/SVGnew/Ytransport2.svg',
        items: [
          { id: 'voiture', label: 'Voiture', icon: '/static/assets_youn/SvIcons/SVGnew/Ycar.svg' },
          { id: 'velo', label: 'Velo', icon: '/static/assets_youn/SvIcons/SVGnew/Ybike.svg' },
          { id: 'public_transport', label: 'Transport public', icon: '/static/assets_youn/SvIcons/SVGnew/Ypublic transport.svg' },
          { id: 'avion', label: 'Avion', icon: '/static/assets_youn/SvIcons/SVGnew/plane-svgrepo-com.svg' },
          { id: 'moto', label: 'Moto', icon: '/static/assets_youn/SvIcons/SVGnew/motorcycle.svg' },
        ],
      },
    ],
    heatmap: [
      {
        key: 'human', label: 'Humain', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg',
        items: [
          { id: 'silhouette', label: 'Silhouette', icon: '/static/assets_youn/SvIcons/SVGnew/Ysilhouette.svg' },
          { id: 'visage', label: 'Visage', icon: '/static/assets_youn/SvIcons/SVGnew/Yface.svg' },
          { id: 'foule', label: 'Foule', icon: '/static/assets_youn/SvIcons/SVGnew/Ycrowdfoule.svg' },
        ],
      },
      {
        key: 'object', label: 'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg',
        items: [
          { id: 'encombrement', label: 'Encombrement', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' },
          { id: 'zone_encombre', label: 'Zone encombre', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' },
          { id: 'fissure', label: 'Fissure', icon: '/static/assets_youn/SvIcons/SVGnew/Yfissure.svg' },
          { id: 'humidity', label: 'Humidite', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg' },
          { id: 'qualitycheck', label: 'Qualite generale', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg' },
        ],
      },
    ],
    quality: [
      {
        key: 'object', label: 'Objet', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg',
        items: [
          { id: 'encombrement', label: 'Encombrement', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' },
          { id: 'zone_encombre', label: 'Zone encombre', icon: '/static/assets_youn/SvIcons/SVGnew/Yobstruction.svg' },
          { id: 'fissure', label: 'Fissure', icon: '/static/assets_youn/SvIcons/SVGnew/Yfissure.svg' },
          { id: 'humidity', label: 'Humidite', icon: '/static/assets_youn/SvIcons/SVGnew/Yhumidity.svg' },
          { id: 'qualitycheck', label: 'Qualite generale', icon: '/static/assets_youn/SvIcons/SVGnew/Yqualitycheck.svg' },
        ],
      },
    ],
  },
}

const FALLBACK_ICON = '/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg'

export interface SkillItem {
  id: string
  label: string
  icon: string
}

export interface SkillGroup {
  key: string
  label: string
  icon: string
  items: SkillItem[]
}

export interface CategoryItem {
  id: string
  label: string
  icon: string
}

export interface CategoryGroup {
  key: string
  label: string
  icon: string
  items: CategoryItem[]
}

export interface SkillsConfig {
  skills: SkillGroup[]
  categories_by_skill: Record<string, CategoryGroup[]>
}

let skillsCache: SkillsConfig | null = null

export async function loadSkillsConfig(): Promise<SkillsConfig> {
  try {
    const data = await apiGet<SkillsConfig>('/skills')
    if (data?.skills?.length >= 1 && data?.categories_by_skill && Object.keys(data.categories_by_skill).length >= 1) {
      skillsCache = data
      return data
    }
  } catch (e) {
    console.warn('[skills] /api/skills non disponible, fallback config:', (e as Error).message)
  }
  skillsCache = DEFAULT_SKILLS_CONFIG
  return DEFAULT_SKILLS_CONFIG
}

export function getSkillsConfig(): SkillsConfig {
  return skillsCache ?? DEFAULT_SKILLS_CONFIG
}

export function getSkillGroups(): SkillGroup[] {
  return getSkillsConfig().skills ?? DEFAULT_SKILLS_CONFIG.skills
}

export function getCategoryGroupsBySkill(skill: string): CategoryGroup[] {
  const cats = getSkillsConfig().categories_by_skill?.[skill]
  return Array.isArray(cats) && cats.length > 0 ? cats : []
}

export function getSkillIcon(skill: string): string {
  const g = getSkillGroups().find((s) => s.key === skill)
  return g?.icon ?? FALLBACK_ICON
}
