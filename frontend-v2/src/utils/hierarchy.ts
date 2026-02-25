import type { Hierarchy, HierarchyLieu, HierarchySite, HierarchyCamera, HierarchyBenefit } from '@/types/hierarchy'

/** Trouve un lieu par son ID */
export function findLieuById(hierarchy: Hierarchy, lieuId: string): HierarchyLieu | null {
  const lieu = hierarchy[lieuId]
  return lieu ?? null
}

/** Première caméra d'un site (pour navigation) */
export function getFirstCameraId(site: HierarchySite): string | null {
  const ids = Object.keys(site.cameras || {})
  return ids[0] ?? null
}

export const countForms = (site: HierarchySite): number =>
  Object.values(site.cameras || {}).reduce(
    (n, cam) => n + Object.values(cam.benefits || {}).reduce((m, b) => m + (b.zone_polygons?.length ?? 0), 0),
    0,
  )

/** Tous les bénéfices d'un site (toutes caméras) */
export function getSiteBenefits(site: HierarchySite): HierarchyBenefit[] {
  const list: HierarchyBenefit[] = []
  for (const cam of Object.values(site.cameras || {})) {
    for (const [bid, b] of Object.entries(cam.benefits || {})) {
      list.push({ ...b, benefit_id: b.benefit_id || bid })
    }
  }
  return list
}

/** Trouve un site par son ID dans la hiérarchie */
export function findSiteById(hierarchy: Hierarchy, siteId: string): HierarchySite | null {
  for (const lieu of Object.values(hierarchy)) {
    const site = lieu.sites?.[siteId]
    if (site) return site
  }
  return null
}

/** Retourne une copie de la hiérarchie avec un bénéfice fusionné (optimistic update) */
export function mergeBenefitIntoHierarchy(
  hierarchy: Hierarchy,
  cameraId: string,
  benefit: HierarchyBenefit
): Hierarchy {
  if (!hierarchy || !cameraId || !benefit?.benefit_id) return hierarchy
  const next = JSON.parse(JSON.stringify(hierarchy)) as Hierarchy
  for (const lieu of Object.values(next)) {
    for (const site of Object.values(lieu.sites || {})) {
      const cam = site.cameras?.[cameraId]
      if (cam) {
        if (!cam.benefits) cam.benefits = {}
        cam.benefits[benefit.benefit_id] = { ...benefit }
        return next
      }
    }
  }
  return next
}

/** Retourne le chemin vidéo pour l'API (fichier ou camera:xxx) */
export function getCameraVideoPath(cam: HierarchyCamera): string {
  if (cam.type === 'webcam' || cam.type === 'rtsp') {
    return `camera:${cam.camera_id}`
  }
  return cam.path || ''
}
