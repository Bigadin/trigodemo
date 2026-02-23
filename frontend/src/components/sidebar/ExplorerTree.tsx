import { useNavigate } from 'react-router-dom'
import { useSiteStore } from '@/stores/siteStore'
import { useUiStore } from '@/stores/uiStore'
import type { HierarchyLieu, HierarchySite, HierarchyCamera, HierarchyBenefit } from '@/types/site'
import styles from './ExplorerTree.module.css'

export default function ExplorerTree() {
  const lieux = useSiteStore((s) => s.lieux)
  const loading = useSiteStore((s) => s.loading)

  if (loading) {
    return <div className={styles.empty}>Chargement…</div>
  }
  if (lieux.length === 0) {
    return <div className={styles.empty}>Aucun lieu configuré</div>
  }

  return (
    <div className={styles.tree}>
      {lieux.map((lieu) => (
        <LieuNode key={lieu.id} lieu={lieu} />
      ))}
    </div>
  )
}

function LieuNode({ lieu }: { lieu: HierarchyLieu }) {
  const collapsed = useUiStore((s) => s.isCollapsed(`lieu:${lieu.id}`))
  const toggle = useUiStore((s) => s.toggleCollapsed)

  return (
    <div className={styles.node}>
      <button
        className={styles.nodeHeader}
        onClick={() => toggle(`lieu:${lieu.id}`)}
      >
        <span className={styles.chevron} data-open={!collapsed}>›</span>
        <span className={styles.icon}>📍</span>
        <span className={styles.label}>{lieu.name}</span>
        <span className={styles.count}>{lieu.sites.length}</span>
      </button>
      {!collapsed && (
        <div className={styles.children}>
          {lieu.sites.map((site) => (
            <SiteNode key={site.id} site={site} />
          ))}
        </div>
      )}
    </div>
  )
}

function SiteNode({ site }: { site: HierarchySite }) {
  const navigate = useNavigate()
  const collapsed = useUiStore((s) => s.isCollapsed(`site:${site.id}`))
  const toggle = useUiStore((s) => s.toggleCollapsed)
  const selectedSiteId = useSiteStore((s) => s.selectedSiteId)
  const selectSite = useSiteStore((s) => s.selectSite)
  const isSelected = selectedSiteId === site.id

  const handleClick = () => {
    selectSite(site.id)
    navigate(`/tracker/${site.id}`)
  }

  return (
    <div className={styles.node}>
      <div className={`${styles.nodeHeader} ${isSelected ? styles.selected : ''}`}>
        <button
          className={styles.chevronBtn}
          onClick={(e) => { e.stopPropagation(); toggle(`site:${site.id}`) }}
        >
          <span className={styles.chevron} data-open={!collapsed}>›</span>
        </button>
        <button className={styles.nodeLabel} onClick={handleClick}>
          <span className={styles.icon}>🏢</span>
          <span className={styles.label}>{site.name}</span>
          <span className={styles.count}>{site.cameras.length} cam</span>
        </button>
      </div>
      {!collapsed && (
        <div className={styles.children}>
          {site.cameras.map((cam) => (
            <CameraNode key={cam.id} camera={cam} siteId={site.id} />
          ))}
        </div>
      )}
    </div>
  )
}

function CameraNode({ camera, siteId }: { camera: HierarchyCamera; siteId: string }) {
  const navigate = useNavigate()
  const collapsed = useUiStore((s) => s.isCollapsed(`cam:${camera.id}`))
  const toggle = useUiStore((s) => s.toggleCollapsed)
  const selectedCameraId = useSiteStore((s) => s.selectedCameraId)
  const selectCamera = useSiteStore((s) => s.selectCamera)
  const selectSite = useSiteStore((s) => s.selectSite)
  const isSelected = selectedCameraId === camera.id

  const handleClick = () => {
    selectSite(siteId)
    selectCamera(camera.id)
    navigate(`/tracker/${siteId}`)
  }

  return (
    <div className={styles.node}>
      <div className={`${styles.nodeHeader} ${isSelected ? styles.selected : ''}`}>
        <button
          className={styles.chevronBtn}
          onClick={(e) => { e.stopPropagation(); toggle(`cam:${camera.id}`) }}
        >
          <span className={styles.chevron} data-open={!collapsed}>
            {camera.benefits.length > 0 ? '›' : ' '}
          </span>
        </button>
        <button className={styles.nodeLabel} onClick={handleClick}>
          <span className={styles.icon}>📷</span>
          <span className={styles.label}>{camera.name}</span>
          {camera.benefits.length > 0 && (
            <span className={styles.count}>{camera.benefits.length}</span>
          )}
        </button>
      </div>
      {!collapsed && camera.benefits.length > 0 && (
        <div className={styles.children}>
          {camera.benefits.map((ben) => (
            <BenefitLeaf key={ben.id} benefit={ben} />
          ))}
        </div>
      )}
    </div>
  )
}

function BenefitLeaf({ benefit }: { benefit: HierarchyBenefit }) {
  const skillIcon: Record<string, string> = {
    detection: '🔍',
    counting: '🔢',
    heatmap: '🌡️',
    quality: '✅',
  }

  return (
    <div className={styles.leaf}>
      <span className={styles.icon}>{skillIcon[benefit.skill] ?? '◆'}</span>
      <span className={styles.label}>{benefit.name}</span>
      <span className={`${styles.dot} ${benefit.active ? styles.dotActive : ''}`} />
    </div>
  )
}
