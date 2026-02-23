import { useEffect, useRef, useCallback } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useSiteStore } from '@/stores/siteStore'

const ACTIVE_MS = 1_200
const IDLE_MS = 5_000

export function useZonesPolling() {
  const activeStreams = useVideoStore((s) => s.activeStreams)
  const refreshZones = useVideoStore((s) => s.refreshZones)
  const refreshPresence = useVideoStore((s) => s.refreshPresence)
  const selectedSiteId = useSiteStore((s) => s.selectedSiteId)
  const getSite = useSiteStore((s) => s.getSite)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const tick = useCallback(async () => {
    const site = selectedSiteId ? getSite(selectedSiteId) : undefined
    if (!site) return

    const videos = site.cameras.map((c) => c.path)
    const hasActive = videos.some((v) => activeStreams.has(v))

    await Promise.all(
      videos.map(async (v) => {
        await refreshZones(v)
        if (activeStreams.has(v)) {
          await refreshPresence(v)
        }
      }),
    )

    const nextMs = hasActive ? ACTIVE_MS : IDLE_MS
    timer.current = setTimeout(tick, nextMs)
  }, [selectedSiteId, getSite, activeStreams, refreshZones, refreshPresence])

  useEffect(() => {
    if (!selectedSiteId) return
    tick()
    return () => clearTimeout(timer.current)
  }, [selectedSiteId, tick])
}
