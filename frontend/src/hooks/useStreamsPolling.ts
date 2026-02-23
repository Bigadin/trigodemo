import { useEffect, useRef } from 'react'
import { useVideoStore } from '@/stores/videoStore'

const INTERVAL_MS = 2_000

export function useStreamsPolling() {
  const refreshStreams = useVideoStore((s) => s.refreshStreams)
  const timer = useRef<ReturnType<typeof setInterval>>(undefined)

  useEffect(() => {
    refreshStreams()
    timer.current = setInterval(refreshStreams, INTERVAL_MS)
    return () => clearInterval(timer.current)
  }, [refreshStreams])
}
