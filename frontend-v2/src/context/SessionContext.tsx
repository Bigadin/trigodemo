import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'

export function benefitElapsedKey(siteId: string, camId: string, benefitId: string): string {
  return `${siteId}:${camId}:${benefitId}`
}

interface SessionContextValue {
  benefitElapsed: Record<string, number>
  streamingSiteId: string | null
  streamingCamId: string | null
  setStreaming: (siteId: string | null, camId: string | null) => void
  tickBenefitTimers: (activeKeys: string[]) => void
  resetBenefitTimers: (keys?: string[]) => void
}

const SessionContext = createContext<SessionContextValue>({
  benefitElapsed: {},
  streamingSiteId: null,
  streamingCamId: null,
  setStreaming: () => {},
  tickBenefitTimers: () => {},
  resetBenefitTimers: () => {},
})

export function SessionProvider({ children }: { children: ReactNode }) {
  const [benefitElapsed, setBenefitElapsed] = useState<Record<string, number>>({})
  const [streamingSiteId, setStreamingSiteId] = useState<string | null>(null)
  const [streamingCamId, setStreamingCamId] = useState<string | null>(null)

  const setStreaming = useCallback((siteId: string | null, camId: string | null) => {
    setStreamingSiteId(siteId)
    setStreamingCamId(camId)
  }, [])

  const tickBenefitTimers = useCallback((activeKeys: string[]) => {
    if (activeKeys.length === 0) return
    setBenefitElapsed((prev) => {
      const next = { ...prev }
      for (const key of activeKeys) {
        next[key] = (prev[key] ?? 0) + 1
      }
      return next
    })
  }, [])

  const resetBenefitTimers = useCallback((keys?: string[]) => {
    if (keys) {
      setBenefitElapsed((prev) => {
        const next = { ...prev }
        for (const k of keys) next[k] = 0
        return next
      })
    } else {
      setBenefitElapsed({})
    }
  }, [])

  return (
    <SessionContext.Provider
      value={{
        benefitElapsed,
        streamingSiteId,
        streamingCamId,
        setStreaming,
        tickBenefitTimers,
        resetBenefitTimers,
      }}
    >
      {children}
    </SessionContext.Provider>
  )
}

export function useSession() {
  return useContext(SessionContext)
}
