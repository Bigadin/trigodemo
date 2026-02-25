import { useState, useEffect, useRef } from 'react'

/**
 * Hook pour le timer de session (temps écoulé depuis le lancement du stream).
 * S'incrémente chaque seconde quand isStreaming est true, se réinitialise à 0 quand false.
 */
export function useSessionTimer(isStreaming: boolean): number {
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef<number | null>(null)

  useEffect(() => {
    if (!isStreaming) {
      startRef.current = null
      setElapsed(0)
      return
    }
    startRef.current = Date.now()
    setElapsed(0)
    const interval = setInterval(() => {
      if (startRef.current != null) {
        setElapsed(Math.floor((Date.now() - startRef.current) / 1000))
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [isStreaming])

  return elapsed
}
