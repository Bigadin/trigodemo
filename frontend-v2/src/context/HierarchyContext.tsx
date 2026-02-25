import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { fetchHierarchy } from '@/api/hierarchy'
import { mergeBenefitIntoHierarchy } from '@/utils/hierarchy'
import type { Hierarchy, HierarchyBenefit } from '@/types/hierarchy'

const HierarchyContext = createContext<{
  hierarchy: Hierarchy
  loading: boolean
  error: string | null
  refetch: () => void
  optimisticMergeBenefit: (cameraId: string, benefit: HierarchyBenefit) => void
}>({
  hierarchy: {},
  loading: true,
  error: null,
  refetch: () => {},
  optimisticMergeBenefit: () => {},
})

export function HierarchyProvider({ children }: { children: React.ReactNode }) {
  const [hierarchy, setHierarchy] = useState<Hierarchy>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async (silent = false) => {
    if (!silent) {
      setLoading(true)
      setError(null)
    }
    try {
      const data = await fetchHierarchy()
      setHierarchy(data)
    } catch (e) {
      if (!silent) {
        setError(e instanceof Error ? e.message : 'Erreur chargement')
        setHierarchy({})
      }
    } finally {
      if (!silent) setLoading(false)
    }
  }

  const refetch = useCallback(() => load(true), [])

  useEffect(() => {
    load(false)
  }, [])

  const optimisticMergeBenefit = useCallback((cameraId: string, benefit: HierarchyBenefit) => {
    setHierarchy((prev) => mergeBenefitIntoHierarchy(prev, cameraId, benefit))
  }, [])

  return (
    <HierarchyContext.Provider value={{ hierarchy, loading, error, refetch, optimisticMergeBenefit }}>
      {children}
    </HierarchyContext.Provider>
  )
}

export function useHierarchy() {
  return useContext(HierarchyContext)
}
