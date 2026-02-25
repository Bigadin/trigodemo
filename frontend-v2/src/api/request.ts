/**
 * Module de communication API avec support des mises à jour optimistes.
 * Permet de réduire la latence perçue en appliquant les changements immédiatement
 * et en synchronisant avec le serveur en arrière-plan.
 */

export interface OptimisticMutationOptions<T> {
  /** Appliqué immédiatement avant l'appel API (mise à jour optimiste) */
  optimisticApply: () => void
  /** L'appel API réel */
  mutate: () => Promise<T>
  /** Appelé en cas de succès (optionnel) */
  onSuccess?: (result: T) => void
  /** Appelé en cas d'erreur - permet de revenir en arrière ou notifier */
  onError?: (error: Error) => void
}

/**
 * Exécute une mutation avec mise à jour optimiste.
 * Applique optimisticApply immédiatement, puis lance mutate en arrière-plan.
 * Le retour est immédiat (pas d'attente de l'API).
 */
export function optimisticMutation<T>(options: OptimisticMutationOptions<T>): void {
  try {
    options.optimisticApply()
  } catch (e) {
    if (options.onError) options.onError(e instanceof Error ? e : new Error(String(e)))
    return
  }
  options.mutate().then(options.onSuccess).catch((e) => {
    if (options.onError) options.onError(e instanceof Error ? e : new Error(String(e)))
  })
}
