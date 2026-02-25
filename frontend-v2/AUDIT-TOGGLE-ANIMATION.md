# Audit : animation toggle vanilla vs React

## Problème identifié

L'animation goo du toggle n'était pas visible dans la version React.

## Cause principale (vanilla index.js:6151)

> "Change sur le toggle bénéfice : **mise à jour de l'état sans re-render (pour garder l'animation)**"

En vanilla, au clic sur le toggle :
- Le checkbox natif change immédiatement (pas de `preventDefault`)
- L'événement `change` met à jour le cache en mémoire
- **Aucun re-render** des benefit rows → l'animation CSS s'exécute sans interruption

En React (avant correction) :
- `onToggleBenefit` → `refetchHierarchy()` → re-render complet
- Le re-render remplace le DOM pendant l'animation → **animation coupée**

## Corrections appliquées

### 1. Optimistic update (BenefitsOverview.tsx)
- État local `optimisticActive` pour chaque bénéfice
- Au clic : mise à jour immédiate de l'affichage (sans attendre l'API)
- L'animation s'exécute car le DOM n'est pas remplacé
- Revert en cas d'erreur API

### 2. Filtre SVG (index.html)
- Région du filtre : `x="-20%" y="-20%" width="140%" height="140%"` pour éviter le clipping du blur
- `color-interpolation-filters="sRGB"` pour des couleurs cohérentes
- `overflow: visible` sur le SVG

### 3. Styles (BenefitsOverview.module.css)
- `overflow: visible` sur le toggle pour ne pas couper l'effet goo

## Structure vanilla (identique)

```html
<label class="ben-toggle ben-toggle--success benefit-row-toggle">
  <input type="checkbox" class="ben-toggle__input">
  <span class="ben-toggle__track">
    <span class="ben-toggle__dot ben-toggle__dot--left"></span>
    <span class="ben-toggle__dot ben-toggle__dot--right"></span>
    <span class="ben-toggle__drop"></span>
  </span>
</label>
```

- Filtre goo : `filter: url(#ben-goo)` sur le track
- Animation : left dot → `translateX(12px) scale(0)`, right dot → `scale(1)`
- Transitions : `0.5s ease` (dots), `0.7s` (drop)
