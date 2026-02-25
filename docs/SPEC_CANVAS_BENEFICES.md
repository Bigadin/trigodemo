# Spécification — Canvas des bénéfices (zones)

> Les zones de détection sont **attachées au bénéfice** (pas à un objet séparé).  
> Ce document décrit le modèle de données et la logique d'affichage.

---

## 1. Champs du bénéfice liés aux zones

| Champ | Type | Description |
|-------|------|-------------|
| `zone_polygons` | `number[][][]` | Liste de polygones. Chaque polygone = liste de points `[x, y]` en coordonnées pixels. Min 3 points par polygone. |
| `zone_polygon_types` | `('include' \| 'exclude')[]` | Type de chaque polygone. Même longueur que `zone_polygons`. |
| `zone_ref_width` | `number` | Largeur de la vidéo au moment du dessin (espace de coordonnées). |
| `zone_ref_height` | `number` | Hauteur de la vidéo au moment du dessin (espace de coordonnées). |

### Exemple

```json
{
  "zone_polygons": [
    [[100, 100], [500, 100], [500, 400], [100, 400]],
    [[200, 200], [300, 200], [300, 300], [200, 300]]
  ],
  "zone_polygon_types": ["include", "exclude"],
  "zone_ref_width": 1920,
  "zone_ref_height": 1080
}
```

---

## 2. Types de polygones (include / exclude)

- **include** : zone d'inclusion — détection à l'intérieur (affichage vert).
- **exclude** : zone d'exclusion — exclusion de la zone (affichage orange).

**Règles :**
- Chaque polygone a son propre type, stocké dans `zone_polygon_types[idx]`.
- On peut avoir **plusieurs include** ou **plusieurs exclude** dans un même bénéfice.
- Pas d'alternance imposée (ancienne logique `idx % 2` supprimée).
- Si `zone_polygon_types` est absent (anciens bénéfices) : fallback = tout en `'include'`.

---

## 3. Système de coordonnées et ratio

### 3.1 Référence de dimensions

Les coordonnées des polygones sont **toujours** dans l'espace `zone_ref_width × zone_ref_height` :

- Au **dessin** : le canvas utilise ces dimensions (ou `videoInfo` si pas encore chargé).
- À la **sauvegarde** : on enregistre `zone_ref_width` et `zone_ref_height` avec les polygones.
- À l'**affichage** : le viewBox SVG utilise ces dimensions pour garder le bon ratio.

### 3.2 Cohérence modal ↔ preview

| Contexte | Dimensions utilisées |
|----------|------------------------|
| Modal édition (canvas) | `zone_ref_width` × `zone_ref_height` ou `videoInfo` |
| Zone Tracker (preview vidéo) | `zone_ref_width` × `zone_ref_height` pour le viewBox SVG |
| Conteneur vidéo | `videoWidth` × `videoHeight` pour l'aspect-ratio |

### 3.3 Bénéfices sans zone_ref (rétrocompat)

Si `zone_ref_width` / `zone_ref_height` sont absents :
- On infère les dimensions à partir de l'étendue des polygones (max X, max Y + 2 %).
- Le viewBox est calculé pour que les formes s'affichent correctement.

---

## 4. Nombre de formes

- **Minimum** : 0 (bénéfice sans zone = analyse sur toute l'image).
- **Recommandé** : au moins 1 forme pour une zone de détection.
- **Maximum** : pas de limite — on peut ajouter autant de formes include/exclude que nécessaire.

---

## 5. API Backend

### Création / mise à jour

```json
{
  "zone_polygons": [[[x,y], ...], ...],
  "zone_polygon_types": ["include", "exclude", "include"],
  "zone_ref_width": 1920,
  "zone_ref_height": 1080
}
```

### Fallback backend

Si `zone_polygon_types` est absent à la création : tous les polygones sont considérés comme `'include'`.

---

## 6. Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `frontend-v2/src/components/tracker/BenefitConfigModal.tsx` | Éditeur de zones, sauvegarde |
| `frontend-v2/src/components/tracker/VideoPlayer.tsx` | Affichage overlay sur la vidéo |
| `frontend-v2/src/hooks/useBenefitEditorCanvas.ts` | Logique dessin, undo, types |
| `main.py` | API benefits, `zone_polygon_types`, `zone_ref_*` |
