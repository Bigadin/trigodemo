# Vocabulaire et mapping — Data Room & Modal bénéfice

> Mots-clés pour ajouter une **carte Data Room** ou un **paramètre dans le modal bénéfice** sans modifier le frontend.

---

## 1. Data Room — Ajouter une carte

La Data Room affiche des **cartes** par type de bénéfice. Les cartes actuelles sont :
- **Rapport d'activité** (agrégée) — une seule carte large avec sous-cartes par zone (Forme 1, Forme 2…), courbes compactes et LOV laps de temps
- **Détection présence** (skill `detection`) — une carte par bénéfice détection, avec LOV Tri
- **Comptage** (skill `counting`) — une carte par bénéfice comptage, avec LOV Mode en header

### Comment le front affiche une carte

| Carte | Condition d'affichage | Éléments header |
|------|------------------------|-----------------|
| Rapport d'activité | Premier bénéfice `detection` avec `detection_presence` | LOV Laps (2, 5, 10, 15 min), CardMenu |
| Détection présence | Chaque bénéfice `detection` | LOV Tri (Par temps, Par %, Par nom), CardMenu |
| Comptage | Chaque bénéfice `counting` avec zones | LOV Mode (Gradient, MOG2), CardMenu |

**Note** : Les chips "Présence / Absence" et "Comptage" ont été supprimés. Chaque carte a une LOV dans le header pour une config liée au type.

### Pour ajouter une nouvelle carte (ex. Heatmap)

1. **Backend** : fournir un bénéfice avec `skill: "heatmap"` (ou autre skill existant dans `api/skills.ts`).
2. **Frontend** : modifier `DataRoomCards.tsx` pour :
   - détecter le skill `heatmap`
   - afficher une section dédiée (ex. `getHeatmapItems()`)
3. **API** : exposer une API spécifique (ex. `/api/heatmap/:videoPath`) pour les données.

### Mots-clés Data Room

| Mot-clé | Usage |
|---------|-------|
| `skill` | `detection`, `counting`, `heatmap`, `quality` — détermine le type de carte |
| `skill_item` | Sous-type (ex. `detection_presence`, `counting_people`) |
| `categories` | `["human::silhouette", "transport::voiture"]` — format `category::subcategory` |
| `benefit_id` | Identifiant unique du bénéfice (utilisé pour les zones) |
| `zones` | Données par `benefit_id` : `total_time`, `is_occupied` |
| `counting` | Données comptage : `count`, `mode`, `zone_name`, `enabled` |

### Icônes par catégorie (Data Room)

| Clé | Fichier |
|-----|---------|
| `human` | `/static/assets_youn/SvIcons/SVGnew/Yhumancat.svg` |
| `voiture` | `/static/assets_youn/SvIcons/SVGnew/Ycar.svg` |
| `velo` | `/static/assets_youn/SvIcons/SVGnew/Ybike.svg` |
| `detection` | `/static/assets_youn/SvIcons/SVGnew/Yclassify.svg` |
| `counting` | `/static/assets_youn/SvIcons/SVGnew/Ycounting.svg` |
| `counting_people` | `/static/assets_youn/SvIcons/SVGnew/Ycountingppl.svg` |

---

## 2. Modal bénéfice — Paramétrage

Le modal bénéfice (`BenefitConfigModal`) a 3 étapes : **Skill** → **Catégorie** → **Infos**.

### Étape 1 — Skill

| Mot-clé | Description |
|---------|-------------|
| `skill` | Groupe principal : `detection`, `counting`, `heatmap`, `quality` |
| `skill_item` | Sous-skill : `detection_presence`, `counting_people`, etc. |

**Source** : `GET /api/skills` ou fallback dans `frontend-v2/src/api/skills.ts` (structure `groups` → `items`).

### Étape 2 — Catégorie

| Mot-clé | Description |
|---------|-------------|
| `categories` | Format `["category::subcategory"]` — ex. `human::silhouette`, `transport::voiture` |
| `category` | Groupe : `human`, `transport`, `object` |
| `subcategory` | Sous-type : `silhouette`, `visage`, `voiture`, `velo`, etc. |

**Source** : `getCategoryGroupsBySkill(skill)` dans `api/skills.ts` — dépend du skill sélectionné.

### Étape 3 — Infos (paramètres)

| Champ | Clé API | Type | Description |
|-------|---------|------|--------------|
| Nom | `name` | string | Nom du bénéfice |
| Créé par | `created_by` | string | (non persisté actuellement) |
| Créé le | `created_at` | string | (non persisté actuellement) |
| Commentaire | `comment` | string | (non persisté actuellement) |
| Zone activée | `active` | boolean | Bénéfice actif ou non |
| Scheduling | `schedule_enabled`, `schedule_start`, `schedule_end` | — | (non persisté actuellement) |

### Pour ajouter un paramètre dans le modal

1. **Backend** : ajouter le champ dans le payload `createBenefit` / `updateBenefit` si besoin.
2. **Frontend** : dans `BenefitConfigModal.tsx`, section `data-ben-step-panel="info"` :
   - ajouter un `useState` pour le nouveau champ
   - ajouter un `<label className="ben-field">` avec input
   - inclure le champ dans `benefitData` au `handleSave`

### Mots-clés modal bénéfice

| Mot-clé | Usage |
|---------|-------|
| `benefit_id` | ID unique (généré par front si création : `ben-{cameraId}-{timestamp}`) |
| `zone_polygons` | `number[][][]` — polygones en coordonnées canvas |
| `zone_polygon_types` | `include` \| `exclude` par polygone |
| `zone_ref_width`, `zone_ref_height` | Dimensions de référence pour le ratio |
| `camera_id` | Caméra associée |

---

## 3. Skills et catégories — Référence complète

### Skills (groupes)

| key | label | items (skill_item) |
|-----|-------|--------|
| `detection` | Détection | `detection_presence`, `detection_linecross`, `detection_zone` |
| `counting` | Comptage | `counting_people`, `counting_objects`, `counting_zone` |
| `heatmap` | Heatmap | `heatmap_density`, `heatmap_presence`, `heatmap_trajectory` |
| `quality` | Qualité | `quality_fissure`, `quality_humidity`, `quality_check` |

### Catégories (par skill)

| Groupe | key | Sous-catégories (id) |
|-------|-----|----------------------|
| Humain | `human` | `silhouette`, `visage`, `foule` |
| Transport | `transport` | `voiture`, `velo`, `public_transport`, `avion`, `moto` |
| Objet | `object` | `encombrement`, `zone_encombre`, `fissure`, `humidity`, `qualitycheck` |

### Format `categories` dans le bénéfice

```
["human::silhouette", "transport::voiture", "transport::velo"]
```

---

## 4. Images lieux

| Fichier | Lieu | Contexte |
|---------|------|----------|
| `/static/assets_youn/OG.jpg` | OG Logistics (usine) | Image lieu par défaut si `lieu_id === "usine"` |
| `/static/assets_youn/westfield.jpg` | Galerie Westfield (mall) | Image lieu par défaut si `lieu_id === "mall"` |

Le champ `icon` dans l'API hierarchy peut surcharger : `lieu.icon` ou `LIEU_ICONS[lieu_id]` dans `utils/theme.ts`.
