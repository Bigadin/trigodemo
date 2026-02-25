# Spécification — Modal de création de bénéfices (zones)

> **Important** : Ce modal est DIFFÉRENT du modal de modification.  
> À la **création**, on propose TOUT : les 4 skills, tous les types de skill, toutes les catégories et tous les sous-types.  
> À la **modification**, on affiche une version simplifiée (données déjà choisies).

---

## 1. Les 4 skills (ordre d'affichage)

| Skill | Label | Icône SVG |
|-------|-------|-----------|
| **detection** | Détection | `/static/assets_youn/SvIcons/SVGnew/Yclassify.svg` |
| **counting** | Comptage | `/static/assets_youn/SvIcons/SVGnew/Ycounting.svg` |
| **heatmap** | Heatmap | `/static/assets_youn/SvIcons/SVGnew/Yheatmap.svg` |
| **quality** | Qualité | `/static/assets_youn/SvIcons/SVGnew/Yqualitydefect.svg` |

> **Qualité** : utiliser `Yqualitycheck.svg` (ou `Yqualitydefect.svg` pour les sous-types), PAS les icônes des autres skills.

---

## 2. Sous-skills par skill

### 2.1 Détection (3 types)
| id | label | icône |
|----|-------|-------|
| `detection_presence` | Présence / Absence | `Ysilhouette.svg` |
| `detection_linecross` | Franchissement ligne | `Ylinecross.svg` |
| `detection_zone` | Détection zone | `Yzonedetect.svg` (renommé depuis Ydetectionsscat) |

### 2.2 Comptage (3 types, icônes différentes)
| id | label | icône |
|----|-------|-------|
| `counting_people` | Comptage personnes | `Ycountingppl.svg` |
| `counting_objects` | Comptage objets | `Ycounting.svg` |
| `counting_zone` | Comptage zone | `square-area-svgrepo-com.svg` |

### 2.3 Heatmap (3 types)
| id | label | icône |
|----|-------|-------|
| `heatmap_density` | Densité de flux | `grid-svgrepo-com.svg` |
| `heatmap_presence` | Heatmap présence | `Yheatmapdense.svg` |
| `heatmap_trajectory` | Heatmap trajectoires | `Ytraj.svg` |

### 2.4 Qualité (3 types)
| id | label | icône |
|----|-------|-------|
| `quality_fissure` | Fissure | `Yfissure.svg` |
| `quality_humidity` | Humidité | `Yhumidity.svg` |
| `quality_check` | Qualité générale | `Yqualitycheck.svg` |

---

## 3. Catégories (3 catégories principales)

| key | label | icône |
|-----|-------|-------|
| **human** | Humain | `Yhumancat.svg` |
| **transport** | Transport | `Ytransport2.svg` |
| **object** | Objet | `Yobstruction.svg` |

---

## 4. Sous-types par catégorie

### 4.1 Humain
| id | label | icône |
|----|-------|-------|
| `silhouette` | Silhouette | `Ysilhouette.svg` |
| `visage` | Visage | `Yface.svg` |
| `foule` | Foule | `Ycrowdfoule.svg` |

### 4.2 Transport
| id | label | icône |
|----|-------|-------|
| `voiture` | Voiture | `Ycar.svg` |
| `velo` | Vélo | `Ybike.svg` |
| `public_transport` | Transport public | `Ypublic transport.svg` |
| `avion` | Avion | `plane-svgrepo-com.svg` |
| `moto` | Moto | `motorcycle.svg` |

### 4.3 Objet (avec encombrement)
| id | label | icône |
|----|-------|-------|
| `encombrement` | Encombrement | `Yobstruction.svg` |
| `zone_encombrée` | Zone encombrée | `Yobstruction.svg` |

---

## 5. Mapping catégories par skill

Chaque skill peut avoir un sous-ensemble des catégories. À la **création**, on propose toutes les combinaisons pertinentes :

| Skill | Catégories disponibles |
|-------|-------------------------|
| detection | human, transport |
| counting | human, transport |
| heatmap | human, object (encombrement) |
| quality | object (defaut, fissure) |

Pour **quality**, les sous-types (défauts) sont :
- `fissure` → `Yfissure.svg`
- `humidity` (Humidité) → `Yhumidity.svg`
- `qualitycheck` (Qualité générale) → `Yqualitycheck.svg`

---

## 6. Icônes de fallback

- Skill non reconnu : `Yqualitydefect.svg`
- Catégorie / sous-type sans icône : `Yqualitydefect.svg`

---

## 7. Étapes du modal (création)

1. **skill** : choix du skill + sous-skill (4 cartes skills avec sous-options)
2. **category** : choix des catégories + sous-types (arbre complet)
3. **info** : nom, opérateur, date, commentaire, zone activée, plage horaire
4. Dessin de la zone (polygone ou full_screen selon le skill)

---

## 8. Règles d'icônes SVTG vs Yquality

| Skill | Icônes à utiliser |
|-------|-------------------|
| **detection** | SVTG standard : `Yclassify.svg`, `Ysilhouette.svg`, `Ylinecross.svg` |
| **counting** | SVTG standard : `Ycounting.svg`, `Ycountingppl.svg`, `square-area-svgrepo-com.svg` — chaque type utilise une icône différente |
| **heatmap** | SVTG standard : `Yheatmap.svg`, `grid-svgrepo-com.svg` (densité), `Yheatmapdense.svg` (présence), `Ytraj.svg` |
| **quality** | **Yquality** uniquement : `Yqualitydefect.svg` (skill), `Yfissure.svg`, `Yhumidity.svg`, `Yqualitycheck.svg` (sous-types) — PAS les icônes des autres skills |

> **Qualité** : toujours utiliser le préfixe `Yquality*` (ex: `Yqualitycheck.svg`, `Yqualitydefect.svg`), jamais les icônes SVTG des skills detection/counting/heatmap.

---

## 9. Fichiers à maintenir synchronisés

- `static/js/skills-adapter.js` — `DEFAULT_SKILLS_CONFIG` (source de vérité pour la création)
- `main.py` — `SKILLS_CONFIG` (endpoint `/api/skills`)
- `patch_skills.py` — si utilisé pour patcher

> **Note** : Le modal de création utilise toujours `DEFAULT_SKILLS_CONFIG` du skills-adapter pour afficher les 4 skills, toutes les catégories et tous les sous-types.

---

## 10. Logique des canvas (zones)

Pour le détail du modèle de données et de la logique des zones (polygones, include/exclude, `zone_ref`, ratio modal ↔ preview), voir **[SPEC_CANVAS_BENEFICES.md](./SPEC_CANVAS_BENEFICES.md)**.

