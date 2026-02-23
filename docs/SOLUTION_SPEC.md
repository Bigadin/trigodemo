# Specification centrale - Solution YRYS

Source de verite: static/config/solution-spec.json

## 1. Les 4 skills

- detection: DÃ©tection (Yclassify.svg)
- counting: Comptage (Ycounting.svg)
- heatmap: Heatmap (Yheatmap.svg)
- quality: QualitÃ© (Yqualitycheck.svg)

## 2. Les 3 categories

| key | label | Icone |
|-----|-------|-------|
| human | Humain | Yhumancat.svg |
| transport | Transport | Ytransport2.svg |
| object | Objet | Yobstruction.svg |

## 3. Transport (sous-cats)
voiture, velo, public_transport, avion, moto

## 4. Objet (sous-cats)
encombrement, zone_encombrÃ©e

## 5. Mapping par skill
- detection: human, transport
- counting: human, transport
- heatmap: human, object
- quality: object