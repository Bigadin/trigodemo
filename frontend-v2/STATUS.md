# Frontend-v2 — Point de situation

## Lignes de code

**~9 560 lignes** (src/**/*.{ts,tsx,css})

- TS/TSX : ~5 180
- CSS : ~4 380

## Structure actuelle

```
src/
├── api/           # client, hierarchy, benefits, lieux, sites, tracker, skills, request
├── components/
│   ├── layout/    # AppLayout, Sidebar, TopBar
│   ├── sidebar/   # ExplorerTree + animations
│   ├── tracker/   # VideoPlayer, CameraGrid, ZoneList, DataRoomCards, BenefitsOverview, BenefitConfigModal
│   └── ui/        # CardMenu, LovDropdown, Toggle, MenuBtn
├── context/       # HierarchyContext
├── hooks/         # useBenefitEditorCanvas
├── styles/        # tokens, reset, benefit-panel
├── types/         # hierarchy
├── utils/         # theme, hierarchy, polygonPreview
└── views/         # SitesView, LieuView, TrackerView, AnalyticsView, LogsView, SettingsView
```

## Réutilisation

- **theme.ts** : `icon()`, `img()`, `getLocColor()`, `getLieuIcon()` — point d'entrée unique pour les assets
- **HierarchyContext** : hiérarchie partagée (lieux → sites → caméras → bénéfices)
- **ExplorerTree** : collapse/expand, sélection, navigation
- **PlaceholderView.module.css** : styles communs aux vues (page, header, placeholder)

## Assets

### Icônes UI — `/static/icons/`
`location`, `site`, `camera`, `folder`, `desktop`, `chart`, `terminal`, `zone`

### Images lieux — `/static/assets_youn/`
| Fichier | Contexte |
|---------|----------|
| `OG.jpg` | Image lieu "OG Logistics" (usine) |
| `westfield.jpg` | Image lieu "Galerie Westfield" (mall) |

### Icônes skills / bénéfices — `/static/assets_youn/SvIcons/`, `/static/assets_youn/YrysUIPackage/`
Voir `docs/VOCABULAIRE_ET_MAPPING.md` pour le mapping complet.

### Images — `/static/images/` (logos, textures)
| Alias | Fichier | Contexte |
|-------|---------|----------|
| `arcy-logo` | arcy-logo.png | Logo Arcy (sidebar) |
| `arcy-logo-white` | arcy-logo-white.svg | Logo blanc (topbar fond sombre) |
| `bg-texture-light` | bg-texture-light.png | Texture fond clair |

## Zone Tracker

Vue fonctionnelle avec :
- `useHierarchy()`, `useParams()`, `useSearchParams()`
- Data Room (cartes Détection présence, Comptage)
- Modal création/édition bénéfice (BenefitConfigModal)
- API `/api/videos`, `/api/zones`, `/api/presence`, `/api/counting`, etc.
