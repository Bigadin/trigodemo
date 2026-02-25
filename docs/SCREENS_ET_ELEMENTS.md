# Screens et éléments — Schémas des pages

> Schémas des vues avec nom des composants et éléments clés pour l'intégration et les tests.

---

## 1. Layout global

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AppLayout                                                                   │
├──────────────┬──────────────────────────────────────────────────────────────┤
│  Sidebar     │  TopBar                                                       │
│  - Logo      │  - Liens Sites | Tracker | Analytics | Logs                  │
│  - Explorer  │  - Breadcrumb (si contexte)                                  │
│  Tree        │                                                               │
├──────────────┼──────────────────────────────────────────────────────────────┤
│              │  <main>                                                       │
│  ExplorerTree│  - Routes : /, /lieu/:lieuId, /tracker, /analytics, /logs    │
│  - Lieux     │                                                               │
│  - Sites     │                                                               │
│  - Caméras   │                                                               │
│  - Bénéfices │                                                               │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 2. Vue d'ensemble (SitesView) — `/`

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Header** | `styles.header` | Titre "Vue d'ensemble", bouton "Créer un lieu" |
| **Group header** | `styles.groupHeader` | Lieu (icône planète, nom lieu) |
| **Carte lieu** | `styles.card` | Carte par lieu : image, nom, 2 mini screens caméras |
| **Menu 3 points** | `CardMenu` | Créer / Modifier / Exporter (par carte) |

**Structure** : `SitesView.tsx` → grille de cartes par lieu (hierarchy).

---

## 3. Vue Lieu (LieuView) — `/lieu/:lieuId`

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Header** | `styles.header` | Titre "Vue Lieu", nom du lieu |
| **Carte site** | `styles.card` | Carte par site : image, nom, mini screens caméras par site |
| **Bénéfices** | `styles.benefits` | Liste défilante des bénéfices dans la carte |
| **Flèches** | `styles.arrowLeft` / `arrowRight` | Défilement bénéfices (si overflow) |
| **Menu 3 points** | `CardMenu` | Créer / Modifier / Exporter |
| **Clic carte** | — | Sélectionne le site, navigation vers `/tracker/:siteId` |

**Structure** : `LieuView.tsx` → grille de cartes par site du lieu.

---

## 4. Vue Tracker (TrackerView) — `/tracker` ou `/tracker/:siteId`

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Header** | `styles.header` | Titre site + caméra, breadcrumb |
| **Onglets** | `styles.tabs` | Overview / Source / Settings |
| **Grille caméras** | `CameraGrid` | Tuiles caméras (miniatures) |
| **Caméra sélectionnée** | `styles.selectedCam` | Bordure / highlight |
| **Lecteur vidéo** | `VideoPlayer` | Zone vidéo principale |
| **Overlay canvas** | — | Zones dessinées sur la vidéo |
| **Liste bénéfices** | `BenefitsOverview` | Liste des bénéfices de la caméra |
| **Data Room** | `DataRoomCards` | Bloc "DATA ROOM" avec cartes |
| **Bouton +** | — | Ouvrir modal création bénéfice |

**Structure** : `TrackerView.tsx` → layout avec sidebar droite (Data Room, BenefitsOverview) et zone vidéo gauche.

---

## 5. Data Room (dans TrackerView)

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Bloc** | `styles.block` | Conteneur DATA ROOM |
| **Titre** | `styles.title` | "DATA ROOM" |
| **Sous-titre** | `styles.subtitle` | "Bénéfices et mesures" |
| **Carte Détection** | `styles.card` | Titre "DÉTECTION PRÉSENCE", barres de présence |
| **Carte Comptage** | `styles.card` | Titre "COMPTAGE", compteurs |
| **Dot couleur** | `styles.dot` | Pastille couleur par bénéfice |
| **Menu 3 points** | `CardMenu` | Modifier / Créer / Exporter |
| **Chips** | `styles.chip` | Badge "Détection" ou "Comptage" |
| **Barres présence** | `styles.presenceBar` | Humain, Voiture, Vélo (selon categories) |
| **Compteurs** | `styles.counterWrap` | Icône + valeur + label |

**Composant** : `DataRoomCards.tsx` — props : `benefits`, `zones`, `counting`, `onAddBenefit`, `onEditBenefit`.

---

## 6. Modal bénéfice (BenefitConfigModal)

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Overlay** | `editor-overlay` | Fond plein écran |
| **Panel gauche** | `ben-panel` | Wizard (étapes) |
| **Panel droit** | `editor-right` | Canvas éditeur zones |
| **Stepper** | `ben-stepper` | Étapes 1 Skill, 2 Catégorie, 3 Infos |
| **Skill cards** | `ben-skill-card` | Boutons Détection, Comptage, Heatmap, Qualité |
| **Skill items** | `ben-tree-leaf` | Sous-skills (Présence, Franchissement ligne, etc.) |
| **Catégories** | `ben-tree-group` | Groupes human, transport, object |
| **Catégorie items** | `ben-tree-leaf` | Silhouette, Voiture, etc. |
| **Champs infos** | `ben-field` | Nom, Créé par, Créé le, Commentaire |
| **Toggle Zone activée** | `ben-toggle` | Actif/inactif |
| **Toggle Scheduling** | `ben-toggle` | Activer planning |
| **Canvas** | `canvas` | Dessin zones (include/exclude) |
| **Barre outils** | `tool-btn` | Sélect, Include, Exclude, Croix |
| **Hover bar** | — | "Supprimer point" / "Supprimer forme" au survol |
| **Boutons** | `ben-step-btn` | Retour, Suivant, SAUVEGARDER |
| **Menu 3 points** | `CardMenu` | Supprimer le bénéfice |

**Composant** : `BenefitConfigModal.tsx` — props : `open`, `onClose`, `cameraId`, `cameraName`, `videoPath`, `benefit`, `onSaved`.

---

## 7. Explorer (Sidebar)

| Élément | Nom composant / classe | Description |
|---------|------------------------|-------------|
| **Arbre** | `ExplorerTree` | Lieux → Sites → Caméras → Bénéfices |
| **Noeud lieu** | — | Icône planète, nom lieu, expand/collapse |
| **Noeud site** | — | Icône lieu, nom site |
| **Noeud caméra** | — | Icône caméra, nom caméra |
| **Noeud bénéfice** | — | Icône skill, nom bénéfice |
| **Sélection** | — | Highlight du noeud sélectionné |

**Composant** : `ExplorerTree.tsx` — props : `hierarchy`, `selectedSiteId`, `selectedCamId`, `onSelectSite`, `onSelectCamera`, `onSelectBenefit`.

---

## 8. Routes et navigation

| Route | Vue | Composant principal |
|-------|-----|---------------------|
| `/` | Vue d'ensemble | SitesView |
| `/lieu/:lieuId` | Vue Lieu | LieuView |
| `/tracker` | Tracker (sans site) | TrackerView |
| `/tracker/:siteId` | Tracker (site sélectionné) | TrackerView |
| `/tracker?cam=xxx` | Tracker (caméra sélectionnée) | TrackerView |
| `/analytics` | Analytics | AnalyticsView |
| `/logs` | Logs | LogsView |

---

## 9. Attributs data pour les tests

| Élément | Attribut | Valeur |
|---------|----------|--------|
| Panel étape Skill | `data-ben-step-panel` | `skill` |
| Panel étape Catégorie | `data-ben-step-panel` | `category` |
| Panel étape Infos | `data-ben-step-panel` | `info` |
| Étape stepper | `data-state` | `active`, `completed`, `inactive` |

Pour ajouter des `data-testid` : cibler ces classes ou attributs dans les tests E2E.
