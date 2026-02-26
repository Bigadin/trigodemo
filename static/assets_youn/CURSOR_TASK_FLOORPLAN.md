# Cursor Task — Floor Plan Viewer (Mall + Site Logistique)

Objectif : intégrer dans l’app une page **Plans** permettant d’afficher un plan (image), d’y placer des caméras (markers), et de cliquer pour ouvrir un panneau de détails. Prévoir 2 plans de démo :
- **Mall floor plan** (plan simplifié)
- **Site logistique** (plan technique)

Les images de démo sont déjà disponibles localement :
- Mall : `/mnt/data/mall-floor-plan-map.webp`
- Logistique : `/mnt/data/abattoir-bergerac.png`

---

## 1) Portée MVP (à livrer)

### 1.1 Affichage plan
- Charger un plan à partir d’un objet `FloorPlan` (voir modèle plus bas).
- Afficher l’image avec :
  - **zoom** molette (desktop) / pinch (si supporté)
  - **pan** (drag sur la scène)
  - bouton **Reset view** (zoom=1, pan=0)
- Le rendu doit rester fluide sur le plan logistique (image plus lourde).

### 1.2 Markers caméra
- Afficher une liste de caméras au-dessus du plan.
- Chaque caméra a :
  - `id`, `name`
  - `x`, `y` en coordonnées **normalisées** (0..1) par rapport à l’image d’origine
  - `status` ∈ `online | offline | alert`
  - optionnel : `rotationDeg` (préparer mais pas obligatoire pour MVP)
- Interactions :
  - **clic** sur un marker : sélection + ouverture d’un panneau latéral (ou modal) avec les infos caméra
  - **drag & drop** du marker en mode édition (toggle “Edit mode”)
  - bouton “Add camera” : clique sur le plan pour déposer un marker à la position du clic
  - bouton “Delete” sur la caméra sélectionnée

### 1.3 Persistance simple
- Sauvegarder la configuration dans `localStorage` (OK pour démo) :
  - floor plans
  - caméras par plan
- Prévoir une API plus tard : isoler la couche stockage dans un petit service.

---

## 2) Recommandation technique

### Option recommandée : `react-konva`
- Motifs :
  - gestion native du **pan/zoom**
  - overlays (markers) simples
  - bonne perf sur grandes images

Dépendances (si pas déjà) :
- `react-konva` et `konva`

Si la stack ne permet pas Konva, fallback :
- `<div>` + `<img>` transformée (CSS transform) + markers en `position:absolute` recalculés.

---

## 3) Structure de fichiers proposée

- `src/features/floorplan/`
  - `FloorPlanPage.tsx` (écran avec tabs Mall / Logistique)
  - `FloorPlanViewer.tsx` (composant réutilisable)
  - `useFloorPlanState.ts` (state + persistance localStorage)
  - `types.ts`
  - `mockData.ts` (démo: 2 plans + caméras initiales)
  - `storage.ts` (wrapper localStorage)

---

## 4) Modèle de données

```ts
export type FloorPlanId = string;
export type CameraId = string;

export type FloorPlan = {
  id: FloorPlanId;
  name: string;
  imageSrc: string; // chemin local ou URL
};

export type CameraStatus = "online" | "offline" | "alert";

export type FloorPlanCamera = {
  id: CameraId;
  floorPlanId: FloorPlanId;
  name: string;
  x: number; // 0..1
  y: number; // 0..1
  status: CameraStatus;
  rotationDeg?: number;
};
```

---

## 5) Données de démo (à intégrer)

### 5.1 Plans
- Mall:
  - id: `mall`
  - name: `Mall (Demo)`
  - imageSrc: `/mnt/data/mall-floor-plan-map.webp`
- Logistique:
  - id: `logistics`
  - name: `Site logistique (Demo)`
  - imageSrc: `/mnt/data/abattoir-bergerac.png`

### 5.2 Caméras initiales (exemple)
Placer 4 caméras par plan, coordonnées normalisées approximatives :

```ts
export const floorPlans: FloorPlan[] = [
  { id: "mall", name: "Mall (Demo)", imageSrc: "/mnt/data/mall-floor-plan-map.webp" },
  { id: "logistics", name: "Site logistique (Demo)", imageSrc: "/mnt/data/abattoir-bergerac.png" },
];

export const initialCameras: FloorPlanCamera[] = [
  // Mall
  { id: "cam-m-1", floorPlanId: "mall", name: "Mall — Entrée", x: 0.18, y: 0.20, status: "online" },
  { id: "cam-m-2", floorPlanId: "mall", name: "Mall — Food court", x: 0.78, y: 0.68, status: "online" },
  { id: "cam-m-3", floorPlanId: "mall", name: "Mall — Cinéma", x: 0.62, y: 0.22, status: "offline" },
  { id: "cam-m-4", floorPlanId: "mall", name: "Mall — Couloir", x: 0.46, y: 0.48, status: "alert" },

  // Logistique
  { id: "cam-l-1", floorPlanId: "logistics", name: "Log — Réception", x: 0.20, y: 0.18, status: "online" },
  { id: "cam-l-2", floorPlanId: "logistics", name: "Log — Stockage", x: 0.57, y: 0.30, status: "online" },
  { id: "cam-l-3", floorPlanId: "logistics", name: "Log — Production 2", x: 0.52, y: 0.63, status: "offline" },
  { id: "cam-l-4", floorPlanId: "logistics", name: "Log — Sortie", x: 0.88, y: 0.22, status: "alert" },
];
```

---

## 6) Détails d’implémentation (Konva)

### 6.1 Conversion coordonnées normalisées → pixels image
- Au rendu : `px = x * imageWidth`, `py = y * imageHeight`
- Au drop/drag : reconvertir en normalisé, clamp 0..1

### 6.2 Pan/Zoom
- Maintenir `scale` et `position {x,y}` pour le Stage (ou Group).
- Molette : zoom centré sur la souris.
- Drag du fond : pan.
- Bouton reset : `scale=1`, `position={0,0}`.

### 6.3 Mode édition
- Toggle `isEditMode`.
- Quand `isEditMode` :
  - markers draggable
  - clic sur le plan (fond) : ajoute une caméra à la position du pointeur

---

## 7) UI attendue (simple)

- En-tête : `Plans`
- Tabs ou dropdown : `Mall (Demo)` / `Site logistique (Demo)`
- Boutons :
  - `Edit mode` (toggle)
  - `Add camera`
  - `Reset view`
- Panneau latéral (à droite) affiché quand une caméra est sélectionnée :
  - nom
  - status (badge)
  - x/y (normalisés)
  - boutons : `Delete`, `Close`

---

## 8) Critères d’acceptation

- Les 2 plans s’affichent.
- Le zoom/pan fonctionne.
- Les caméras sont visibles et cliquables.
- En mode édition : on peut déplacer, ajouter, supprimer.
- Les modifications persistent après refresh (localStorage).

---

## 9) Notes
- Ne pas implémenter d’auth ni backend pour l’instant.
- Ne pas coder de “champ de vision” (cone) pour MVP, mais laisser `rotationDeg` dans le modèle.
