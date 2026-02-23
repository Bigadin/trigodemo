# Checklist de régression — Migration Frontend

> À valider après chaque phase de migration.
> Réf : [FRONTEND_MIGRATION.md](./FRONTEND_MIGRATION.md) · [DEPENDENCY_MAP.md](./DEPENDENCY_MAP.md)

---

## Légende

- 🔴 **Bloquant** — La feature doit absolument marcher, sinon la phase n'est pas terminée
- 🟡 **Important** — Doit marcher, peut être fixé dans la foulée
- 🟢 **Nice-to-have** — Cosmétique ou edge case, peut attendre

---

## Phase 0 — Scaffolding

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 0.1 | `npm run dev` démarre sans erreur | 🔴 | Terminal : pas d'erreur, port 5173 accessible |
| 0.2 | Proxy API fonctionne | 🔴 | `fetch('/api/hierarchy')` depuis la console du navigateur retourne du JSON |
| 0.3 | L'ancien frontend (`localhost:8000`) marche toujours | 🔴 | Ouvrir `:8000`, tout fonctionne normalement |
| 0.4 | TypeScript compile sans erreur | 🔴 | `npx tsc --noEmit` sort 0 erreurs |
| 0.5 | Tokens CSS chargés | 🟡 | `getComputedStyle(document.body).getPropertyValue('--brand-01')` retourne une couleur |

---

## Phase 1 — Layout + Navigation

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 1.1 | 4 vues accessibles (Sites, Tracker, Analytics, Logs) | 🔴 | Cliquer chaque item de la topbar → la vue change |
| 1.2 | Sidebar visible sur toutes les vues | 🔴 | Barre latérale présente à gauche |
| 1.3 | URL change avec la navigation | 🟡 | Barre d'adresse reflète `/`, `/tracker`, etc. |
| 1.4 | Back/Forward navigateur fonctionne | 🟡 | Boutons navigateur changent la vue |
| 1.5 | La topbar highlight l'onglet actif | 🟢 | Classe CSS active sur le bon bouton |

---

## Phase 2 — Sites + Sidebar

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 2.1 | Liste des sites s'affiche au chargement | 🔴 | Sites "Lyon" et "Paris" visibles en grille |
| 2.2 | Toggle grille ↔ liste | 🔴 | Cliquer icônes grid/list → affichage change |
| 2.3 | Créer un site | 🔴 | Formulaire → submit → nouveau site apparaît |
| 2.4 | Supprimer un site | 🔴 | Menu contextuel → confirmer → site disparaît |
| 2.5 | Arbre explorateur : lieux → sites → caméras → bénéfices | 🔴 | Cliquer expand → enfants visibles |
| 2.6 | Collapse/expand persiste | 🟡 | Fermer un noeud, naviguer, revenir → toujours fermé |
| 2.7 | Cliquer un site dans l'arbre → sélection | 🔴 | Site mis en surbrillance + vue tracker s'ouvre |
| 2.8 | Breadcrumb : lieu → site → caméra | 🔴 | Fil d'ariane correct, cliquable |
| 2.9 | LOV dropdown dans le breadcrumb | 🟡 | Cliquer le lieu → dropdown avec les autres lieux |
| 2.10 | Badges caméras/bénéfices sur les cartes site | 🟢 | Compteurs corrects |

### Données à vérifier (API)
- `GET /api/hierarchy` retourne les 2 lieux, 2 sites, 3 caméras, 4 bénéfices
- `POST /api/lieux` crée bien un lieu dans `data/lieux.json`
- `DELETE /api/sites/{id}` supprime le site + cascade

---

## Phase 3 — Tracker core

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 3.1 | Sélectionner une caméra → vidéo/stream s'affiche | 🔴 | Cliquer caméra dans sidebar → `<img>` montre le flux |
| 3.2 | Canvas overlay se superpose à la vidéo | 🔴 | Zones dessinées visibles au-dessus du flux |
| 3.3 | Grille caméras du site | 🔴 | Toutes les caméras du site actif affichées en tuiles |
| 3.4 | Tuile caméra : menu ⋯ (renommer, supprimer) | 🟡 | Menu contextuel fonctionnel |
| 3.5 | Ajouter caméra (vidéo) | 🔴 | Modal → sélectionner vidéo → caméra créée |
| 3.6 | Ajouter caméra (webcam) | 🟡 | Détecter webcams → sélectionner → caméra créée |
| 3.7 | Upload vidéo | 🔴 | Choisir fichier → upload → vidéo disponible |
| 3.8 | KPIs header (étapes 1/2/3) compteurs corrects | 🟡 | Nombre de caméras, zones, présences actives |
| 3.9 | Liste bénéfices pour la caméra sélectionnée | 🔴 | Bénéfices listés avec badges skill + catégories |
| 3.10 | Onglets Overview / Details / Settings | 🔴 | Cliquer change le panneau droit |
| 3.11 | Polling zones fonctionne | 🔴 | Données de présence se mettent à jour (~1.2s) |
| 3.12 | Polling streams fonctionne | 🔴 | Badges streams actifs se mettent à jour (~2s) |

### Données à vérifier (API)
- `GET /api/zones/{video}` retourne les zones et présences
- `GET /api/streams` retourne les streams actifs
- `POST /api/cameras` crée la caméra dans `data/cameras.json`
- `GET /api/videos` liste les fichiers vidéo disponibles

---

## Phase 4 — DataRoom + Detection

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 4.1 | Bouton Start détection → stream démarre | 🔴 | Vidéo s'anime, badge "active" apparaît |
| 4.2 | Bouton Stop → stream s'arrête | 🔴 | Vidéo figée, badge disparaît |
| 4.3 | Stop All → tous les streams s'arrêtent | 🔴 | Tous les badges disparaissent |
| 4.4 | Cartes ROI DataRoom s'affichent | 🔴 | Pour chaque bénéfice actif, une carte avec métriques |
| 4.5 | Métriques live (temps présence, occupation %) | 🔴 | Valeurs se mettent à jour en temps réel |
| 4.6 | Cartes ROI : menu ⋯ fonctionnel | 🟡 | Fullscreen, détails, etc. |
| 4.7 | Comptage : sélection zone + mode | 🔴 | Dropdown zones peuplé, mode (ROI/simple) sélectionnable |
| 4.8 | Comptage : toggle on/off | 🔴 | Active/désactive le comptage, compteur réagit |
| 4.9 | Comptage : flip direction | 🟡 | La direction de comptage s'inverse |
| 4.10 | Comptage : reset | 🟡 | Compteur revient à 0 |
| 4.11 | Blur toggle | 🟡 | Active/désactive le floutage sur le flux |
| 4.12 | Anonymize toggle | 🟡 | Active/désactive l'anonymisation |
| 4.13 | Site Overview (multi-caméras) | 🔴 | Cartes de toutes les caméras avec statut |
| 4.14 | Settings : statistiques site (caméras, zones, temps run) | 🟢 | Valeurs correctes |

### Données à vérifier (API)
- `POST /api/stream/{video}/start` démarre la détection
- `GET /api/presence/{video}` retourne les zones avec `occupied`
- `GET /api/counting/{video}` retourne le compteur
- `GET /api/blur` retourne l'état blur

---

## Phase 5 — Editor canvas ⚠️

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 5.1 | Ouvrir l'éditeur (plein écran) | 🔴 | Modal overlay couvre tout l'écran |
| 5.2 | Frame vidéo affichée dans l'éditeur | 🔴 | Image de la vidéo visible |
| 5.3 | Preview play/pause | 🟡 | Bouton toggle → animation frame par frame |
| 5.4 | Outil "Include" : dessiner un polygone | 🔴 | Cliquer points → polygone vert se forme |
| 5.5 | Outil "Exclude" : dessiner une zone d'exclusion | 🔴 | Polygone rouge |
| 5.6 | Outil "Count line" : dessiner une ligne | 🔴 | Ligne avec flèche directionnelle |
| 5.7 | Outil "Select" : cliquer un polygone → sélection | 🔴 | Polygone mis en surbrillance |
| 5.8 | Déplacer un polygone (drag) | 🔴 | Grab → drag → polygone suit la souris |
| 5.9 | Déplacer un vertex (drag) | 🔴 | Grab vertex → drag → forme se déforme |
| 5.10 | Ajouter un point sur une arête | 🟡 | Double-clic arête → nouveau vertex |
| 5.11 | Supprimer un point (hover bar) | 🟡 | Hover → barre contextuelle → supprimer point |
| 5.12 | Supprimer une forme (hover bar) | 🔴 | Hover → barre contextuelle → supprimer forme |
| 5.13 | Undo / Redo | 🔴 | Ctrl+Z → dernière action annulée |
| 5.14 | Clear (supprimer tout) | 🟡 | Bouton clear → toutes les formes supprimées |
| 5.15 | Sauvegarder (PUT zones) | 🔴 | Bouton save → zones persistées côté backend |
| 5.16 | Fermer éditeur → zones visibles dans le tracker | 🔴 | Retour tracker, canvas overlay montre les zones |
| 5.17 | Autosave (après 650ms d'inactivité) | 🟡 | Modifier → attendre → saved automatiquement |
| 5.18 | Liste zones dans le panneau droit | 🔴 | Noms des zones listés, sélectionnables |
| 5.19 | Ajouter/supprimer zone depuis le panneau | 🔴 | Boutons + / corbeille fonctionnels |
| 5.20 | Raccourcis clavier (Ctrl+Z, Escape, etc.) | 🟢 | Touches fonctionnelles |

### Données à vérifier (API)
- `PUT /api/zones/{video}/{zone}` sauvegarde les polygones
- `POST /api/zones` crée une nouvelle zone
- `DELETE /api/zones/{video}/{zone}` supprime la zone
- `GET /api/videos/{video}/info` retourne les dimensions

---

## Phase 6 — Wizard bénéfice

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 6.1 | Ouvrir le wizard (bouton "+ Bénéfice") | 🔴 | Panneau wizard apparaît dans l'éditeur |
| 6.2 | Étape 1 — Choix skill (4 skills) | 🔴 | 4 cartes cliquables (detection, counting, heatmap, quality) |
| 6.3 | Étape 2 — Sous-skill + catégories | 🔴 | Liste sous-skills, arbre catégories avec checkboxes |
| 6.4 | Étape 3 — Dessin zones (intégré éditeur) | 🔴 | L'éditeur s'active avec l'outil adapté au skill |
| 6.5 | Étape 4 — Récap + nom + sauvegarde | 🔴 | Récap complet, champ nom, bouton sauvegarder |
| 6.6 | Sauvegarde crée le bénéfice (POST) | 🔴 | Bénéfice visible dans la liste après save |
| 6.7 | Édition d'un bénéfice existant | 🔴 | Cliquer "éditer" → wizard pré-rempli |
| 6.8 | Suppression d'un bénéfice | 🔴 | Cliquer "supprimer" → confirm → bénéfice retiré |
| 6.9 | Navigation entre étapes (retour/avancer) | 🟡 | Flèches ou onglets pour naviguer |
| 6.10 | Validation des étapes (disabled si incomplet) | 🟡 | Bouton "suivant" grisé si rien sélectionné |
| 6.11 | Icônes skills et catégories | 🟢 | SVG correctement affichées |

### Données à vérifier (API)
- `POST /api/benefits` crée dans `data/benefits.json`
- `PUT /api/benefits/{id}` met à jour
- `DELETE /api/benefits/{id}` supprime
- `GET /api/skills` retourne la config skills

---

## Phase 7 — Analytics + Logs

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 7.1 | Dashboard analytics s'affiche | 🔴 | Widgets visibles en grille |
| 7.2 | Heatmap canvas se dessine | 🔴 | Carte de chaleur visible |
| 7.3 | Barres divergentes se dessinent | 🟡 | Graphique barres |
| 7.4 | Barres stats animées | 🟡 | Animation d'entrée |
| 7.5 | Jauge circulaire animée | 🟡 | Animation rotation |
| 7.6 | Menu contextuel widget (fullscreen, delete) | 🟢 | Menu ⋯ fonctionnel |
| 7.7 | Hover tooltip sur les canvas | 🟢 | Tooltip avec valeurs au survol |
| 7.8 | Console logs charge les entrées | 🔴 | Logs affichés avec timestamp, niveau, catégorie |
| 7.9 | Filtres catégorie logs | 🔴 | Cliquer filtre → logs filtrés |
| 7.10 | Clear logs | 🟡 | Bouton clear → logs vidés |
| 7.11 | Auto-refresh logs (2s) | 🟡 | Nouveaux logs apparaissent automatiquement |
| 7.12 | Graphique performance | 🟡 | Canvas perf avec légendes |

### Données à vérifier (API)
- `GET /api/logs?limit=500&category=...` retourne les logs
- `GET /api/metrics?points=120` retourne les métriques perf
- `DELETE /api/logs` efface les logs

---

## Phase 8 — Polish + Nettoyage

| # | Test | Priorité | Comment vérifier |
|---|---|---|---|
| 8.1 | Build production réussit | 🔴 | `npm run build` → 0 erreurs |
| 8.2 | Backend sert le build Vite | 🔴 | `localhost:8000` affiche le nouveau frontend |
| 8.3 | Tous les tests Phase 1–7 passent sur le build prod | 🔴 | Vérifier chaque checklist |
| 8.4 | Ancien code supprimé | 🟡 | `static/js/index.js` supprimé |
| 8.5 | Pas de console.error/warn en production | 🟡 | DevTools → Console propre |
| 8.6 | Responsive topbar | 🟢 | Réduire la fenêtre → navigation reste utilisable |
| 8.7 | `spec.html` toujours accessible | 🟡 | `/static/spec.html` fonctionne |

---

## Tests transversaux (à vérifier à chaque phase)

| # | Test | Comment vérifier |
|---|---|---|
| T.1 | Pas de fuite mémoire | DevTools → Memory → Heap snapshot stable après navigation |
| T.2 | Pas de polling qui tourne quand la vue n'est pas active | Network tab → les requêtes s'arrêtent quand on quitte la vue |
| T.3 | Les accents s'affichent correctement partout | "Entrepôt", "Détection présence", "caméra" sans caractères cassés |
| T.4 | Le backend n'a pas été modifié | `git diff main.py` → aucun changement |
| T.5 | Les données JSON sont intactes | Vérifier `data/*.json` pas de corruption |

---

## Quand tester ?

| Moment | Action |
|---|---|
| **Après chaque composant** | Vérifier visuellement dans le navigateur |
| **Après chaque phase** | Dérouler la checklist complète de la phase |
| **Avant de commencer une nouvelle phase** | Re-tester la checklist de la phase précédente (régression) |
| **Avant Phase 8 (cleanup)** | Dérouler TOUTES les checklists (1→7) une dernière fois |
