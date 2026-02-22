# Cursor — Réduire l'OOM et pérenniser l'éditeur

## 1. Paramètres déjà appliqués (`.vscode/settings.json`)

- `extensions.autoUpdate: false` — pas de mise à jour auto des extensions
- `git.autorefresh: false` — moins de requêtes Git en arrière-plan
- `editor.minimap.enabled: false` — économise de la RAM
- `editor.smoothScrolling: false` — moins de charge CPU
- `workbench.list.smoothScrolling: false` — idem

## 2. Désactiver les extensions

### Option A : Désactiver manuellement
1. `Ctrl+Shift+X` (Extensions)
2. Clic droit sur chaque extension → **Désactiver** ou **Désactiver (Workspace)**
3. Priorité aux extensions lourdes : linters, formatters, thèmes complexes, Git graphiques, etc.

### Option B : Lancer Cursor sans extensions
```powershell
cursor --disable-extensions
```
Ou créer un raccourci Windows avec cette commande pour une session légère.

### Option C : Profil minimal
1. `Ctrl+Shift+P` → "Profiles: Create Profile"
2. Créer un profil "Minimal" sans extensions
3. Basculer sur ce profil quand tu codes sur ce projet

## 3. Autres pistes anti-OOM

- **Nettoyer les chats** : supprimer les anciens dossiers dans `%USERPROFILE%\.cursor\chat-sessions`
- **Fermer les onglets** : garder peu de fichiers ouverts
- **Redémarrer** : relancer Cursor tous les 2–3 jours
- **Limiter les agents** : éviter trop de sous-agents en parallèle
