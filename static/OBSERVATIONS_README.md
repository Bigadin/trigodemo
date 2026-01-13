# Observations Tracker

Application standalone de gestion de tickets/observations pour l'équipe.

## 🚀 Utilisation

Ouvre simplement `observations.html` dans un navigateur moderne. Aucune installation nécessaire.

## ✨ Fonctionnalités

- ✅ Création de tickets avec criticité (K1/K2/K3)
- ✅ Édition directe dans le tableau
- ✅ Filtres par criticité, état ticket, recherche texte
- ✅ Tri par colonnes (cliquez sur les en-têtes)
- ✅ Modal pour détails de reproduction et résolution
- ✅ Export JSON des données
- ✅ Sauvegarde automatique en localStorage

## 📦 Déploiement

### GitHub Pages (gratuit)

1. Push le repo sur GitHub
2. Va dans Settings → Pages
3. Sélectionne la branche `main` et le dossier `/` (root)
4. Le site sera accessible sur `https://ton-username.github.io/nom-du-repo/`

### Scaleway Static Site Hosting

1. Crée un bucket Object Storage sur Scaleway
2. Active le "Static Site Hosting"
3. Upload `observations.html` (renomme-le en `index.html` si besoin)
4. Configure le domaine si nécessaire

## 💾 Stockage

Les données sont stockées dans le **localStorage** du navigateur. Pour partager entre équipe, il faudra ajouter une API backend + base de données (voir section "Évolution").

## 🔄 Évolution future

Pour rendre l'application collaborative (multi-utilisateurs) :
- Ajouter une API backend (FastAPI/Flask)
- Utiliser une base de données (PostgreSQL/Scaleway Database)
- Ajouter authentification (optionnel)

## 📝 Notes

- Les données sont stockées localement (localStorage)
- Chaque navigateur/domaine a son propre stockage
- Utilisez "Exporter JSON" pour sauvegarder/partager les données
