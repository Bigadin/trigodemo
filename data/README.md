# Données de session (ignorées par git)

Ce dossier contient les fichiers JSON générés par l'application lors de l'utilisation :

| Fichier | Description |
|---------|-------------|
| `presence.json` | Temps de présence cumulés par zone |
| `counting.json` | Config et état du comptage par vidéo |
| `zones.json` | Polygones des zones par vidéo |
| `benefits.json` | Bénéfices configurés |
| `cameras.json` | Caméras (webcam, RTSP) |
| `lieux.json`, `sites.json` | Hiérarchie lieux/sites |
| `counting_params.json` | Paramètres algorithme (optionnel) |
| `audit_log.jsonl` | Journal des événements CRUD |

**Ces fichiers sont ignorés par git** (`data/*.json`, `data/*.jsonl`) — chaque développeur a ses propres données de session. L'app crée les fichiers à la volée au premier usage.
