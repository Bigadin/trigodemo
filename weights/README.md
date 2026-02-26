# Bibliothèque des poids YOLO

Poids des modèles de détection par classe (human, voiture, vélo, camion, etc.).

## Structure

```
weights/
├── registry.json      # Métadonnées : version, date, performances
├── README.md
├── human.pt          # (ou à la racine du projet)
├── voiture.pt        # À ajouter
├── velo.pt           # À ajouter
└── camion.pt         # À ajouter
```

## Format du registry

Chaque entrée dans `registry.json` contient :

| Champ | Description |
|-------|-------------|
| `id` | Identifiant unique (snake_case) |
| `class` | Classe COCO ou custom |
| `label` | Libellé affiché |
| `file` | Nom du fichier .pt (relatif à `weights/` ou racine) |
| `version` | Version sémantique (ex: 1.0.0) |
| `updated_at` | Date de mise à jour (YYYY-MM-DD) |
| `status` | `active` \| `preparation` \| `deprecated` |
| `performance` | mAP50, mAP50-95, precision, recall, inference_ms |
| `notes` | Commentaires |

## Ajouter un nouveau poids

1. Placer le fichier `.pt` dans `weights/`
2. Ajouter une entrée dans `registry.json`
3. Renseigner les métriques après validation sur jeu de test

## Utilisation

- **Spec** : onglet « Poids » dans spec.html pour visualiser le catalogue
- **Backend** : `weights_registry.get_weights()` pour charger la config
