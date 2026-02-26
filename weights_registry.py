"""
Bibliothèque des poids YOLO — chargement et validation du registry.

Usage:
    from weights_registry import get_weights_registry, get_weight_path

    registry = get_weights_registry()
    path = get_weight_path("human")
"""
from pathlib import Path
import json

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
REGISTRY_FILE = WEIGHTS_DIR / "registry.json"
# Fallback: human.pt peut être à la racine (legacy)
ROOT_DIR = Path(__file__).resolve().parent


def _load_registry() -> dict:
    """Charge le registry depuis le fichier JSON."""
    if not REGISTRY_FILE.exists():
        return {"weights": [], "description": "", "updated_at": ""}
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_weights_registry() -> dict:
    """Retourne le registry complet (pour API / spec)."""
    return _load_registry()


def get_weight_path(weight_id: str) -> Path | None:
    """
    Retourne le chemin absolu du fichier .pt pour un poids donné.
    Cherche dans weights/ puis à la racine.
    """
    registry = _load_registry()
    for w in registry.get("weights", []):
        if w.get("id") == weight_id:
            fname = w.get("file")
            if not fname:
                return None
            # Chercher dans weights/ d'abord
            p = WEIGHTS_DIR / fname
            if p.exists():
                return p
            # Fallback racine (human.pt)
            p = ROOT_DIR / fname
            if p.exists():
                return p
            return WEIGHTS_DIR / fname  # chemin attendu même si fichier absent
    return None


def get_active_weights() -> list[dict]:
    """Retourne la liste des poids avec status=active."""
    registry = _load_registry()
    return [w for w in registry.get("weights", []) if w.get("status") == "active"]
