"""
Couche de persistance centralisée — CRUD et journal d'événements.

Toutes les écritures passent par ce module pour :
- Centraliser les sauvegardes (zones, lieux, sites, benefits, etc.)
- Garantir un journal d'événements cohérent (audit_log.jsonl)
- Traçabilité : chaque modification est enregistrée avec entity, action, payload

Format événement CRUD :
{
  "ts": "ISO8601",
  "entity": "lieu|site|camera|benefit|zone|presence|counting|counting_params",
  "action": "created|updated|deleted|reset",
  "detail": "Description humaine",
  "level": "info|warn|error|success",
  "meta": { "id", ... },  // identifiants + contexte
  "payload": { ... }      // données pour traçabilité (optionnel, selon action)
}
"""

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# Paths — configurés par main au démarrage
DATA_DIR: Path = Path(__file__).parent / "data"
AUDIT_LOG_FILE: Path = DATA_DIR / "audit_log.jsonl"


def init(data_dir: Path) -> None:
    """Configure le répertoire de données (appelé par main au démarrage)."""
    global DATA_DIR, AUDIT_LOG_FILE
    DATA_DIR = Path(data_dir)
    AUDIT_LOG_FILE = DATA_DIR / "audit_log.jsonl"

_audit_log: deque = deque(maxlen=2000)
_audit_lock = threading.Lock()

# Entités CRUD reconnues
ENTITY_TYPES = {"lieu", "site", "camera", "benefit", "zone", "presence", "counting", "counting_params"}
CRUD_ACTIONS = {"created", "updated", "deleted", "reset"}


def _append_event(entry: dict) -> None:
    """Append event to in-memory buffer and append-only file."""
    with _audit_lock:
        _audit_log.append(entry)
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


def load_audit_log() -> None:
    """Charge le journal existant au démarrage."""
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _audit_log.append(json.loads(line))
        except Exception:
            pass


def get_audit_entries() -> list:
    """Retourne une copie des entrées du journal (pour API)."""
    with _audit_lock:
        return list(_audit_log)


def clear_audit_log() -> None:
    """Efface le journal (mémoire + fichier)."""
    with _audit_lock:
        _audit_log.clear()
        try:
            AUDIT_LOG_FILE.write_text("")
        except Exception:
            pass


def persist(
    save_fn: Callable[[], None],
    entity: str,
    action: str,
    detail: str = "",
    level: str = "info",
    meta: Optional[dict] = None,
    payload: Optional[dict] = None,
) -> None:
    """
    Persiste les données et enregistre l'événement dans le journal.

    Args:
        save_fn: Fonction à appeler pour écrire sur disque (ex: save_lieux)
        entity: Type d'entité (lieu, site, camera, benefit, zone, presence, counting, counting_params)
        action: created | updated | deleted | reset
        detail: Description humaine de l'événement
        level: info | warn | error | success
        meta: Métadonnées (id, ids, etc.)
        payload: Données structurées pour traçabilité (optionnel)
    """
    save_fn()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "entity": entity,
        "action": action,
        "detail": detail,
        "level": level,
    }
    if meta:
        entry["meta"] = meta
    if payload is not None:
        entry["payload"] = payload
    _append_event(entry)


def append_event(entry: dict) -> None:
    """Append a pre-built event (ex: seed/demo). Use persist() or audit_event() en priorité."""
    _append_event(entry)


def audit_event(
    category: str,
    action: str,
    detail: str = "",
    level: str = "info",
    meta: Optional[dict] = None,
) -> None:
    """
    Enregistre un événement non-CRUD (stream, detection, blur, video, system).

    Pour les événements CRUD, utiliser persist() à la place.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "action": action,
        "detail": detail,
        "level": level,
    }
    if meta:
        entry["meta"] = meta
    _append_event(entry)
