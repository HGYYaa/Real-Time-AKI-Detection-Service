"""
This module encapsulates the persistence layer for the AKI detection system.

It provides a minimal key–value interface built on Python's ``dbm`` module.

Data stored:
- Creatinine histories: stored under the MRN as the key.
  Values are JSON-encoded lists of ``(timestamp, creatinine)`` pairs.
- Patient gender (optional): stored under the key ``gender:{MRN}`` as a plain string.

Example usage::

    from . import storage

    storage.save_history("123", [("20240101", 100.0), ("20240103", 110.0)])
    history = storage.load_history("123")

    storage.save_gender("123", "F")
    gender = storage.load_gender("123")

    all_histories = storage.load_all()
    all_genders = storage.load_all_genders()
"""


from __future__ import annotations

import dbm
import json
import os
import glob
from typing import Dict, List, Tuple, Optional

_DEFAULT_DBM_PATH = "patient_history.db"
_GENDER_PREFIX = "gender:"


def _db_path() -> str:
    return os.environ.get("DBM_PATH", _DEFAULT_DBM_PATH)

def db_exists() -> bool:
    """
    Return True if the DBM backing files already exist.
    """
    base = _db_path()
    if os.path.exists(base):
        return True
    matches = glob.glob(base + "*")
    return len(matches) > 0


def save_history(mrn: str, history: List[Tuple[str, float]]) -> None:
    """
    Persist a patient's creatinine history.

    Stored format: JSON string of [[timestamp, value], ...]
    """
    if not mrn:
        return
    payload = json.dumps(history, separators=(",", ":"))
    with dbm.open(_db_path(), "c") as db:
        db[mrn] = payload.encode("utf-8")


def load_history(mrn: str) -> List[Tuple[str, float]]:
    """Load a patient's history. Returns [] if missing."""
    if not mrn:
        return []
    with dbm.open(_db_path(), "c") as db:
        if mrn.encode("utf-8") in db:
            raw = db[mrn.encode("utf-8")].decode("utf-8")
            data = json.loads(raw)
            return [(str(ts), float(val)) for ts, val in data]
    return []


def load_all() -> Dict[str, List[Tuple[str, float]]]:
    """Load all MRN->history entries from DBM (excluding gender keys)."""
    out: Dict[str, List[Tuple[str, float]]] = {}
    with dbm.open(_db_path(), "c") as db:
        for k in db.keys():
            key = k.decode("utf-8", errors="ignore")
            if key.startswith(_GENDER_PREFIX):
                continue
            raw = db[k].decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
                out[key] = [(str(ts), float(val)) for ts, val in data]
            except Exception:
                # Skip malformed values without breaking load
                continue
    return out


def save_gender(mrn: str, gender: str) -> None:
    """Persist gender for a patient."""
    if not mrn or not gender:
        return
    key = f"{_GENDER_PREFIX}{mrn}"
    with dbm.open(_db_path(), "c") as db:
        db[key] = gender.encode("utf-8")


def load_gender(mrn: str) -> Optional[str]:
    """Load gender for a patient; returns None if missing."""
    if not mrn:
        return None
    key = f"{_GENDER_PREFIX}{mrn}".encode("utf-8")
    with dbm.open(_db_path(), "c") as db:
        if key in db:
            return db[key].decode("utf-8", errors="ignore") or None
    return None


def load_all_genders() -> Dict[str, str]:
    """Load all persisted genders."""
    out: Dict[str, str] = {}
    with dbm.open(_db_path(), "c") as db:
        for k in db.keys():
            key = k.decode("utf-8", errors="ignore")
            if not key.startswith(_GENDER_PREFIX):
                continue
            mrn = key[len(_GENDER_PREFIX):]
            out[mrn] = db[k].decode("utf-8", errors="ignore")
    return out