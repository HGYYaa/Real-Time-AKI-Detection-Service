"""
This module is responsible for tracking patient state and creatinine
measurements.  It exposes two public functions:

1.initialize: Load historical data from disk and reset all in-memory
  structures.  This should be called once at start-up.
2.process_HL7: Parse a single HL7 message and update the internal state
  accordingly. When a new creatinine result arrives it returns a tuple
  ``(mrn, history, gender)`` for the caller to pass to inference.

Persistence note: this module does not write to disk on the critical path.
The caller should invoke ``save_patient_history(mrn)`` after alerting.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

from . import storage  # dbm persistence layer

__all__ = ["initialize", "process_HL7", "save_patient_history"]


_patient_history: Dict[str, List[Tuple[str, float]]] = {}
_current_in_hospital: set[str] = set()
_patient_gender: Dict[str, str] = {}


def _parse_fields(segment: str) -> List[str]:
    return segment.split("|")


def _extract_mrn_from_pid(pid_fields: List[str]) -> Optional[str]:
    # PID.3 -> split index 3
    if len(pid_fields) > 3 and pid_fields[3].strip():
        return pid_fields[3].strip()
    return None


def _extract_gender_from_pid(pid_fields: List[str]) -> Optional[str]:
    # PID.8 -> split index 8 (PID|...|Sex|...)
    if len(pid_fields) > 8 and pid_fields[8].strip():
        return pid_fields[8].strip()
    return None


def _obx_is_creatinine(obx_fields: List[str]) -> bool:
    # OBX.3 -> split index 3
    if len(obx_fields) > 3 and obx_fields[3].strip():
        return obx_fields[3].strip().upper() == "CREATININE"
    return False


def _extract_obr_time(obr_fields: List[str]) -> Optional[str]:
    # OBR.7 -> split index 7
    if len(obr_fields) > 7 and obr_fields[7].strip():
        return obr_fields[7].strip()
    return None


def _extract_obx_value(obx_fields: List[str]) -> Optional[float]:
    # OBX.5 -> split index 5
    if len(obx_fields) > 5 and obx_fields[5].strip():
        try:
            return float(obx_fields[5].strip())
        except ValueError:
            return None
    return None


def _load_history_from_csv(path: str) -> None:
    """
    Load historical creatinine results from a CSV file into memory, WITHOUT sorting.

    Assumptions:
    - CSV is already sorted in non-decreasing timestamp order per MRN.
    - If out-of-order rows are found for a MRN, log and skip.

    This avoids expensive merges/sorts at startup.
    """
    global _patient_history
    _patient_history.clear()

    if not os.path.exists(path):
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # MRN
            mrn = None
            for key in row:
                if key.lower() in {"mrn", "patient", "id", "pid"}:
                    mrn = row[key].strip()
                    break
            if not mrn:
                continue

            # Timestamp
            timestamp = None
            for key in row:
                if key.lower() in {"timestamp", "time", "date", "datetime"}:
                    timestamp = row[key].strip()
                    break
            if not timestamp:
                continue

            # Creatinine
            value_str = None
            for key in row:
                if key.lower() in {"creatinine", "value", "result"}:
                    value_str = row[key].strip()
                    break
            if not value_str:
                continue

            try:
                value = float(value_str)
            except ValueError:
                continue

            history = _patient_history.setdefault(mrn, [])
            if not history:
                history.append((timestamp, value))
                continue

            last_ts, last_val = history[-1]

            # exact duplicate of last -> skip
            if last_ts == timestamp and last_val == value:
                continue

            # in-order append
            if timestamp >= last_ts:
                history.append((timestamp, value))
            else:
                print(
                    f"[data] WARNING: CSV out-of-order for MRN={mrn}: "
                    f"new_ts={timestamp} < last_ts={last_ts}. Skipping row."
                )


def initialize() -> None:
    """
    Initialise module state and load historical data.

    Data consistency strategy:
    - If DBM files already exist: treat DB as source of truth; load from DB only.
    - If DBM files do NOT exist (first run): load from CSV and seed DB once.
    - CSV is assumed to always exist; we do not branch on "missing CSV".
    """
    global _patient_history, _current_in_hospital, _patient_gender
    _patient_history = {}
    _current_in_hospital = set()
    _patient_gender = {}

    # If DB already exists, always load from DB (ignore CSV to avoid overwriting data)
    if storage.db_exists():
        try:
            _patient_history = {mrn: list(hist) for mrn, hist in storage.load_all().items()}
            _patient_gender = dict(storage.load_all_genders())
        except Exception as e:
            print(f"[data] WARNING: failed to load from DBM: {e}")
            _patient_history = {}
            _patient_gender = {}
        return

    # First run: seed DB from CSV (CSV assumed to exist)
    csv_path = os.environ.get("HISTORY_PATH", "/data/history.csv")

    _load_history_from_csv(csv_path)

    try:
        for mrn, history in _patient_history.items():
            storage.save_history(mrn, history)
        # gender seeding is optional; usually gender arrives from ADT, so leave empty
    except Exception as e:
        print(f"[data] WARNING: failed to seed DBM from CSV: {e}")



def save_patient_history(mrn: str) -> None:
    """
    Explicit persistence hook. Call from main if you want to delay writes.
    """
    if not mrn:
        return
    hist = _patient_history.get(mrn, [])
    try:
        storage.save_history(mrn, hist)
    except Exception as e:
        print(f"[data] WARNING: save_patient_history failed for MRN={mrn}: {e}")


def _handle_admit(pid_fields: List[str]) -> None:
    mrn = _extract_mrn_from_pid(pid_fields)
    if not mrn:
        return
    _current_in_hospital.add(mrn)
    _patient_history.setdefault(mrn, [])

    gender = _extract_gender_from_pid(pid_fields)
    if gender:
        _patient_gender[mrn] = gender
        try:
            storage.save_gender(mrn, gender)
        except Exception as e:
            print(f"[data] WARNING: failed to persist gender for MRN={mrn}: {e}")

    # Optionally ensure empty history exists in DBM
    try:
        if not storage.load_history(mrn):
            storage.save_history(mrn, _patient_history[mrn])
    except Exception:
        # Do not block
        pass


def _handle_discharge(pid_fields: List[str]) -> None:
    mrn = _extract_mrn_from_pid(pid_fields)
    if not mrn:
        return
    _current_in_hospital.discard(mrn)
    # keep gender/history


def _handle_creatinine(
    pid_fields: List[str],
    obr_fields: List[str],
    obx_fields: List[str],
) -> Optional[Tuple[str, List[Tuple[str, float]], str]]:
    """
    Update history using O(1) append logic and return (mrn, history, gender)
    only if currently admitted.
    """
    mrn = _extract_mrn_from_pid(pid_fields)
    if not mrn:
        return None

    timestamp = _extract_obr_time(obr_fields)
    if not timestamp:
        return None

    if not _obx_is_creatinine(obx_fields):
        return None

    value = _extract_obx_value(obx_fields)
    if value is None:
        return None

    history = _patient_history.setdefault(mrn, [])

    # first record
    if not history:
        history.append((timestamp, value))
    else:
        last_ts, last_val = history[-1]

        # duplicate last
        if timestamp == last_ts and value == last_val:
            pass
        # same timestamp but different value -> log + skip
        elif timestamp == last_ts and value != last_val:
            print(
                f"[data] WARNING: same-timestamp different value for MRN={mrn}: "
                f"ts={timestamp}, last_val={last_val}, new_val={value}. Skipping."
            )
        # out-of-order -> log + skip
        elif timestamp < last_ts:
            print(
                f"[data] WARNING: out-of-order creatinine for MRN={mrn}: "
                f"new_ts={timestamp} < last_ts={last_ts}. Skipping."
            )
        else:
            # in-order append
            history.append((timestamp, value))

    gender = _patient_gender.get(mrn, "U")
    return mrn, list(history), gender



def process_HL7(raw_HL7: str) -> Optional[Tuple[str, List[Tuple[str, float]], str]]:
    """
    Parse a single HL7 message and update internal state.

    Returns:
      - None if no inference is needed
      - (mrn, history, gender) if creatinine result should be forwarded
    """
    if not raw_HL7:
        return None

    message = raw_HL7.replace("\n", "\r")
    segments = [seg for seg in message.split("\r") if seg]
    if not segments:
        return None

    msh_fields = _parse_fields(segments[0])
    message_type = msh_fields[8].strip() if len(msh_fields) > 8 else ""

    pid_seg = next((s for s in segments if s.startswith("PID|")), None)
    if not pid_seg:
        return None
    pid_fields = _parse_fields(pid_seg)

    if message_type == "ADT^A01":
        _handle_admit(pid_fields)
        return None

    if message_type == "ADT^A03":
        _handle_discharge(pid_fields)
        return None

    if message_type == "ORU^R01":
        obr_seg = next((s for s in segments if s.startswith("OBR|")), None)
        obx_seg = next((s for s in segments if s.startswith("OBX|")), None)
        if not obr_seg or not obx_seg:
            return None
        obr_fields = _parse_fields(obr_seg)
        obx_fields = _parse_fields(obx_seg)
        return _handle_creatinine(pid_fields, obr_fields, obx_fields)

    return None