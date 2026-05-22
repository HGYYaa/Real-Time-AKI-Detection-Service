import warnings
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

_MODEL = None
_SCALER = None
_FEATURES: List[str] = []

_THRESHOLD = 0.67
_DEFAULT_AGE = 0.0


def initialize() -> None:
    """Loads model from disk"""
    global _MODEL, _SCALER, _FEATURES

    base_dir = Path(__file__).parent

    _MODEL = joblib.load(base_dir / "inference_model.pkl").get("model")
    _SCALER = joblib.load(base_dir / "scaler.pkl")
    _FEATURES = joblib.load(base_dir / "feature_columns.pkl")


def _parse_history(history: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Parses timestamps and values to numpy arrays."""
    ts_list, val_list = [], []

    for t_str, v_str in history:
        t_str = str(t_str)
        if len(t_str) < 12: continue
        try:
            # Fast manual parsing for 'YYYYMMDDHHMM'
            dt = datetime(
                int(t_str[:4]), int(t_str[4:6]), int(t_str[6:8]),
                int(t_str[8:10]), int(t_str[10:12])
            )
            ts_list.append(dt)
            val_list.append(float(v_str))
        except (ValueError, IndexError):
            continue

    return np.array(ts_list), np.array(val_list)


def _compute_features(gender: str, ts: np.ndarray, vals: np.ndarray) -> dict:
    """Calculates statistical features from patient history."""
    if len(vals) == 0: return {}

    last = vals[-1]
    # Base is median of previous values, or last if only 1 exists
    base = np.median(vals[:-1]) if len(vals) > 1 else last

    slope = 0.0
    if len(vals) >= 3:
        t_days = np.array([t.timestamp() / 86400.0 for t in ts])
        # Center data to avoid floating point errors
        x = t_days - t_days.mean()
        vx = np.var(x)
        if vx > 1e-9:
            slope = np.mean(x * (vals - vals.mean())) / vx

    return {
        "age": _DEFAULT_AGE,
        "sex_n": 1.0 if str(gender).strip().lower() in {"f", "female"} else 0.0,
        "last": last,
        "base": base,
        "ratio": (last / base) if base != 0 else 0.0,
        "delta": last - base,
        "max": np.max(vals),
        "min": np.min(vals),
        "mean": np.mean(vals),
        "slope": slope,
    }


def predict(gender: str, history: list) -> bool:
    """
    Main inference entry point.
    Returns True if AKI probability >= threshold.
    """
    try:
        ts, vals = _parse_history(history)
        feats = _compute_features(gender, ts, vals)

        # Create feature vector (default 0.0 for missing keys)
        X = np.array([[feats.get(col, 0.0) for col in _FEATURES]])

        # Scale and Predict
        prob = _MODEL.predict_proba(_SCALER.transform(X))[0, 1]
        return prob >= _THRESHOLD

    except Exception:
        return False
