"""
Raw landmark storage shared by the server, the trainer and the evaluation scripts.

Every sample is stored as the 21 raw MediaPipe points (x, y, z), never as
pre-computed features. That way a new feature version can be computed for all
data, old and new, without recording anything again.

Two sources are combined:
  dataset/asl_now_landmarks.csv   the public ASL Now dataset (built by build_dataset.py)
  dataset/recordings/samples.csv  samples the team records in the app (/api/collect)

Columns in both files: META_COLUMNS followed by COORD_COLUMNS.
  label        letter or phrase name, e.g. "A" or "Thank You"
  kind         "letter" or "phrase"
  signer       who signed it ("asl_now" for the public dataset)
  hand         MediaPipe's handedness label ("Left" / "Right"), empty if unknown
  split        "test" keeps the sample out of training (new-signer test set)
  source       "asl_now" or "recorded"
  recorded_at  ISO timestamp, empty for the public dataset
"""

import csv
import os
from datetime import datetime, timezone

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "dataset")
ASL_NOW_DIR = os.path.join(DATA_DIR, "raw_asl_now")
ASL_NOW_CSV = os.path.join(DATA_DIR, "asl_now_landmarks.csv")
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
RECORDINGS_CSV = os.path.join(RECORDINGS_DIR, "samples.csv")

META_COLUMNS = ["label", "kind", "signer", "hand", "split", "source", "recorded_at"]
COORD_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
COLUMNS = META_COLUMNS + COORD_COLUMNS
KINDS = ("letter", "phrase")


def landmarks_to_array(landmarks):
    """MediaPipe landmark list (dicts with x, y, z) or array-like -> (21, 3) float array."""
    if landmarks is None:
        raise ValueError("Need 21 hand landmarks")
    if isinstance(landmarks, np.ndarray):
        arr = landmarks.astype(np.float64)
    else:
        if len(landmarks) < 21:
            raise ValueError("Need 21 hand landmarks")
        first = landmarks[0]
        if isinstance(first, dict):
            arr = np.array([[float(p["x"]), float(p["y"]), float(p.get("z") or 0.0)] for p in landmarks[:21]])
        else:
            arr = np.asarray(landmarks, dtype=np.float64)
    arr = arr.reshape(-1, 3)[:21]
    if arr.shape != (21, 3) or not np.all(np.isfinite(arr)):
        raise ValueError("Need 21 hand landmarks with finite x, y, z")
    return arr


def _clean(text, limit=60):
    return " ".join(str(text or "").split())[:limit]


def append_recording(label, landmarks, kind="letter", signer="", hand="", split=""):
    """Append one recorded sample. The caller holds the CSV lock."""
    arr = landmarks_to_array(landmarks)
    kind = kind if kind in KINDS else "letter"
    row = {
        "label": _clean(label),
        "kind": kind,
        "signer": _clean(signer) or "unknown",
        "hand": _clean(hand, 10),
        "split": "test" if str(split).strip().lower() == "test" else "",
        "source": "recorded",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update({c: f"{v:.6f}" for c, v in zip(COORD_COLUMNS, arr.reshape(-1))})
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    new_file = not os.path.exists(RECORDINGS_CSV)
    with open(RECORDINGS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


_cache = {}


def _read_rows(path):
    """Read a CSV as dicts, re-reading only when the file has changed."""
    if not os.path.exists(path):
        return []
    stat = os.stat(path)
    key = (stat.st_mtime_ns, stat.st_size)
    hit = _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _cache[path] = (key, rows)
    return rows


def load_samples(kind="letter", include_test=True, sources=("asl_now", "recorded")):
    """
    Return (points, meta) for one kind of sample.
      points: float array (N, 21, 3)
      meta:   list of dicts with the META_COLUMNS for each row
    """
    rows = []
    if "asl_now" in sources:
        rows += _read_rows(ASL_NOW_CSV)
    if "recorded" in sources:
        rows += _read_rows(RECORDINGS_CSV)
    rows = [r for r in rows if r.get("kind") == kind and r.get("label")]
    if not include_test:
        rows = [r for r in rows if r.get("split") != "test"]
    if not rows:
        return np.zeros((0, 21, 3)), []
    points = np.array([[float(r[c]) for c in COORD_COLUMNS] for r in rows]).reshape(-1, 21, 3)
    meta = [{k: r.get(k, "") for k in META_COLUMNS} for r in rows]
    return points, meta


def counts(kind="letter", label=None):
    """Sample counts for the UI: total for a kind, or for one label."""
    total = 0
    for path in (ASL_NOW_CSV, RECORDINGS_CSV):
        for r in _read_rows(path):
            if r.get("kind") == kind and (label is None or r.get("label") == label):
                total += 1
    return total
