"""
Raw landmark storage shared by the server, the trainers and the evaluation scripts.

Every sample is stored as raw MediaPipe points (x, y, z), never as pre-computed
features. That way a new feature version can be computed for all data, old and
new, without recording anything again.

Two sources are combined:
  dataset/asl_now_landmarks.csv   the public ASL Now dataset (built by build_dataset.py)
  dataset/recordings/samples.csv  samples the team records in the app (/api/collect)

Columns
  label        letter or sign name, e.g. "A" or "Thank You"
  kind         "letter" (fingerspelling), "phrase" (ASL word signs) or "isl" (ISL signs)
  signer       who signed it ("asl_now" for the public dataset)
  hand         MediaPipe's handedness label for the first hand ("Left" / "Right")
  hand2        handedness of the second hand, empty when only one hand was seen
  split        "test" keeps the sample out of training (new-signer test set)
  source       "asl_now" or "recorded"
  recorded_at  ISO timestamp, empty for the public dataset
  x0 .. z20    the first hand's 21 points
  bx0 .. bz20  the second hand's 21 points (empty for one hand)
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

META_COLUMNS = ["label", "kind", "signer", "hand", "hand2", "split", "source", "recorded_at"]
COORD_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
COORD2_COLUMNS = ["b" + c for c in COORD_COLUMNS]
COLUMNS = META_COLUMNS + COORD_COLUMNS + COORD2_COLUMNS
KINDS = ("letter", "phrase", "isl")


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


def parse_hands(payload):
    """
    Request body -> list of (points (21, 3), handedness) for up to two hands.
    Accepts {"hands": [{"landmarks": [...], "handedness": "Left"}, ...]} or the
    older {"landmarks": [...], "hand": "Right"}.
    """
    hands = []
    for h in (payload.get("hands") or [])[:2]:
        if isinstance(h, dict) and h.get("landmarks") is not None:
            hands.append((landmarks_to_array(h["landmarks"]), _clean(h.get("handedness"), 10)))
    if not hands and payload.get("landmarks") is not None:
        hands.append((landmarks_to_array(payload["landmarks"]), _clean(payload.get("hand"), 10)))
    if not hands:
        raise ValueError("Need 21 hand landmarks")
    return hands


def _clean(text, limit=60):
    return " ".join(str(text or "").split())[:limit]


def _migrate_header(path):
    """Rewrite an older recordings file so it has every current column."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == COLUMNS:
            return
        rows = list(reader)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})


def append_recording(label, hands, kind="letter", signer="", split=""):
    """Append one recorded sample (one or two hands). The caller holds the CSV lock."""
    if not hands:
        raise ValueError("Need 21 hand landmarks")
    kind = kind if kind in KINDS else "letter"
    row = {
        "label": _clean(label),
        "kind": kind,
        "signer": _clean(signer) or "unknown",
        "hand": hands[0][1],
        "hand2": hands[1][1] if len(hands) > 1 else "",
        "split": "test" if str(split).strip().lower() == "test" else "",
        "source": "recorded",
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    row.update({c: f"{v:.6f}" for c, v in zip(COORD_COLUMNS, hands[0][0].reshape(-1))})
    if len(hands) > 1:
        row.update({c: f"{v:.6f}" for c, v in zip(COORD2_COLUMNS, hands[1][0].reshape(-1))})
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    new_file = not os.path.exists(RECORDINGS_CSV)
    if not new_file:
        _migrate_header(RECORDINGS_CSV)
    with open(RECORDINGS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in COLUMNS})
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


def _rows(kind, include_test, sources):
    rows = []
    if "asl_now" in sources:
        rows += _read_rows(ASL_NOW_CSV)
    if "recorded" in sources:
        rows += _read_rows(RECORDINGS_CSV)
    rows = [r for r in rows if r.get("kind") == kind and r.get("label")]
    if not include_test:
        rows = [r for r in rows if r.get("split") != "test"]
    return rows


def _meta(rows):
    return [{k: r.get(k) or "" for k in META_COLUMNS} for r in rows]


def load_samples(kind="letter", include_test=True, sources=("asl_now", "recorded")):
    """
    First hand only, for the letter model.
      points: float array (N, 21, 3)
      meta:   list of dicts with the META_COLUMNS for each row
    """
    rows = _rows(kind, include_test, sources)
    if not rows:
        return np.zeros((0, 21, 3)), []
    points = np.array([[float(r[c]) for c in COORD_COLUMNS] for r in rows]).reshape(-1, 21, 3)
    return points, _meta(rows)


def load_two_hand_samples(kind="phrase", include_test=True, sources=("recorded",)):
    """
    Both hands, for the phrase / ISL models.
      points:  float array (N, 2, 21, 3); the second hand is zeros when absent
      hands:   list of (handedness_1, handedness_2) strings
      meta:    list of dicts with the META_COLUMNS for each row
    """
    rows = _rows(kind, include_test, sources)
    if not rows:
        return np.zeros((0, 2, 21, 3)), [], []
    points = np.zeros((len(rows), 2, 21, 3))
    for i, r in enumerate(rows):
        points[i, 0] = np.array([float(r[c]) for c in COORD_COLUMNS]).reshape(21, 3)
        if r.get("bx0"):
            points[i, 1] = np.array([float(r[c]) for c in COORD2_COLUMNS]).reshape(21, 3)
    hands = [((r.get("hand") or ""), (r.get("hand2") or "") if r.get("bx0") else "") for r in rows]
    return points, hands, _meta(rows)


def counts(kind="letter", label=None):
    """Sample counts for the UI: total for a kind, or for one label."""
    total = 0
    for path in (ASL_NOW_CSV, RECORDINGS_CSV):
        for r in _read_rows(path):
            if r.get("kind") == kind and (label is None or r.get("label") == label):
                total += 1
    return total
