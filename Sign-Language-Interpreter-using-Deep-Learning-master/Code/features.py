"""
Turn MediaPipe's 21 hand landmarks into the feature vector the classifier uses.

Version 1 (original, 63 numbers)
    Every point minus the wrist. Removes where the hand is in the frame,
    but not how big it looks or how it is tilted.

Version 2 (current, 88 numbers)
    63  wrist-relative points divided by palm length (wrist to middle-finger
        knuckle), so the same sign gives the same numbers near or far from the camera
    10  distances between every pair of fingertips (also divided by palm length)
    15  bend angle at each finger joint, in radians

    Distances and angles do not change when the hand is mirrored or rotated,
    which is what separates look-alike letters such as D (straight index
    finger) and X (hooked index finger).

Two hands (phrase and ISL models, 181 numbers)
    88  version-2 features of the left hand  (zeros if it is not visible)
    88  version-2 features of the right hand (zeros if it is not visible)
     2  which hands are visible (1 / 0)
     3  position of the second wrist relative to the first, divided by palm length

The trainer stores the version it used inside the model file, and the server
computes features with that same version, so an old model keeps working.
"""

import numpy as np

from data import landmarks_to_array

FEATURE_VERSION = 2

WRIST, MIDDLE_KNUCKLE = 0, 9
FINGERTIPS = (4, 8, 12, 16, 20)
TIP_PAIRS = tuple((a, b) for i, a in enumerate(FINGERTIPS) for b in FINGERTIPS[i + 1:])
# (point before, joint, point after) for the three joints of each finger
JOINTS = (
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # pinky
)
FEATURE_SIZES = {1: 63, 2: 63 + len(TIP_PAIRS) + len(JOINTS)}


def _as_batch(points):
    p = np.asarray(points, dtype=np.float64)
    return p[None] if p.ndim == 2 else p


def features_v1(points):
    """(N, 21, 3) -> (N, 63): wrist-relative coordinates."""
    p = _as_batch(points)
    return (p - p[:, :1]).reshape(len(p), -1)


def normalise(points):
    """(N, 21, 3) -> wrist at the origin, palm length = 1."""
    p = _as_batch(points)
    q = p - p[:, :1]
    palm = np.linalg.norm(q[:, MIDDLE_KNUCKLE, :2], axis=1)
    palm[palm < 1e-6] = 1.0
    return q / palm[:, None, None]


def joint_angles(q):
    """Bend angle at each joint: theta = arccos(u.v / (|u||v|))."""
    out = []
    for a, b, c in JOINTS:
        u = q[:, a] - q[:, b]
        v = q[:, c] - q[:, b]
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
        out.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.stack(out, 1)


def fingertip_distances(q):
    return np.stack([np.linalg.norm(q[:, a] - q[:, b], axis=1) for a, b in TIP_PAIRS], 1)


def features_v2(points):
    """(N, 21, 3) -> (N, 88): scaled coordinates, fingertip distances, joint angles."""
    q = normalise(points)
    return np.hstack([q.reshape(len(q), -1), fingertip_distances(q), joint_angles(q)])


def batch_features(points, version=FEATURE_VERSION):
    if version == 1:
        return features_v1(points)
    if version == 2:
        return features_v2(points)
    raise ValueError(f"Unknown feature version {version}")


def landmarks_to_features(landmarks, version=FEATURE_VERSION):
    """One MediaPipe landmark list -> one feature vector (list of floats)."""
    return batch_features(landmarks_to_array(landmarks), version)[0].tolist()


HAND_SLOTS = ("Left", "Right")
TWO_HAND_SIZE = 2 * FEATURE_SIZES[2] + 2 + 3


def _slots(handedness_pair, present):
    """Map up to two detected hands to the Left / Right slots using MediaPipe's labels."""
    slots, used = [None, None], set()
    for i, label in enumerate(handedness_pair):
        if not present[i]:
            continue
        s = HAND_SLOTS.index(label) if label in HAND_SLOTS else None
        if s is None or s in used:
            s = 0 if 0 not in used else 1
        slots[i] = s
        used.add(s)
    return slots


def two_hand_features(points, hands):
    """
    points: (N, 2, 21, 3) with zeros for a missing second hand
    hands:  list of N (handedness_1, handedness_2) pairs
    -> (N, 181)
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    out = np.zeros((n, TWO_HAND_SIZE))
    size = FEATURE_SIZES[2]
    present = np.abs(points).sum(axis=(2, 3)) > 0
    per_hand = [features_v2(points[:, k]) for k in range(2)]
    for i in range(n):
        slots = _slots(hands[i], present[i])
        for k in range(2):
            if slots[k] is not None:
                out[i, slots[k] * size:(slots[k] + 1) * size] = per_hand[k][i]
                out[i, 2 * size + slots[k]] = 1.0
        if present[i].all():
            palm = np.linalg.norm(points[i, 0, MIDDLE_KNUCKLE, :2] - points[i, 0, WRIST, :2]) or 1.0
            out[i, 2 * size + 2:] = (points[i, 1, WRIST] - points[i, 0, WRIST]) / palm
    return out


def hands_to_two_hand_features(hands):
    """Server helper: list of (points (21, 3), handedness) -> one 181-number vector."""
    pts = np.zeros((1, 2, 21, 3))
    labels = ["", ""]
    for k, (p, label) in enumerate(hands[:2]):
        pts[0, k] = p
        labels[k] = label
    return two_hand_features(pts, [tuple(labels)])[0]
