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
