"""
Data augmentation for hand landmarks (training data only).

Each original sample produces realistic variations that should not change the
letter, so the model learns to ignore them:

  mirror   flip x, which turns a right hand into a left hand
  rotate   tilt the hand around the wrist by up to +/- max_rotation degrees
  jitter   move every point by a small random amount (a fraction of palm length),
           like the small errors MediaPipe makes from frame to frame

Augmentation is applied only after the train / validation / test split and only
to the training part. Augmenting before splitting would put near-copies of the
same sample on both sides and give an over-optimistic score (data leakage).
"""

import numpy as np


def mirror(points):
    """Flip left and right. Points are in image coordinates (x in [0, 1])."""
    p = np.array(points, dtype=np.float64, copy=True)
    p[:, :, 0] = 1.0 - p[:, :, 0]
    return p


def rotate(points, degrees):
    """Rotate each hand in the image plane around its wrist. degrees: one value per sample."""
    p = np.asarray(points, dtype=np.float64)
    wrist = p[:, :1]
    q = p - wrist
    th = np.deg2rad(np.asarray(degrees, dtype=np.float64))[:, None]
    c, s = np.cos(th), np.sin(th)
    x = q[:, :, 0] * c - q[:, :, 1] * s
    y = q[:, :, 0] * s + q[:, :, 1] * c
    return np.stack([x, y, q[:, :, 2]], 2) + wrist


def jitter(points, amount, rng):
    """Add Gaussian noise scaled by each hand's palm length."""
    p = np.asarray(points, dtype=np.float64)
    palm = np.linalg.norm(p[:, 9, :2] - p[:, 0, :2], axis=1)
    return p + rng.normal(0.0, 1.0, p.shape) * amount * palm[:, None, None]


def augment(points, labels, copies=2, max_rotation=15.0, jitter_amount=0.01, use_mirror=True, seed=42):
    """
    Return (points, labels) with the originals first, then the new samples.

    With use_mirror=True and copies=2 every sample becomes 6:
    original, mirrored, and 2 rotated+jittered versions of each.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    base_p, base_y = [points], [labels]
    if use_mirror:
        base_p.append(mirror(points))
        base_y.append(labels)
    base_p = np.concatenate(base_p)
    base_y = np.concatenate(base_y)

    out_p, out_y = [base_p], [base_y]
    for _ in range(copies):
        angles = rng.uniform(-max_rotation, max_rotation, len(base_p))
        out_p.append(jitter(rotate(base_p, angles), jitter_amount, rng))
        out_y.append(base_y)
    return np.concatenate(out_p), np.concatenate(out_y)


def _swap(label):
    return {"Left": "Right", "Right": "Left"}.get(label, label)


def augment_two_hands(points, hands, labels, copies=2, max_rotation=15.0, jitter_amount=0.01, use_mirror=True, seed=42):
    """
    Two-hand version for the phrase / ISL models.
      points (N, 2, 21, 3) with zeros for a missing hand, hands = list of handedness pairs.
    Mirroring flips x and swaps Left / Right, so a left-handed signer is covered.
    Rotation turns both hands together around the first wrist, keeping their relative position.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    present = (np.abs(points).sum(axis=(2, 3)) > 0)[:, :, None, None]

    base_p, base_h, base_y = [points], [list(hands)], [labels]
    if use_mirror:
        m = points.copy()
        m[:, :, :, 0] = 1.0 - m[:, :, :, 0]
        base_p.append(np.where(present, m, 0.0))
        base_h.append([(_swap(a), _swap(b)) for a, b in hands])
        base_y.append(labels)
    base_p = np.concatenate(base_p)
    base_h = sum(base_h, [])
    base_y = np.concatenate(base_y)
    base_present = (np.abs(base_p).sum(axis=(2, 3)) > 0)[:, :, None, None]

    out_p, out_h, out_y = [base_p], [base_h], [base_y]
    for _ in range(copies):
        th = np.deg2rad(rng.uniform(-max_rotation, max_rotation, len(base_p)))[:, None, None]
        c, s = np.cos(th), np.sin(th)
        pivot = base_p[:, :1, :1]
        q = base_p - pivot
        x = q[..., 0] * c - q[..., 1] * s
        y = q[..., 0] * s + q[..., 1] * c
        r = np.stack([x, y, q[..., 2]], -1) + pivot
        palm = np.linalg.norm(base_p[:, 0, 9, :2] - base_p[:, 0, 0, :2], axis=1)[:, None, None, None]
        r = r + rng.normal(0.0, 1.0, r.shape) * jitter_amount * palm
        out_p.append(np.where(base_present, r, 0.0))
        out_h.append(base_h)
        out_y.append(base_y)
    return np.concatenate(out_p), sum(out_h, []), np.concatenate(out_y)
