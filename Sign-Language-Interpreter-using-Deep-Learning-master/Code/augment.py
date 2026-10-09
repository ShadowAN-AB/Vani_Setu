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
