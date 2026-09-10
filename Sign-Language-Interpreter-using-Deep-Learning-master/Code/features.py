"""Convert MediaPipe 21-point landmarks into a 63-D wrist-relative vector."""


def landmarks_to_features(landmarks):
    if not landmarks or len(landmarks) < 21:
        raise ValueError("Need 21 hand landmarks")
    wrist = landmarks[0]
    wx, wy, wz = float(wrist["x"]), float(wrist["y"]), float(wrist.get("z") or 0)
    feats = []
    for p in landmarks[:21]:
        feats.append(float(p["x"]) - wx)
        feats.append(float(p["y"]) - wy)
        feats.append(float(p.get("z") or 0) - wz)
    return feats
