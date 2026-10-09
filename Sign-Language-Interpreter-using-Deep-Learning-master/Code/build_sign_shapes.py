"""
Pick one typical hand shape per letter for the Dictionary drawings.

For every letter, all samples are first turned into a right hand (samples that
look like a left hand are mirrored), then the medoid is chosen: the real sample
whose features are closest to the letter's average. That sample is flipped
left-right so it matches the mirrored camera view in the app, scaled into a
unit box, and saved to dataset/sign_shapes.json as 21 (x, y) points.

Usage (from Code/):  .venv/bin/python build_sign_shapes.py
"""

import json
import os

import numpy as np

import data
from features import features_v2

OUT = os.path.join(data.DATA_DIR, "sign_shapes.json")


def main():
    points, meta = data.load_samples("letter", sources=("asl_now",))
    labels = np.array([m["label"] for m in meta])
    shapes = {}
    for letter in sorted(set(labels)):
        p = points[labels == letter].copy()
        # index knuckle (5) left of pinky knuckle (17) in the raw image -> looks like a left hand
        left = p[:, 5, 0] < p[:, 17, 0]
        p[left, :, 0] = 1.0 - p[left, :, 0]
        f = features_v2(p)
        f = (f - f.mean(0)) / (f.std(0) + 1e-9)
        medoid = p[np.argmin(np.linalg.norm(f - np.median(f, 0), axis=1))]
        xy = medoid[:, :2].copy()
        xy[:, 0] = 1.0 - xy[:, 0]            # mirrored, like the camera view in the app
        xy -= xy.min(0)
        xy /= max(xy.max(), 1e-9)            # keep the aspect ratio, fit in a 1 x 1 box
        xy += (1.0 - xy.max(0)) / 2          # centre it
        shapes[letter] = [[round(float(x), 4), round(float(y), 4)] for x, y in xy]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "ASL Now Fingerspelling Landmarks (MIT), medoid sample per letter",
                   "view": "mirrored camera view, right hand", "shapes": shapes}, f, indent=1)
    print(f"Wrote {len(shapes)} letter shapes -> {os.path.relpath(OUT, data.APP_DIR)}")


if __name__ == "__main__":
    main()
