"""
Extract MediaPipe hand landmarks from a Kaggle ASL alphabet image dataset
and append them as training rows to dataset/kaggle_asl_landmarks.csv.

Why this exists
---------------
Our letter SVM trains on 21-point hand landmarks, not on raw images. Kaggle
ships whole-image ASL datasets; this script pushes each image through the same
MediaPipe hand tracker the browser uses, so the extracted landmarks live in
the same feature space as our existing asl_now and recorded samples.

Usage (from the repo root)
--------------------------
  python tools/kaggle_to_landmarks.py \
      --input  /tmp/.../asl_alphabet_train \
      --per-class 400 \
      --signer kaggle_grassknoted

Writes
------
  Sign-Language-Interpreter-using-Deep-Learning-master/Code/dataset/kaggle_asl_landmarks.csv
    one row per image where MediaPipe produced a hand, matching the schema
    of dataset/asl_now_landmarks.csv (label, kind=letter, signer, source, 63 coords)
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime, timezone

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE_DIR = os.path.join(REPO, "Sign-Language-Interpreter-using-Deep-Learning-master", "Code")
DATASET_DIR = os.path.join(CODE_DIR, "dataset")
DEFAULT_OUT = os.path.join(DATASET_DIR, "kaggle_asl_landmarks.csv")
HAND_TASK = os.path.join(CODE_DIR, "hand_landmarker.task")

COORD_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
COORD2_COLUMNS = ["b" + c for c in COORD_COLUMNS]
META_COLUMNS = ["label", "kind", "signer", "hand", "hand2", "split", "source", "recorded_at"]
COLUMNS = META_COLUMNS + COORD_COLUMNS + COORD2_COLUMNS

# The 26 letters we actually classify. The Kaggle grassknoted set also includes
# 'del', 'nothing' and 'space' folders which we ignore.
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def iter_images(root, per_class, seed):
    """Yield (label, image_path) at most `per_class` times per class."""
    rng = random.Random(seed)
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and d.upper() in LETTERS)
    for cls in classes:
        folder = os.path.join(root, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        for f in files[:per_class]:
            yield cls.upper(), os.path.join(folder, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to asl_alphabet_train/ (contains A/, B/, ...)")
    p.add_argument("--out", default=DEFAULT_OUT, help="output CSV (default: dataset/kaggle_asl_landmarks.csv)")
    p.add_argument("--per-class", type=int, default=400, help="max images to process per letter (default 400)")
    p.add_argument("--signer", default="kaggle_grassknoted", help="signer tag written into the CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", default="kaggle_asl_alphabet",
                   help="value for the 'source' column (lets train_classifier distinguish origins)")
    args = p.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input not a directory: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_TASK),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    ok = fail = 0
    per_class_ok = {}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        writer.writeheader()
        for label, path in iter_images(args.input, args.per_class, args.seed):
            img = cv2.imread(path)
            if img is None:
                fail += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)
            if not res.hand_landmarks:
                fail += 1
                continue
            lm = res.hand_landmarks[0]
            handed = ""
            if res.handedness:
                handed = res.handedness[0][0].category_name

            row = {k: "" for k in COLUMNS}
            row.update({
                "label": label,
                "kind": "letter",
                "signer": args.signer,
                "hand": handed,
                "source": args.source,
                "recorded_at": stamp,
            })
            for i in range(21):
                row[f"x{i}"] = f"{lm[i].x:.6f}"
                row[f"y{i}"] = f"{lm[i].y:.6f}"
                row[f"z{i}"] = f"{lm[i].z:.6f}"
            writer.writerow(row)
            ok += 1
            per_class_ok[label] = per_class_ok.get(label, 0) + 1
            if ok % 500 == 0:
                print(f"  {ok} landmarks extracted...", flush=True)

    detector.close()
    print(f"\nDone. {ok} landmarks written, {fail} images without a detected hand.")
    print(f"  Output: {args.out}")
    print(f"  Per class:")
    for cls in sorted(per_class_ok):
        print(f"    {cls}: {per_class_ok[cls]}")


if __name__ == "__main__":
    main()
