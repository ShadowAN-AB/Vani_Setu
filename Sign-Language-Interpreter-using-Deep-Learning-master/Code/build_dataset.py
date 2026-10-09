"""
Convert the ASL Now JSON files into one raw-landmark CSV.

The public dataset (https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling,
MIT licence) stores one JSON file per sample in a folder per letter. This script
writes dataset/asl_now_landmarks.csv with the raw 21 (x, y, z) points, so the
trainer never depends on pre-computed features.

The CSV is small (about 2 MB) and is committed, so teammates do not need the
JSON folder. Re-run this only if the JSON files change.

Usage (from Code/):  .venv/bin/python build_dataset.py
"""

import csv
import glob
import json
import os
import sys

from data import ASL_NOW_CSV, ASL_NOW_DIR, COLUMNS, COORD_COLUMNS, landmarks_to_array


def main():
    if not os.path.isdir(ASL_NOW_DIR):
        print(f"JSON folder not found: {ASL_NOW_DIR}")
        print("Download it from the Hugging Face page above, or use the committed asl_now_landmarks.csv.")
        sys.exit(1)

    rows, skipped = [], 0
    for letter in sorted(os.listdir(ASL_NOW_DIR)):
        folder = os.path.join(ASL_NOW_DIR, letter)
        if not os.path.isdir(folder):
            continue
        for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    arr = landmarks_to_array(json.load(f))
            except (ValueError, KeyError, json.JSONDecodeError):
                skipped += 1
                continue
            row = {"label": letter, "kind": "letter", "signer": "asl_now", "hand": "",
                   "split": "", "source": "asl_now", "recorded_at": ""}
            row.update({c: f"{v:.6f}" for c, v in zip(COORD_COLUMNS, arr.reshape(-1))})
            rows.append(row)

    with open(ASL_NOW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    letters = sorted({r["label"] for r in rows})
    print(f"Wrote {len(rows)} samples, {len(letters)} letters -> {os.path.relpath(ASL_NOW_CSV)}")
    if skipped:
        print(f"Skipped {skipped} unreadable files")


if __name__ == "__main__":
    main()
