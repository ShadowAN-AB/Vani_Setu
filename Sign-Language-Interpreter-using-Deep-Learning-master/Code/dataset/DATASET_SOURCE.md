# Dataset source

**Name:** ASL Now Fingerspelling Landmarks
**Repo:** https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling
**License:** MIT
**Collected by:** multiple participants signing ASL letters to a webcam; landmarks from the MediaPipe Web Hand Landmarker (the same detector the app uses)

## What is stored here

| File | Contents |
|------|----------|
| `raw_asl_now/` (not in git) | The original dataset: one JSON file per sample, 21 points (x, y, z) |
| `asl_now_landmarks.csv` (in git) | The same 2,122 samples as raw points, built by `build_dataset.py` |
| `recordings/samples.csv` (in git) | Samples the team records in the app, with signer name, hand and split |
| `landmarks.csv` (old, not in git) | Pre-computed 63-D features used before October 2026; kept only so `experiments_m2.py` reproduces the Milestone 2 numbers |

The Hugging Face page lists 44,562 rows because it counts each landmark point:
2,122 samples × 21 points = 44,562. The full dataset is the 2,122 samples above.

## Letter set

| Item | Value |
|------|-------|
| Samples | 2,122 (smallest class C = 53, largest Z = 155) |
| Letters | A–Z (26 classes) |
| Split | 70% train / 15% validation / 15% test, stratified, seed 42 |
| Training data after augmentation | 8,904 (mirror + 2 rotated/jittered copies) |
| Features | version 2, 88 numbers (see `features.py`) |
| Test accuracy | see `reports/metrics.json` |

J and Z involve motion, so a single frame describes them only partly.

## Citation

Dataset by AxelS27 / ASLNow on Hugging Face (MIT). Landmarks extracted with Google MediaPipe Hand Landmarker.
