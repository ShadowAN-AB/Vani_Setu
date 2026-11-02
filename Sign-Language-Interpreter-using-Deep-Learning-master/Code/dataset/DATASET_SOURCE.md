# Dataset sources

The letter model trains on hand landmarks from **two independent sources**,
merged into one training set.

## 1. ASL Now Fingerspelling Landmarks

| | |
|---|---|
| Repo | https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling |
| License | MIT |
| Collected by | multiple participants signing ASL letters to a webcam; landmarks from MediaPipe Hand Landmarker |
| Samples | 2,122 (smallest class C = 53, largest Z = 155) |
| File | `asl_now_landmarks.csv` (in git), built by `build_dataset.py` from the per-sample JSON in `raw_asl_now/` (not in git) |

The Hugging Face page lists 44,562 rows because it counts each landmark point:
2,122 samples × 21 points = 44,562.

## 2. Kaggle ASL Alphabet (preprocessed through MediaPipe)

| | |
|---|---|
| Repo | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| License | GPL-2.0 (image dataset) |
| Collected by | 29-class image dataset (A–Z + del / nothing / space); only A–Z are used here |
| Samples retained | 8,296 landmarks (from 10,400 sampled images; MediaPipe failed to detect a hand on ~20%) |
| File | `kaggle_asl_landmarks.csv` (in git) |
| Preprocessor | `tools/kaggle_to_landmarks.py` — runs the same MediaPipe `hand_landmarker.task` the browser uses, writes rows in the same schema as `asl_now_landmarks.csv` |

Re-running the preprocessor (if the image dataset is on disk):

```bash
python tools/kaggle_to_landmarks.py \
  --input /path/to/asl_alphabet_train \
  --per-class 400
```

## Combined letter set

| Item | Value |
|------|-------|
| Total samples | **10,418** (ASL Now 2,122 + Kaggle 8,296) |
| Letters | A–Z (26 classes) |
| Split | 70 % train / 15 % validation / 15 % test, stratified, seed 42 |
| Training data after augmentation | 43,752 (mirror + rotated / jittered copies) |
| Features | version 2, 88 numbers per hand (see `features.py`) |
| Test accuracy | see `reports/metrics.json` |

J and Z involve motion, so a single frame describes them only partly. M and N
share a near-identical closed-fist shape in still photos and account for most
of the remaining per-class error on the Kaggle portion of the test set.

## Recorded samples (optional)

`recordings/samples.csv` holds any extra samples the team records in the app
locally (created on first recording; gitignored by default). The server tags
each sample with the signer's name and whether it was marked **Test set**.

`landmarks.csv` (old, not in git) is a pre-computed 63-D feature file used
before October 2026, kept only so `experiments_m2.py` reproduces the
Milestone 2 numbers.

## Citations

- AxelS27 / ASLNow on Hugging Face (MIT), landmarks via Google MediaPipe
- `grassknoted/asl-alphabet` on Kaggle (GPL-2.0), landmarks re-extracted locally
  with the same Google MediaPipe `hand_landmarker.task`
