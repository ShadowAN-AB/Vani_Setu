# Dataset source

**Name:** ASL Now Fingerspelling Landmarks  
**Repo:** https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling  
**License:** MIT  
**Format:** MediaPipe Hand Landmarker 21 points (x, y, z) per sample  
**Converted to:** `landmarks.csv` with wrist-relative 63-D features (same as app `/api/collect`)

## Local letter set (Spell / ML)

| Item | Value |
|------|--------|
| Samples | **2122** |
| Letters | **A–Z** (26 classes) |
| Held-out accuracy | **95.1%** (RBF SVM) |
| Model file | `models/sign_rf.joblib` |

The SVM was picked by measurement, not preference — see `reports/model_comparison.md` (SVM 95.1% vs Random Forest 88.1%, KNN 85.7%, Decision Tree 74.6%).

Static landmarks struggle with motion letters (**J**, **Z**). Live app uses ML in Spell mode with a **0.50** confidence floor, rule fallback, and multi-frame voting.

## Phrase / ISL samples (your contribution)

Phrase recordings go to **`landmarks_phrases.csv`** (separate from letters) so they do not confuse the A–Z model.

See **`RECORDING_GUIDE.md`** in this folder.

## How to rebuild the letter model

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
.venv/bin/python train_classifier.py
```

## Citation

Dataset by AxelS27 / ASLNow on Hugging Face. Landmarks extracted with MediaPipe Hand Landmarker.
