# Dataset source

**Name:** ASL Now Fingerspelling Landmarks  
**Repo:** https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling  
**License:** MIT  
**Format:** MediaPipe Hand Landmarker 21 points (x, y, z) per sample  
**Converted to:** `landmarks.csv` with wrist-relative 63-D features (same as app `/api/collect`)

## What we have locally

| Item | Value |
|------|--------|
| Samples | 1779 |
| Letters | A–W (23 classes) |
| Held out accuracy | ~89% (RandomForest) |
| Model file | `models/sign_rf.joblib` |

X / Y / Z were incomplete because Hugging Face rate-limited the download mid-way. You can re-download later and re-run conversion + `train_classifier.py`.

## Important for your college report

- This is **ASL fingerspelling letters**, not ISL phrases.
- **Spell mode** can use the ML model once the Flask server loads `sign_rf.joblib`.
- **Phrase / ISL modes** still use rule-based poses unless you record your own samples in the Dataset panel.
- Static landmarks struggle with motion letters (J, Z).

## How it was prepared

1. Download Hugging Face dataset into `dataset/raw_asl_now/`
2. Convert JSON landmarks → wrist-relative 63 floats → `landmarks.csv`
3. Train: `python train_classifier.py`

## Citation

Dataset by AxelS27 / ASLNow on Hugging Face. Landmarks extracted with MediaPipe Hand Landmarker.
