# How to record your own phrase samples

This is the college “your contribution” path. Phrase/ISL stay **rule-based** until you record enough samples here.

## Important

| Mode | Saves to | Used by |
|------|----------|---------|
| **SPELL** | `landmarks.csv` | Letter Random Forest (`train_classifier.py`) |
| **PHRASE** / ISL | `landmarks_phrases.csv` | Your phrase dataset (kept separate on purpose) |

Do **not** mix Hello / Namaste into the letter CSV — that would confuse Spell mode.

## Goal

About **25 samples per sign** for at least:

- Hello  
- Thank You  
- Namaste (ISL)  
- Yes / No  

## Steps

1. Start the app (`./run.sh` or `PORT=5001 .venv/bin/python server.py`).
2. Log in → **Start camera**.
3. Switch to **PHRASE** (or ISL tab for Indian signs).
4. Pick the sign in the dropdown (e.g. Hello).
5. Click **Record dataset**, hold the pose ~2–3 seconds while slowly moving distance/angle.
6. Click **Stop recording** when the counter reaches ~25.
7. Repeat for the next sign.

## After recording

Phrase samples are stored for your report (`landmarks_phrases.csv`).  
Retrain the **letter** model only with:

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
.venv/bin/python train_classifier.py
```

(That script reads `landmarks.csv` only.)

## Tips for good data

- Good lighting, one hand fully in frame  
- Vary distance and slight wrist rotation  
- Hold still between tiny movements (avoid blur)  
- Same hand you will use in the demo  
