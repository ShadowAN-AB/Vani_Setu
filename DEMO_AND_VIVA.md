# Vani-Setu — How to run & viva notes

## Quick start (Mac / Linux)

```bash
cd Vani-Setu
./run.sh
```

Open **http://127.0.0.1:5001**  
OTP prints in the terminal if email is not configured in `Code/.env`.

Optional:

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_web.txt
# first time: download MediaPipe hand model as hand_landmarker.task into Code/
PORT=5001 python server.py
```

Retrain letter model after changing `dataset/landmarks.csv`:

```bash
.venv/bin/python train_classifier.py
```

Then restart the server so it loads `models/sign_rf.joblib`.

## What works in the demo

1. Login with OTP  
2. Start camera  
3. **Spell** mode → letters A–Z via ML (badge shows **ML MODEL**)  
4. **Phrase** / **ISL** → rule-based poses (record your own samples for the report)  
5. Translate + Speak  

## Pre-demo health check

```bash
curl -s http://127.0.0.1:5001/api/health
```

`ready_for_demo: true` means the classifier and the MediaPipe hand model are both present.

## Demo checklist (screen recording)

- [ ] Run the health check above
- [ ] Hard-refresh the page after restarting the server  
- [ ] Show **ML MODEL** badge  
- [ ] Spell: hold **L**, **V**, **W**, **B** for ~1 second each  
- [ ] Switch to Phrase: Hello / Peace / OK  
- [ ] Switch language (e.g. Hindi) and Speak  
- [ ] Open Sign Guide briefly  
- [ ] Logout (camera LED should turn off)

## Honest limitations (say these in viva)

- One hand, mostly **static** poses (motion letters J/Z are weak)  
- Spell ML is **ASL fingerspelling**, not a full sign-language corpus  
- ISL in the app is a **classroom approximation**, not a certified lexicon  
- Phrase mode is rule-based until you record `landmarks_phrases.csv`  
- Free Google Translate can fail; Hindi has an offline fallback  
- Use port **5001** on macOS (5000 is often taken by AirPlay)

## Project layout (what you changed)

| Path | Role |
|------|------|
| `Code/index.html` | UI + MediaPipe + Spell ML client |
| `Code/server.py` | OTP, translate, collect, classify |
| `Code/train_classifier.py` | Random Forest trainer |
| `Code/dataset/` | Letter CSV + phrase CSV + guides |
| `Code/models/sign_rf.joblib` | Trained letter model (local, not in git) |
| `reports/` | Accuracy + confusion matrix for report |

## Dataset citation

ASL Now Fingerspelling landmarks — [AxelS27/asl-now-fingerspelling](https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling) (MIT).  
See `Code/dataset/DATASET_SOURCE.md` and `RECORDING_GUIDE.md`.
