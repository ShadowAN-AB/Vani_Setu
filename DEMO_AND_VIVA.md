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
.venv/bin/python train_classifier.py       # train + reports
.venv/bin/python compare_models.py         # benchmark table for the report
```

Or just click **Retrain model** in the app — it trains and hot-reloads the
model without restarting the server.

## What works in the demo

1. Login with OTP  
2. Start camera  
3. **Spell** mode → letters A–Z via SVM, 95.1% held-out accuracy (badge shows **ML MODEL**)  
4. **Live confidence bar** showing the model's probability per frame  
5. **Word suggestions** — spell a prefix, tap a chip to complete the word  
6. **Practice mode** — the app prompts a letter and scores your attempts  
7. **Phrase** / **ISL** → rule-based poses (record your own samples for the report)  
8. Translate into 25+ languages + Speak  
9. **Retrain model** button — trains and hot-reloads without a terminal  

## Pre-demo health check

```bash
curl -s http://127.0.0.1:5001/api/health
```

`ready_for_demo: true` means the classifier and the MediaPipe hand model are both present.

## Demo checklist (screen recording)

- [ ] Run the health check above
- [ ] Hard-refresh the page after restarting the server  
- [ ] Show **ML MODEL** badge  
- [ ] Spell: hold **L**, **V**, **W**, **B** for ~1 second each — point out the **confidence bar** moving  
- [ ] Spell 3 letters of a word and tap a **suggestion chip** to complete it  
- [ ] Click **Practice**, sign 3–4 prompted letters, show the score and streak  
- [ ] Switch to Phrase: Hello / Peace / OK  
- [ ] Switch language (e.g. Hindi) and Speak  
- [ ] Click **Retrain model** to show training runs from the browser  
- [ ] Open Sign Guide briefly  
- [ ] Logout (camera LED should turn off)

## Talking points for the examiner

**Why an SVM?** Because it measured best. `compare_models.py` benchmarks five
classifiers on identical features; results are in `reports/model_comparison.md`:

| Model | Test accuracy | 5-fold CV |
|-------|---------------|-----------|
| **SVM (RBF)** | **95.1%** | 94.1% |
| Logistic Regression | 94.0% | 92.5% |
| Random Forest | 88.1% | 87.9% |
| K-Nearest Neighbours | 85.7% | 84.9% |
| Decision Tree | 74.6% | 74.8% |

**Why a confidence threshold of 0.50?** Measured on held-out samples, it keeps
about 96% of correct predictions while rejecting roughly 69% of the errors.
Multi-frame voting filters whatever gets past it.

**Why is Practice mode useful?** It turns a one-way translator into a learning
tool for hearing users, and it exercises the classifier repeatedly on camera.

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
| `Code/index.html` | UI, MediaPipe, Spell ML client, suggestions, practice mode |
| `Code/server.py` | OTP, translate, collect, classify, train, health |
| `Code/train_classifier.py` | SVM trainer (writes model + reports) |
| `Code/compare_models.py` | Benchmarks 5 classifiers for the report |
| `Code/dataset/` | Letter CSV + phrase CSV + guides |
| `Code/models/sign_rf.joblib` | Trained letter model (local, not in git) |
| `Code/reports/` | Accuracy, confusion matrix, model comparison |

## Dataset citation

ASL Now Fingerspelling landmarks — [AxelS27/asl-now-fingerspelling](https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling) (MIT).  
See `Code/dataset/DATASET_SOURCE.md` and `RECORDING_GUIDE.md`.
