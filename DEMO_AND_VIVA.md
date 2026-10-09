# Vani-Setu — How to run & viva notes

## Quick start (Mac / Linux)

```bash
cd Vani-Setu
./run.sh
```

Open **http://127.0.0.1:5001**  
Create an account or sign in. Accounts are stored in `Code/users.db` (SQLite).

Optional:

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_web.txt
# first time: download MediaPipe hand model as hand_landmarker.task into Code/
PORT=5001 python server.py
```

Retrain the letter model after recording new samples (`dataset/recordings/samples.csv`):

```bash
.venv/bin/python train_classifier.py       # train + reports/metrics.json
.venv/bin/python compare_models.py         # six classifiers, same data
.venv/bin/python evaluate.py               # reports/evaluation.md with all figures
```

Or just click **Retrain model** in the app — it trains and hot-reloads the
model without restarting the server.

## What works in the demo

1. Create account / Sign in (email + password)  
2. Start camera  
3. **Spell** mode → letters A–Z via SVM, 96.9% test accuracy (badge shows **ML MODEL**)  
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

**Why an SVM?** Because it measured best. `compare_models.py` trains six
classifiers on the same 70/15/15 split, augmented training data and 88 features,
and ranks them on the validation set; results are in `reports/model_comparison.md`.

**What changed from the first version?** `reports/evaluation.md` compares them on the
same test set. The original model (63 features, no augmentation) drops to 64% when
the other hand is used and 72% when the hand is farther away; the new model stays
at about 97% in every case.

**How is the confidence threshold chosen?** `train_classifier.py` picks the highest
threshold that still keeps 98% of correct validation predictions; the server sends
it to the app. Frames below it vote "unsure", and the 14-frame vote filters the rest.

**Why is Practice mode useful?** It turns a one-way translator into a learning
tool for hearing users, and it exercises the classifier repeatedly on camera.

## Honest limitations (say these in viva)

- One hand, mostly **static** poses (motion letters J/Z are weak)  
- Spell ML is **ASL fingerspelling**, not a full sign-language corpus  
- ISL in the app is a **classroom approximation**, not a certified lexicon  
- Phrase mode is rule-based until phrase samples are recorded and a phrase model is trained  
- Free Google Translate can fail; Hindi has an offline fallback  
- Use port **5001** on macOS (5000 is often taken by AirPlay)

## Project layout (what you changed)

| Path | Role |
|------|------|
| `Code/index.html` | UI, MediaPipe, Spell ML client, suggestions, practice mode |
| `Code/server.py` | Password login, translate, collect, classify, train, health |
| `Code/models_user.py` | User table (SQLite / Postgres) |
| `Code/data.py`, `build_dataset.py` | Raw landmark storage and the ASL Now converter |
| `Code/features.py`, `augment.py` | 88 features; mirror / rotate / jitter augmentation |
| `Code/train_classifier.py` | SVM trainer (writes model + reports) |
| `Code/compare_models.py` | Benchmarks 6 classifiers for the report |
| `Code/evaluate.py` | Before/after, CV, learning and validation curves, PCA/t-SNE, speed |
| `Code/dataset/` | `asl_now_landmarks.csv`, `recordings/samples.csv`, guides |
| `Code/models/sign_svm.joblib` | Trained letter model (in git) |
| `Code/reports/` | Metrics, evaluation report, confusion matrix, model comparison |

## Dataset citation

ASL Now Fingerspelling landmarks — [AxelS27/asl-now-fingerspelling](https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling) (MIT).  
See `Code/dataset/DATASET_SOURCE.md` and `RECORDING_GUIDE.md`.
