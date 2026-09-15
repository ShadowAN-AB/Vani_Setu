# Vani-Setu

<p align="center">
  <strong>voice-bridge</strong> — real-time ASL / ISL translation from a laptop camera
</p>

<p align="center">
  <a href="https://vani-setu.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-vani--setu.onrender.com-0b5f66?style=for-the-badge" alt="Live demo"></a>
  <a href="https://github.com/ShadowAN-AB/Vani_Setu"><img src="https://img.shields.io/badge/GitHub-ShadowAN--AB%2FVani__Setu-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Landmarker-009688?style=flat-square&logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/scikit--learn-SVM_RBF-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/SQLite-accounts-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
  <img src="https://img.shields.io/badge/Render-HTTPS-46E3B7?style=flat-square&logo=render&logoColor=black" alt="Render">
</p>

**Live app:** [https://vani-setu.onrender.com](https://vani-setu.onrender.com)  
Open the link → **Continue as guest** → **Start camera** (allow the webcam). The Free Render instance sleeps after idle time; the first load can take about a minute.

A laptop camera tracks one hand in the browser. The app turns poses into letters or short phrases, builds a sentence, can speak it, and can translate it into 25+ languages.

Day to day, run it on **localhost** at `http://127.0.0.1:5001` (`./run.sh`). The public URL is the same product over HTTPS.

---

## Tech stack

| Layer | Tools |
|:------|:------|
| **Interface** | HTML5 · CSS3 · vanilla JavaScript · Google Fonts (Fraunces / Sora) |
| **Vision** | [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (21-point Hand Landmarker in the browser) · `getUserMedia` camera |
| **Backend** | Python 3 · Flask · Flask-CORS · Gunicorn |
| **Machine learning** | scikit-learn **RBF SVM** (~95.1% held-out A–Z) · joblib · pandas · matplotlib (reports) |
| **Accounts** | Flask-SQLAlchemy · Werkzeug password hashes · SQLite locally · optional Postgres (`DATABASE_URL`) |
| **Speech & language** | Web Speech API (speak) · deep-translator / Google Translate · offline Hindi fallback |
| **Data** | ASL Now fingerspelling landmarks (Hugging Face, MIT) · 63-D wrist-relative features |
| **Run & ship** | `run.sh` on localhost · Render (HTTPS) · GitHub · python-dotenv |

```text
Browser (HTML / CSS / JS + MediaPipe)
        │  camera landmarks
        ▼
Flask API  ──►  SVM (Spell)  |  pose rules (Phrase / ISL)
        │
        ├── SQLite / Postgres   (login)
        └── Translate + Speak
```

---

## What it does

| Mode | What happens |
|------|----------------|
| **Spell** | Fingerspelling A–Z. An RBF **SVM** classifies 63-D MediaPipe landmarks (~**95.1%** held-out accuracy on 2122 samples). |
| **Phrase** | Whole signs such as Hello, Yes, OK. Rule-based finger geometry. |
| **ISL** | Simplified single-hand classroom signs (Namaste, Thank You, Water, Help, …). Teaching set, not a certified ISL lexicon. |
| **Practice** | The app prompts a letter and scores your attempts. |
| **Translate / Speak** | English sentence → 25+ languages, plus browser speech. Hindi has an offline fallback if Google Translate is down. |

Also included:

- Email + password login (hashed passwords in a local **SQLite** file)
- Live confidence bar in Spell mode
- Word-suggestion chips while spelling
- Record dataset + **Retrain model** from the browser
- Sign Guide overlay

---

## How it works

1. You create an account or sign in.
2. The browser camera + **MediaPipe Hand Landmarker** finds 21 landmarks on one hand. Tracking stays on the device.
3. Spell mode sends a 63-D feature vector to Flask (`/api/classify`). The SVM returns a letter and a confidence score. A 0.50 floor plus multi-frame voting reduces flicker.
4. Phrase / ISL modes match poses with rules unless you record your own phrase samples.
5. You assemble a sentence, speak it, or translate it.

Login does **not** need MongoDB. Accounts are stored in `Sign-Language-Interpreter-using-Deep-Learning-master/Code/users.db`. On a host you can set `DATABASE_URL` for Postgres instead.

---

## Requirements

- macOS or Linux (Windows: use the `.bat` launchers under `Sign-Language-Interpreter-using-Deep-Learning-master/`)
- **Python 3.10+** (`python3 --version`)
- A laptop **camera**
- **Internet** the first time (pip packages + MediaPipe JS from `cdn.jsdelivr.net`)
- Port **5001** free — on macOS, port 5000 is often taken by AirPlay

---

## Run on localhost (recommended)

From this repository root (`Vani-Setu/`):

```bash
chmod +x run.sh
./run.sh
```

`run.sh` creates a virtualenv if needed, installs `Code/requirements_web.txt`, and starts Flask.

Open **http://127.0.0.1:5001**

1. **Create account** (email + password, at least 6 characters).
2. Later visits: **Sign in**.
3. Click **Start camera** and allow the browser.
4. Use **SPELL** for letters, **PHRASE** or ISL for words.
5. **Logout** when finished (this also stops the camera LED).

Stop the server with `Ctrl+C` in the terminal.

### Manual start (if you do not use `run.sh`)

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_web.txt
PORT=5001 python server.py
```

### Windows

Double-click `Sign-Language-Interpreter-using-Deep-Learning-master/RUN_VANI_SETU.bat`  
(or `QUICK_START.bat`). The default port on those scripts may be 5000.

---

## First-time files

| File | In git? | What happens |
|------|---------|----------------|
| `Code/models/sign_rf.joblib` | Yes | Spell-mode SVM. Present after `git clone`. |
| `Code/hand_landmarker.task` | No (~7.5 MB) | MediaPipe hand model. The server **downloads it on first run** if missing. |

`users.db` stays local and gitignored. A hosted demo uses guest login so it does not depend on that file.

Keep this project folder for the viva. GitHub now has the letter model; you do not need a USB copy of `sign_rf.joblib`.

---

## Public URL

**Live:** [https://vani-setu.onrender.com](https://vani-setu.onrender.com)

Localhost does not change. To redeploy after a code push:

1. Push `main` to GitHub (Render auto-builds).
2. Or [render.com](https://render.com) → **New** → **Web Service**:
   - Build: `pip install -r requirements.txt`
   - Start: `bash start.sh`
   - `AUTH_MODE=guest`
3. Open the `https://` URL → **Continue as guest** → **Start camera**.

Full notes: [DEPLOY.md](DEPLOY.md). Free Render apps sleep after idle time; the first hit after sleep can take about a minute.

---

## Health check

With the server running:

```bash
curl -s http://127.0.0.1:5001/api/health
```

`ready_for_demo: true` means the classifier and `hand_landmarker.task` are both present.

---

## Coming back after a break

```bash
cd /path/to/Vani-Setu
./run.sh
```

Then open **http://127.0.0.1:5001**. Accounts in `users.db` stay on disk. Nothing expires. If Python on the Mac was upgraded, `./run.sh` recreates `.venv`.

---

## Auth options

Default is **password**. Copy `Code/.env.example` to `Code/.env` only if you need to change this.

| `AUTH_MODE` | Behaviour |
|-------------|-----------|
| `password` | Email + password (SQLite). **Default.** |
| `guest` | Password login plus **Continue as guest**. |
| `otp` | Email / terminal 6-digit code (needs SMTP for real email). |
| `none` | Skip the login screen. |

Hosted deploy notes (Render, HTTPS, guest login): see [DEPLOY.md](DEPLOY.md).

---

## Retrain / record

- **Record dataset** in the UI writes landmarks to `Code/dataset/landmarks.csv` (letters) or `landmarks_phrases.csv` (phrases).
- **Retrain model** in the UI trains the SVM and reloads it without restarting Flask.
- From a terminal:

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
.venv/bin/python train_classifier.py    # train + reports
.venv/bin/python compare_models.py      # SVM vs RF vs KNN vs tree (for the report)
```

Reports land in `Code/reports/` (confusion matrix, accuracy, model comparison).

---

## Project layout

```
Vani-Setu/
├── README.md                          ← this file
├── run.sh                             ← Mac / Linux localhost launcher (port 5001)
├── start.sh / Procfile / render.yaml  ← public host (Render / Railway)
├── DEPLOY.md                          ← localhost vs Render
├── DEMO_AND_VIVA.md                   ← demo checklist and viva talking points
└── Sign-Language-Interpreter-using-Deep-Learning-master/
    ├── README.md                      ← original HackUNT-19 project
    ├── DOCUMENTATION.md               ← longer architecture notes
    └── Code/
        ├── index.html                 ← UI + MediaPipe
        ├── server.py                  ← Flask APIs
        ├── models_user.py             ← User table (SQLite / Postgres)
        ├── features.py                ← 63-D landmark features
        ├── signs.py                   ← phrase / ISL rules + Hindi fallback
        ├── train_classifier.py        ← SVM trainer
        ├── compare_models.py          ← classifier benchmark
        ├── requirements_web.txt
        ├── .env.example
        ├── dataset/                   ← landmarks + recording guides
        └── models/sign_rf.joblib      ← trained letter SVM (in git)
```

---

## Dataset

Spell-mode letters come from **ASL Now Fingerspelling** landmarks  
([AxelS27/asl-now-fingerspelling](https://huggingface.co/datasets/AxelS27/asl-now-fingerspelling), MIT), converted to the same 63-D features the live app uses.

See `Code/dataset/DATASET_SOURCE.md` and `RECORDING_GUIDE.md`.

---

## Honest limits

- One hand, mostly **static** poses. Motion letters **J** and **Z** are weak.
- Spell ML is **ASL fingerspelling**, not a full sign-language corpus.
- ISL in the app is a **classroom approximation**.
- Phrase mode is rule-based until you record phrase samples.
- Browser camera works on `http://127.0.0.1` but **needs HTTPS** on a public URL.
- This is a demo, not a production account system (session is `sessionStorage` after login).

---

## More docs

- [DEMO_AND_VIVA.md](DEMO_AND_VIVA.md) — what to show an examiner
- [DEPLOY.md](DEPLOY.md) — guest mode, skip login, Gmail OTP, Postgres
- [Sign-Language-Interpreter-using-Deep-Learning-master/DOCUMENTATION.md](Sign-Language-Interpreter-using-Deep-Learning-master/DOCUMENTATION.md) — older architecture write-up
- Original hackathon app (CNN / OpenCV, not the web demo): [Sign-Language-Interpreter-using-Deep-Learning-master/README.md](Sign-Language-Interpreter-using-Deep-Learning-master/README.md)
