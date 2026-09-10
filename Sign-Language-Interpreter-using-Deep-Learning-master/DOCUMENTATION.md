# Vani-Setu — Project Documentation

**Vani-Setu** (*voice-bridge*) is a real-time American Sign Language (ASL) translator. A camera captures hand signs, the app classifies them, builds words and sentences, speaks the result, and can translate the English output into 25+ languages.

This document describes the codebase as it exists today: what it is, how it is built, what actually works, and what is incomplete.

---

## 1. Executive summary

| Area | Status | Estimate |
| --- | --- | --- |
| **Web product (Vani-Setu)** | Runnable, feature-complete for a demo | **~75%** |
| **Desktop MediaPipe app** | Runnable if Python deps are installed | **~80%** |
| **Original CNN / Keras pipeline** | Source present, training data and model missing | **~20%** |
| **Overall (as a shippable product)** | Working prototype, not production-ready | **~70%** |

The **working product** is the browser app: Flask serves `index.html`, MediaPipe runs in the browser, OTP login and translation hit the Python backend. Sign recognition is **rule-based finger geometry**, not a trained deep-learning classifier.

The original HackUNT-19 CNN pipeline (`final.py`, `cnn_model_train.py`, gesture capture scripts) is **not runnable in this checkout**. Gesture images, the histogram file, SQLite labels, and `cnn_model_keras2.h5` are not in the repository.

---

## 2. What the project does

1. User logs in with email OTP and a math CAPTCHA.
2. Browser camera + MediaPipe Hand Landmarker tracks 21 hand landmarks.
3. A custom classifier maps finger pose to either:
   - **Phrase mode** — whole words (`Hello`, `Yes`, `I Love You`, …)
   - **Spell mode** — letters / digits (`A`, `B`, `L`, `5`, …)
4. Signs are temporally smoothed (buffer of 14 frames, commit after 9 agreeing frames) so jitter does not spam letters.
5. The user builds a sentence, can hear it via speech synthesis, and can translate it to another language through the Flask `/api/translate` endpoint.

---

## 3. Repository layout

Git remote: `https://github.com/KartikeyaShrivastava/Vani-Setu.git`

```
Vani-Setu/
└── Sign-Language-Interpreter-using-Deep-Learning-master/
    ├── RUN_VANI_SETU.bat          # Windows launcher (install + Flask + browser)
    ├── QUICK_START.bat            # Windows launcher (kills port 5000, then starts)
    ├── README.md                  # Original hackathon README + Vani-Setu notes
    ├── LICENSE
    └── Code/
        ├── index.html             # Full web UI + MediaPipe classifier (main product)
        ├── server.py              # Flask: OTP, translation, static files
        ├── requirements_web.txt   # flask, flask-cors, deep-translator
        ├── sign_language_app.py   # Desktop OpenCV + MediaPipe translator
        ├── final.py               # Legacy CNN webcam app (broken without model)
        ├── cnn_model_train.py     # Legacy Keras CNN trainer
        ├── set_hand_histogram.py  # Legacy skin-histogram capture
        ├── create_gestures.py     # Legacy gesture dataset capture
        ├── Rotate_images.py       # Legacy dataset augmentation
        ├── load_images.py         # Legacy train/val/test pickle split
        ├── display_gestures.py    # Legacy gesture grid viewer
        ├── Install_Packages.txt
        └── Install_Packages_gpu.txt
```

**Not present in this checkout (required by the legacy pipeline):**

- `gestures/` image dataset
- `hist` hand-histogram pickle
- `gesture_db.db`
- `cnn_model_keras2.h5`
- `cnn_tf.py` (imported by `final.py`)
- `train_images` / `train_labels` / `val_images` / `val_labels` pickles
- `img/` demo screenshots referenced by the original README

---

## 4. Architecture

```
Browser (index.html)
  ├── MediaPipe HandLandmarker (CDN + Google model)
  ├── Rule-based ASL classifier (JavaScript)
  ├── Web Speech API (TTS)
  └── sessionStorage (login flag + email)
        │
        │  HTTP
        ▼
Flask server.py  (localhost:5000)
  ├── GET  /                 → index.html
  ├── POST /api/send-otp     → 6-digit OTP, email or console
  ├── POST /api/verify-otp   → in-memory OTP check
  ├── POST /api/translate    → Google Translate via deep-translator
  └── GET  /api/languages    → language map
```

Hand pixels stay in the browser. Only OTP emails and translation strings go to the backend (and then to Gmail / Google Translate).

The desktop app (`sign_language_app.py`) is a parallel stack: OpenCV window + MediaPipe Tasks Python + `pyttsx3`. It does **not** use Flask, OTP, or translation.

---

## 5. Working features (web app)

### 5.1 Authentication — working

- Email validation, send OTP, 6-digit input with paste support, 60s resend cooldown.
- Server: 5-minute expiry, max 5 attempts, 60s rate limit per email, one-time use.
- After OTP, a client-side arithmetic CAPTCHA must be solved.
- Session is stored in `sessionStorage` (`vani_logged_in`, `vani_email`). Refresh keeps the user logged in for that tab.

**Limits:** OTP store is in-memory (lost on server restart). There is no user database, JWT, or password. CAPTCHA is not a real bot defense. If SMTP fails, the server still returns success and prints the OTP in the console.

### 5.2 Camera and hand tracking — working

- MediaPipe Tasks Vision `0.10.18` loaded from jsDelivr.
- Hand skeleton overlay, FPS badge, per-finger on/off indicators, TRACKING / NO HAND badge.
- Requires Chrome or Edge, HTTPS or `localhost`, and camera permission.

### 5.3 Sign classification — working (limited vocabulary)

**Phrase mode (15 signs)**

| Sign | Approximate pose |
| --- | --- |
| Hello | Open hand, fingers spread |
| Stop | Open hand, fingers together |
| Good | Thumbs up |
| Yes | Fist, thumb up |
| No | Closed fist |
| I Love You | Thumb + index + pinky |
| Peace | V-sign, spread |
| OK | Thumb–index circle |
| Call Me | Thumb + pinky |
| Rock On | Index + pinky |
| Wait | Index up |
| Me | Pinky up |
| Awesome | L-shape |
| You | Index + middle together |
| Three | Three fingers spread |

**Spell mode (implemented letters)**

`A B D E F G I L O S U V W Y` and digit `5`.  
`C` is listed in the UI help text but the JavaScript classifier does not emit `C`.  
`K` is listed in the desktop app help text but is not classified.

**Not implemented:** `H J M N P Q R T X Z`, motion letters, two-handed signs, Indian Sign Language (ISL).

This is **not** full ASL. It is a landmark heuristic that looks similar to common beginner signs.

### 5.4 Sentence building and speech — working

| Control | Action |
| --- | --- |
| Phrase commit | Appends the word to the sentence and speaks it |
| Spell commit | Appends a letter to the current word |
| Space | Finishes the current word |
| Backspace | Deletes last letter or last word |
| Enter | Ends sentence with a period, logs it, speaks it |
| C | Clears text |
| V | Toggles voice |
| S / Speak | Speaks English, then the translated line if another language is selected |
| M | Toggles Phrase / Spell |

Stats: signs / words / sentences. Activity log of recent detections.

### 5.5 Multi-language translation — working (needs backend)

Supported codes: English, Hindi, French, Spanish, German, Japanese, Arabic, Bengali, Tamil, Korean, Portuguese, Russian, Simplified Chinese, Urdu, Marathi, Telugu, Gujarati, Kannada, Malayalam, Punjabi, Odia, Italian, Turkish, Thai, Vietnamese.

English stays on the main panel. Other languages appear in a secondary panel via `/api/translate`. Results are cached on both client and server.

Requires `deep-translator` and network access to Google Translate. If the package or network is missing, the UI shows `(translation unavailable)` or `(server unreachable)`.

---

## 6. Incomplete or non-functional pieces

| Item | What you see | Reality |
| --- | --- | --- |
| **ISL tab** | Indian Sign Language tab next to ASL | No click handler. Does nothing. |
| **About Us / Home** | Navbar links | `href="#"`. No pages. |
| **True deep learning** | README claims CNN + Keras + >95% accuracy | Web and desktop apps do **not** load a CNN. Classification is geometric rules. |
| **Legacy `final.py`** | Original webcam + calculator modes | Needs missing `cnn_model_keras2.h5`, `hist`, `gesture_db.db`, `cnn_tf.py`, `gestures/`. Also uses deprecated Keras APIs. |
| **Training pipeline** | Capture → rotate → split → train | Code exists. Dataset and model artifacts are not in the repo. `create_gestures.py` uses OpenCV 3 contour indexing (`[1]`), which breaks on OpenCV 4. |
| **macOS / Linux launch** | `RUN_VANI_SETU.bat`, `QUICK_START.bat` | Windows only. On macOS/Linux run `python server.py` from `Code/`. |
| **Production auth** | OTP email login | Hardcoded SMTP credentials in `server.py`. OTP always treated as “sent.” No HTTPS, no persistent users. |
| **Two-handed / continuous signing** | — | Single hand only. No movement trajectories (J, Z). |

---

## 7. Desktop app (`sign_language_app.py`)

A local OpenCV window with the same Phrase / Spell classifier, TTS (`pyttsx3`), sentence history, and keyboard shortcuts (`M`, Space, Backspace, `C`, `V`, `S`, Enter, `Q`).

**Dependencies:** `opencv-python`, `mediapipe`, `pyttsx3`, `numpy`. On first run it downloads `hand_landmarker.task` from Google storage.

It does not include OTP, translation, or the web UI. It is the best way to use the translator without a browser.

---

## 8. Backend API

Base URL: `http://localhost:5000`

### `POST /api/send-otp`

```json
{ "email": "user@example.com" }
```

Success: `{ "success": true, "message": "OTP sent successfully! Check your email." }`  
Errors: 400 invalid email, 429 rate limited.

### `POST /api/verify-otp`

```json
{ "email": "user@example.com", "otp": "123456" }
```

Success deletes the OTP (one-time). Failures: expired, not found, wrong code, too many attempts.

### `POST /api/translate`

```json
{ "text": "Hello", "target": "hi", "source": "en" }
```

Returns `{ "success": true, "translated": "नमस्ते" }`.

### `GET /api/languages`

Returns the supported language map.

---

## 9. How to run

### Web app (primary)

From `Sign-Language-Interpreter-using-Deep-Learning-master/Code/`:

```bash
pip install -r requirements_web.txt
python server.py
```

Open [http://localhost:5000](http://localhost:5000).

On Windows you can double-click `RUN_VANI_SETU.bat` instead.

**Flow:** enter email → read OTP from email or the Flask console → enter 6 digits → solve CAPTCHA → Start Camera → show a hand.

SMTP is configured via environment variables (recommended):

```bash
export SMTP_USER="your-gmail@gmail.com"
export SMTP_PASS="your-app-password"
python server.py
```

If email is not configured, the OTP is printed in the terminal.

### Desktop app

```bash
pip install opencv-python mediapipe pyttsx3 numpy
python sign_language_app.py
```

### Legacy CNN pipeline (not usable until data is restored)

1. `python set_hand_histogram.py` — press `c` to capture skin histogram, `s` to save.
2. `python create_gestures.py` — record labeled gestures.
3. `python Rotate_images.py` — flip augmentation.
4. `python load_images.py` — write train/val/test pickles.
5. `python cnn_model_train.py` — train Keras CNN.
6. `python final.py` — webcam recognition.

This path also needs Keras/TensorFlow versions compatible with `K.set_image_dim_ordering` and `keras.utils.np_utils` (old Keras 2). Modern TensorFlow will not run this file unchanged.

---

## 10. Security notes

These issues matter if the app is deployed or shared:

1. **SMTP username and app password are hardcoded** in `server.py`. Rotate that Gmail app password and load secrets from the environment only.
2. **CORS is wide open** (`flask-cors` with default config). Fine for local demo; not for production.
3. **OTP fallback always returns success**, including when email sending fails. Anyone who can read server logs can log in.
4. **`create_gestures.py` builds SQL with string formatting** (injection risk if that script is reused).
5. Login is a browser `sessionStorage` flag. It is not a server session.

---

## 11. Technology stack

| Layer | Technology |
| --- | --- |
| Web UI | Single HTML file, Inter / JetBrains Mono, CSS custom properties |
| Hand tracking | MediaPipe Hand Landmarker (browser WASM + Python Tasks API) |
| Classification | Custom 21-landmark heuristics + temporal voting |
| Backend | Flask, flask-cors |
| Translation | deep-translator → Google Translate |
| Speech | Web Speech API (web), pyttsx3 (desktop) |
| Legacy ML | Keras Sequential CNN (3× Conv2D + Dense), OpenCV histogram back-projection |
| Original project | HackUNT 2019, MIT-style license in this fork |

---

## 12. Honest assessment

**What works today**

- A polished local web demo: login → camera → detect a small set of signs → speak and translate.
- A matching desktop OpenCV app with the same classifier.
- Flask OTP and translation endpoints that function for a demo.

**What does not match the original marketing**

- It does not interpret full ASL or 44 characters with a trained CNN in this repo.
- Indian Sign Language is a label only.
- The prize-winning 2019 CNN system cannot be reproduced from this checkout because models and gesture data are missing.

**Recommended next steps**

1. Remove hardcoded SMTP secrets; use environment variables.
2. Wire or remove the ISL tab and About page.
3. Add macOS/Linux start scripts.
4. Expand spell-mode letters (especially `C`, `K`, and missing alphabet).
5. If true ML is required, restore or recapture the gesture dataset and retrain with current TensorFlow/Keras APIs.
6. Replace in-memory OTP with a real store if more than one server process is needed.

---

## 13. Keyboard reference (web and desktop)

| Key | Action |
| --- | --- |
| `M` | Toggle Phrase / Spell |
| `Space` | Finish word |
| `Backspace` | Delete |
| `Enter` | Finish sentence |
| `C` | Clear |
| `V` | Voice on/off |
| `S` | Speak sentence |
| `Q` | Quit (desktop only) |

---

## 14. Credits and lineage

- **This fork / product name:** Vani-Setu (web UI, OTP, translation, MediaPipe rewrite).
- **Upstream:** [harshbg/Sign-Language-Interpreter-using-Deep-Learning](https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning) — HackUNT 2019, team of Harsh Gupta, Siddharth Oza, Ashish Sharma, and Manish Shukla.
- **Citation** for the original work is in `README.md`.
