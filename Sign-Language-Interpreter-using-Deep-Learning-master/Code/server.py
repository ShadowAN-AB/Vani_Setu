"""
Vani-Setu backend: password auth, translation, landmark collection, and ML classify.
"""

import json
import os
import random
import re
import string
import subprocess
import sys
import time
import smtplib
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import data as dataset
from features import hands_to_two_hand_features, landmarks_to_features
from nn.numpy_mlp import NumpyMLP
from suggest import Suggester
from models_user import User, db, init_db
from signs import OFFLINE_HI, catalog

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("  [WARN] deep-translator not installed.")

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "dataset")
MODEL_PATH = os.path.join(APP_DIR, "models", "sign_svm.joblib")
LEGACY_MODEL_PATH = os.path.join(APP_DIR, "models", "sign_rf.joblib")  # pre-v2 file name
DEFAULT_THRESHOLD = 0.50
# Neural-network models for whole signs, trained with nn/train_mlp.py (one file per kind)
SIGN_MODEL_PATHS = {
    "phrase": os.path.join(APP_DIR, "models", "phrase_mlp.npz"),
    "isl": os.path.join(APP_DIR, "models", "isl_mlp.npz"),
}
HAND_TASK_PATH = os.path.join(APP_DIR, "hand_landmarker.task")
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
TARGET_SAMPLES_PER_LABEL = 25

# Only these files are served from the app folder. Everything else in it
# (users.db, .env, datasets, source code) must never be downloadable.
PUBLIC_FILES = {"index.html", "hand_landmarker.task", "favicon.ico"}


def _on_paas():
    return any(
        os.environ.get(k)
        for k in ("RENDER", "RAILWAY_ENVIRONMENT", "FLY_APP_NAME", "K_SERVICE")
    )


SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
SENDER_NAME = "Vani-Setu"
PORT = int(os.environ.get("PORT", "5001"))
HOST = os.environ.get("HOST", "0.0.0.0" if _on_paas() else "127.0.0.1").strip() or (
    "0.0.0.0" if _on_paas() else "127.0.0.1"
)
# password = email + password stored in SQLite (or DATABASE_URL Postgres)
# otp      = email code if SMTP is set, otherwise the terminal (local only)
# guest    = password/OTP plus a "Continue as guest" button
# none     = skip the login screen entirely
AUTH_MODE = os.environ.get("AUTH_MODE", "password").strip().lower()
if AUTH_MODE not in ("password", "otp", "guest", "none"):
    AUTH_MODE = "password"
EMAIL_CONFIGURED = bool(SMTP_USER and SMTP_PASS)

# Recording samples and retraining change the shared model, so they are on for
# localhost and off on a public host unless ALLOW_TRAINING=1 is set explicitly.
_allow = os.environ.get("ALLOW_TRAINING", "").strip().lower()
TRAINING_ENABLED = (_allow in ("1", "true", "yes")) if _allow else not _on_paas()

OTP_LENGTH = 6
OTP_EXPIRY_SECONDS = 300
MAX_ATTEMPTS = 5
RATE_LIMIT_SECONDS = 60

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
    "de": "German", "ja": "Japanese", "ar": "Arabic", "bn": "Bengali",
    "ta": "Tamil", "ko": "Korean", "pt": "Portuguese", "ru": "Russian",
    "zh-CN": "Chinese (Simplified)", "ur": "Urdu", "mr": "Marathi",
    "te": "Telugu", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "it": "Italian", "tr": "Turkish",
    "th": "Thai", "vi": "Vietnamese",
}

otp_store = {}
store_lock = Lock()
translation_cache = {}
cache_lock = Lock()
csv_lock = Lock()
train_lock = Lock()
MAX_CACHE_SIZE = 500
ml_model = None       # the classifier (has predict_proba and classes_)
ml_info = {}          # feature version, threshold and headline metrics saved by the trainer
sign_models = {}      # kind -> NumpyMLP for phrase / ISL signs
suggester = Suggester()  # word prediction; uses the letter model's confusion counts


def ensure_hand_landmarker():
    """Fetch MediaPipe's hand model if it is missing (fresh clone or a host)."""
    if os.path.exists(HAND_TASK_PATH) and os.path.getsize(HAND_TASK_PATH) > 1000:
        return
    print("  [OK] Downloading hand_landmarker.task (first run)...")
    tmp_path = HAND_TASK_PATH + ".tmp"
    try:
        urllib.request.urlretrieve(HAND_TASK_URL, tmp_path)
        os.replace(tmp_path, HAND_TASK_PATH)
        print(f"  [OK] Saved {HAND_TASK_PATH}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"  [WARN] Could not download hand_landmarker.task: {e}")
        print(f"  [WARN] Get it from: {HAND_TASK_URL}")


def load_ml_model():
    """Load the trainer's bundle; an old bare pipeline still works with version-1 features."""
    global ml_model, ml_info
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
    if not JOBLIB_OK or not os.path.exists(path):
        ml_model, ml_info = None, {}
        return
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            ml_model = obj["model"]
            ml_info = {
                "path": os.path.relpath(path, APP_DIR),
                "feature_version": int(obj.get("feature_version", 1)),
                "threshold": float(obj.get("threshold", DEFAULT_THRESHOLD)),
                "metrics": obj.get("metrics", {}),
            }
        else:
            ml_model = obj
            ml_info = {"path": os.path.relpath(path, APP_DIR), "feature_version": 1,
                       "threshold": DEFAULT_THRESHOLD, "metrics": {}}
        print(f"  [OK] Loaded classifier: {ml_info['path']} (features v{ml_info['feature_version']})")
        suggester.set_confusion(obj.get("confusion") if isinstance(obj, dict) else None)
    except Exception as e:
        ml_model, ml_info = None, {}
        print(f"  [WARN] Could not load model: {e}")


def load_sign_models():
    """Load whichever phrase / ISL networks have been trained; Phrase and ISL modes use rules otherwise."""
    global sign_models
    loaded = {}
    for kind, path in SIGN_MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded[kind] = NumpyMLP(path)
                print(f"  [OK] Loaded {kind} network: {len(loaded[kind].classes_)} signs")
            except Exception as e:
                print(f"  [WARN] Could not load {path}: {e}")
    sign_models = loaded


def generate_otp():
    return "".join(random.choices(string.digits, k=OTP_LENGTH))


def cleanup_expired():
    now = time.time()
    expired = [k for k, v in otp_store.items() if now - v["created_at"] > OTP_EXPIRY_SECONDS]
    for k in expired:
        del otp_store[k]


def send_email_otp(email, otp):
    if not SMTP_USER or not SMTP_PASS:
        print(f"\n{'=' * 50}")
        print(f"  OTP for {email}: {otp}")
        print("  (Email not configured — use the code above)")
        print(f"{'=' * 50}\n")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Vani-Setu Login OTP: {otp}"
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = email
        html = f"""
        <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:24px;">
          <h2>Vani-Setu</h2>
          <p>Your verification code:</p>
          <p style="font-size:28px;letter-spacing:8px;font-weight:800;">{otp}</p>
          <p>Expires in 5 minutes.</p>
        </div>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, email, msg.as_string())
        print(f"  [OK] OTP email sent to {email}")
        return True
    except Exception as e:
        print(f"  [ERROR] Email send failed: {e}")
        print(f"  OTP for {email}: {otp} (console fallback)")
        return False


def offline_translate(text, target):
    if target != "hi":
        return None
    words = re.split(r"(\s+)", text)
    out = []
    for w in words:
        key = w.strip(".,!?").lower()
        if key in OFFLINE_HI:
            out.append(OFFLINE_HI[key])
        else:
            out.append(w)
    joined = "".join(out).strip()
    return joined if joined and joined.lower() != text.lower() else None


app = Flask(__name__, static_folder=APP_DIR)
CORS(app)
ensure_hand_landmarker()
init_db(app)
load_ml_model()
load_sign_models()


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    if filename not in PUBLIC_FILES:
        return jsonify({"success": False, "message": "Not found"}), 404
    return send_from_directory(APP_DIR, filename)


def _training_disabled():
    return jsonify({
        "success": False,
        "message": "Recording and retraining are turned off on this server. Run the app locally to use them.",
    }), 403


@app.route("/api/auth-config", methods=["GET"])
def auth_config():
    return jsonify({
        "success": True,
        "mode": AUTH_MODE,
        "email_configured": EMAIL_CONFIGURED,
        "guest_allowed": AUTH_MODE in ("guest", "none"),
        "password_enabled": AUTH_MODE in ("password", "guest"),
        "otp_enabled": AUTH_MODE in ("otp", "guest"),
    })


def _valid_email(email):
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email or ""))


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    if not _valid_email(email):
        return jsonify({"success": False, "message": "Enter a valid email address"}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "That email is already registered. Sign in instead."}), 409
    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"success": True, "message": "Account created. You are signed in.", "email": user.email})


@app.route("/api/login", methods=["POST"])
def password_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    user = User.query.filter_by(email=email).first() if _valid_email(email) else None
    if not user or not user.check_password(password):
        return jsonify({"success": False, "message": "Invalid email or password"}), 401
    return jsonify({"success": True, "message": "Signed in.", "email": user.email})


@app.route("/api/guest-login", methods=["POST"])
def guest_login():
    if AUTH_MODE not in ("guest", "none"):
        return jsonify({"success": False, "message": "Guest login is disabled"}), 403
    return jsonify({"success": True, "email": "guest@vani-setu.local", "message": "Continuing as guest"})


@app.route("/api/send-otp", methods=["POST"])
def send_otp():
    data = request.get_json(silent=True)
    if not data or "email" not in data:
        return jsonify({"success": False, "message": "Email is required"}), 400

    email = data["email"].strip().lower()
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        return jsonify({"success": False, "message": "Invalid email format"}), 400

    with store_lock:
        cleanup_expired()
        if email in otp_store:
            elapsed = time.time() - otp_store[email]["created_at"]
            if elapsed < RATE_LIMIT_SECONDS:
                remaining = int(RATE_LIMIT_SECONDS - elapsed)
                return jsonify({
                    "success": False,
                    "message": f"Please wait {remaining}s before requesting a new OTP",
                }), 429

        otp = generate_otp()
        otp_store[email] = {"otp": otp, "created_at": time.time(), "attempts": 0}
        print(f"  [OTP] {email} -> {otp}", flush=True)

    emailed = send_email_otp(email, otp)
    message = (
        "OTP sent to your email."
        if emailed
        else "OTP printed in the server terminal (email is not configured)."
    )
    return jsonify({"success": True, "message": message, "console_otp": not emailed})


@app.route("/api/verify-otp", methods=["POST"])
def verify_otp():
    data = request.get_json(silent=True)
    if not data or "email" not in data or "otp" not in data:
        return jsonify({"success": False, "message": "Email and OTP are required"}), 400

    email = data["email"].strip().lower()
    user_otp = data["otp"].strip()

    with store_lock:
        cleanup_expired()
        if email not in otp_store:
            return jsonify({"success": False, "message": "OTP expired or not found. Request a new one."}), 400
        entry = otp_store[email]
        if time.time() - entry["created_at"] > OTP_EXPIRY_SECONDS:
            del otp_store[email]
            return jsonify({"success": False, "message": "OTP has expired. Request a new one."}), 400
        if entry["attempts"] >= MAX_ATTEMPTS:
            del otp_store[email]
            return jsonify({"success": False, "message": "Too many attempts. Request a new OTP."}), 400
        if user_otp == entry["otp"]:
            del otp_store[email]
            return jsonify({"success": True, "message": "Login successful! Welcome to Vani-Setu."})
        entry["attempts"] += 1
        remaining = MAX_ATTEMPTS - entry["attempts"]
        return jsonify({"success": False, "message": f"Incorrect OTP. {remaining} attempts remaining."}), 400


@app.route("/api/translate", methods=["POST"])
def translate_text():
    data = request.get_json(silent=True)
    if not data or "text" not in data or "target" not in data:
        return jsonify({"success": False, "message": "text and target language are required"}), 400

    text = data["text"].strip()
    target = data["target"].strip()
    source = data.get("source", "en").strip()

    if not text:
        return jsonify({"success": True, "translated": "", "source": "empty"})
    if target == source:
        return jsonify({"success": True, "translated": text, "source": "same"})
    if target not in SUPPORTED_LANGUAGES:
        return jsonify({"success": False, "message": f"Unsupported language: {target}"}), 400

    cache_key = f"{source}:{target}:{text}"
    with cache_lock:
        if cache_key in translation_cache:
            return jsonify({"success": True, "translated": translation_cache[cache_key], "source": "cache"})

    translated = None
    used = None
    if TRANSLATOR_AVAILABLE:
        try:
            translated = GoogleTranslator(source=source, target=target).translate(text)
            used = "google"
        except Exception as e:
            print(f"  [ERROR] Translation failed: {e}")

    if not translated:
        translated = offline_translate(text, target)
        used = "offline" if translated else None

    if not translated:
        return jsonify({
            "success": False,
            "message": "Translation unavailable. Offline Hindi works for known signs.",
        }), 503

    with cache_lock:
        if len(translation_cache) >= MAX_CACHE_SIZE:
            for k in list(translation_cache.keys())[:100]:
                del translation_cache[k]
        translation_cache[cache_key] = translated

    return jsonify({"success": True, "translated": translated, "source": used})


@app.route("/api/languages", methods=["GET"])
def get_languages():
    return jsonify({"success": True, "languages": SUPPORTED_LANGUAGES})


@app.route("/api/signs", methods=["GET"])
def get_signs():
    return jsonify({"success": True, "catalog": catalog()})


@app.route("/api/health", methods=["GET"])
def health():
    checks = {
        "server": True,
        "model_loaded": ml_model is not None,
        "dataset_letters": dataset.counts("letter"),
        "dataset_phrases": dataset.counts("phrase"),
        "hand_model_file": os.path.exists(os.path.join(APP_DIR, "hand_landmarker.task")),
        "translator": TRANSLATOR_AVAILABLE,
        "email_configured": EMAIL_CONFIGURED,
        "auth_mode": AUTH_MODE,
    }
    ready = checks["model_loaded"] and checks["hand_model_file"]
    return jsonify({"success": True, "ready_for_demo": ready, "checks": checks})


@app.route("/api/model-status", methods=["GET"])
def model_status():
    return jsonify({
        "success": True,
        "model_loaded": ml_model is not None,
        "model_path": ml_info.get("path"),
        "feature_version": ml_info.get("feature_version"),
        "threshold": ml_info.get("threshold", DEFAULT_THRESHOLD),
        "test_accuracy": (ml_info.get("metrics") or {}).get("accuracy"),
        "samples": dataset.counts("letter"),
        "phrase_samples": dataset.counts("phrase"),
        "target_per_label": TARGET_SAMPLES_PER_LABEL,
        "training_enabled": TRAINING_ENABLED,
        "sign_models": {
            kind: {"loaded": kind in sign_models,
                   "signs": [str(c) for c in sign_models[kind].classes_] if kind in sign_models else [],
                   "threshold": sign_models[kind].threshold if kind in sign_models else None,
                   "samples": dataset.counts(kind)}
            for kind in SIGN_MODEL_PATHS
        },
    })


@app.route("/api/collect", methods=["POST"])
def collect_sample():
    """Save one sample as raw landmarks plus who signed it, which hand, and its split."""
    if not TRAINING_ENABLED:
        return _training_disabled()
    data = request.get_json(silent=True)
    if not data or "label" not in data or ("landmarks" not in data and "hands" not in data):
        return jsonify({"success": False, "message": "label and landmarks are required"}), 400

    label = str(data["label"]).strip()
    if not label or label == "?":
        return jsonify({"success": False, "message": "Pick a sign label first"}), 400

    # Letters, ASL phrases and ISL signs share one file but are kept apart by "kind",
    # so phrase samples never leak into the letter model.
    kind = str(data.get("kind") or "letter").strip().lower()
    if kind not in dataset.KINDS:
        kind = "letter"

    try:
        with csv_lock:
            hands = dataset.parse_hands(data)
            dataset.append_recording(label, hands, kind=kind,
                                     signer=data.get("signer", ""), split=data.get("split", ""))
            count = dataset.counts(kind)
            label_count = dataset.counts(kind, label)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    return jsonify({
        "success": True,
        "samples": count,
        "label_count": label_count,
        "label": label,
        "kind": kind,
        "hands": len(hands),
        "split": "test" if str(data.get("split", "")).lower() == "test" else "train",
        "target": TARGET_SAMPLES_PER_LABEL,
        "path": os.path.relpath(dataset.RECORDINGS_CSV, APP_DIR),
    })


@app.route("/api/train", methods=["POST"])
def train_model():
    """Retrain from the browser and hot-reload, so a demo never needs a terminal.
    kind=letter runs train_classifier.py (SVM); kind=phrase / isl runs nn/train_mlp.py (MLP)."""
    if not TRAINING_ENABLED:
        return _training_disabled()
    body = request.get_json(silent=True) or {}
    kind = str(body.get("kind") or "letter").lower()
    if kind not in dataset.KINDS:
        kind = "letter"
    if not train_lock.acquire(blocking=False):
        return jsonify({"success": False, "message": "Training already in progress"}), 409

    try:
        if kind == "letter":
            cmd = [sys.executable, os.path.join(APP_DIR, "train_classifier.py")]
            report = os.path.join(APP_DIR, "reports", "metrics.json")
        else:
            cmd = [sys.executable, os.path.join(APP_DIR, "nn", "train_mlp.py"), "--kind", kind]
            report = os.path.join(APP_DIR, "reports", f"nn_{kind}.json")
        proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            lines = [l for l in (proc.stdout + "\n" + proc.stderr).strip().splitlines() if l.strip()]
            message = lines[-1] if lines else "Training failed"
            if proc.returncode == 2 and "TensorFlow" in proc.stdout:
                message = "Training the neural network needs TensorFlow: pip install -r requirements_train.txt"
            elif "No " in proc.stdout and "samples yet" in proc.stdout:
                message = f"No {kind} samples yet. Record some first."
            return jsonify({"success": False, "message": message}), 500

        load_ml_model()
        load_sign_models()
        metrics = {}
        if os.path.exists(report):
            with open(report, encoding="utf-8") as f:
                metrics = json.load(f)
        test = metrics.get("test") or {}
        return jsonify({
            "success": True,
            "kind": kind,
            "accuracy": metrics.get("accuracy", test.get("accuracy")),
            "n_samples": metrics.get("n_samples", sum((metrics.get("split") or {}).get(k, 0) for k in ("train", "validation", "test"))),
            "n_classes": metrics.get("n_classes", len(metrics.get("classes", []))),
            "model_loaded": ml_model is not None if kind == "letter" else kind in sign_models,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "message": "Training timed out"}), 504
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        train_lock.release()


@app.route("/api/suggest", methods=["POST"])
def suggest_words():
    """Noisy-channel word prediction for the letters spelled so far (see suggest.py)."""
    data = request.get_json(silent=True) or {}
    letters = str(data.get("letters") or "")[:20]
    return jsonify({"success": True, "letters": letters.upper(), "suggestions": suggester.suggest(letters, k=4)})


@app.route("/api/classify", methods=["POST"])
def classify_landmarks():
    """mode=letter: SVM on the first hand. mode=phrase / isl: the MLP for that kind on both hands."""
    data = request.get_json(silent=True)
    if not data or ("landmarks" not in data and "hands" not in data):
        return jsonify({"success": False, "message": "landmarks required"}), 400
    mode = str(data.get("mode") or "letter").lower()

    start = time.perf_counter()
    try:
        hands = dataset.parse_hands(data)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    if mode in SIGN_MODEL_PATHS:
        model = sign_models.get(mode)
        if model is None:
            return jsonify({"success": False, "message": f"No {mode} network trained yet"}), 404
        feats = hands_to_two_hand_features(hands)
        threshold = model.threshold
    else:
        model = ml_model
        if model is None:
            return jsonify({"success": False, "message": "No trained model yet. Run train_classifier.py"}), 404
        feats = landmarks_to_features(hands[0][0], ml_info.get("feature_version", 1))
        threshold = ml_info.get("threshold", DEFAULT_THRESHOLD)

    try:
        # The label always comes from the highest probability, so the label and
        # the confidence shown in the app can never disagree.
        probs = model.predict_proba([feats])[0]
        classes = [str(c) for c in model.classes_]
        order = sorted(range(len(classes)), key=lambda i: -probs[i])
        top = [{"label": classes[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
        return jsonify({
            "success": True,
            "mode": mode if mode in SIGN_MODEL_PATHS else "letter",
            "label": top[0]["label"],
            "confidence": top[0]["confidence"],
            "top": top,
            "threshold": threshold,
            "server_ms": round((time.perf_counter() - start) * 1000, 2),
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    print()
    print("  +===============================================+")
    print("  |     Vani-Setu                                 |")
    print(f"  |     Auth: {AUTH_MODE:<36} |")
    if AUTH_MODE == "password":
        print("  |     Login: email + password (SQLite)          |")
    elif AUTH_MODE == "guest":
        print("  |     Guest login enabled                       |")
    elif AUTH_MODE == "none":
        print("  |     Login screen skipped                      |")
    elif EMAIL_CONFIGURED:
        print(f"  |     Email: {SMTP_USER[:34]:<34} |")
    else:
        print("  |     OTP: terminal (set SMTP or AUTH_MODE)     |")
    print(f"  |     ML model: {'loaded':<32} |" if ml_model is not None else "  |     ML model: not trained yet               |")
    print(f"  |     http://{HOST}:{PORT:<22} |")
    print("  +===============================================+")
    print()
    app.run(host=HOST, port=PORT, debug=False)
