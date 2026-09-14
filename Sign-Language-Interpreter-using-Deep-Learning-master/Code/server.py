"""
Vani-Setu backend: OTP auth, translation, landmark collection, and ML classify.
"""

import csv
import json
import os
import random
import re
import string
import subprocess
import sys
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from features import landmarks_to_features
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
MODEL_PATH = os.path.join(APP_DIR, "models", "sign_rf.joblib")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
PHRASE_CSV_PATH = os.path.join(DATA_DIR, "landmarks_phrases.csv")
TARGET_SAMPLES_PER_LABEL = 25


def _count_csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _count_label(path, label):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f) if row.get("label") == label)

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
SENDER_NAME = "Vani-Setu"
PORT = int(os.environ.get("PORT", "5001"))
HOST = os.environ.get("HOST", "127.0.0.1").strip() or "127.0.0.1"
# password = email + password stored in SQLite (or DATABASE_URL Postgres)
# otp      = email code if SMTP is set, otherwise the terminal (local only)
# guest    = password/OTP plus a "Continue as guest" button
# none     = skip the login screen entirely
AUTH_MODE = os.environ.get("AUTH_MODE", "password").strip().lower()
if AUTH_MODE not in ("password", "otp", "guest", "none"):
    AUTH_MODE = "password"
EMAIL_CONFIGURED = bool(SMTP_USER and SMTP_PASS)

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
ml_model = None


def load_ml_model():
    global ml_model
    if not JOBLIB_OK or not os.path.exists(MODEL_PATH):
        ml_model = None
        return
    try:
        ml_model = joblib.load(MODEL_PATH)
        print(f"  [OK] Loaded classifier: {MODEL_PATH}")
    except Exception as e:
        ml_model = None
        print(f"  [WARN] Could not load model: {e}")


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
init_db(app)
load_ml_model()


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(APP_DIR, filename)


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
        "dataset_letters": _count_csv_rows(CSV_PATH),
        "dataset_phrases": _count_csv_rows(PHRASE_CSV_PATH),
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
        "model_path": os.path.relpath(MODEL_PATH, APP_DIR) if os.path.exists(MODEL_PATH) else None,
        "samples": _count_csv_rows(CSV_PATH),
        "phrase_samples": _count_csv_rows(PHRASE_CSV_PATH),
        "target_per_label": TARGET_SAMPLES_PER_LABEL,
    })


@app.route("/api/collect", methods=["POST"])
def collect_sample():
    data = request.get_json(silent=True)
    if not data or "label" not in data or "landmarks" not in data:
        return jsonify({"success": False, "message": "label and landmarks are required"}), 400

    label = str(data["label"]).strip()
    if not label or label == "?":
        return jsonify({"success": False, "message": "Pick a sign label first"}), 400

    # Keep phrase samples out of the letter CSV so Spell ML stays clean
    kind = str(data.get("kind") or "letter").strip().lower()
    if kind not in ("letter", "phrase"):
        kind = "letter"
    path = PHRASE_CSV_PATH if kind == "phrase" else CSV_PATH

    try:
        feats = landmarks_to_features(data["landmarks"])
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    os.makedirs(DATA_DIR, exist_ok=True)
    header = ["label"] + [f"f{i}" for i in range(len(feats))]
    with csv_lock:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow([label] + feats)
        count = _count_csv_rows(path)
        label_count = _count_label(path, label)

    return jsonify({
        "success": True,
        "samples": count,
        "label_count": label_count,
        "label": label,
        "kind": kind,
        "target": TARGET_SAMPLES_PER_LABEL,
        "path": os.path.basename(path),
    })


@app.route("/api/train", methods=["POST"])
def train_model():
    """Run train_classifier.py and hot-reload the model, so a demo never needs a terminal."""
    if not train_lock.acquire(blocking=False):
        return jsonify({"success": False, "message": "Training already in progress"}), 409

    try:
        script = os.path.join(APP_DIR, "train_classifier.py")
        proc = subprocess.run(
            [sys.executable, script],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()
            return jsonify({
                "success": False,
                "message": tail[-1] if tail else "Training failed",
            }), 500

        load_ml_model()

        metrics = {}
        metrics_path = os.path.join(APP_DIR, "reports", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)

        return jsonify({
            "success": True,
            "accuracy": metrics.get("accuracy"),
            "n_samples": metrics.get("n_samples"),
            "n_classes": metrics.get("n_classes"),
            "model_loaded": ml_model is not None,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "message": "Training timed out"}), 504
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        train_lock.release()


@app.route("/api/classify", methods=["POST"])
def classify_landmarks():
    if ml_model is None:
        return jsonify({"success": False, "message": "No trained model yet. Record samples and run train_classifier.py"}), 404

    data = request.get_json(silent=True)
    if not data or "landmarks" not in data:
        return jsonify({"success": False, "message": "landmarks required"}), 400

    try:
        feats = landmarks_to_features(data["landmarks"])
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    try:
        pred = ml_model.predict([feats])[0]
        proba = None
        if hasattr(ml_model, "predict_proba"):
            probs = ml_model.predict_proba([feats])[0]
            classes = list(ml_model.classes_)
            idx = classes.index(pred)
            proba = float(probs[idx])
        return jsonify({"success": True, "label": str(pred), "confidence": proba})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    print()
    print("  +===============================================+")
    print("  |     Vani-Setu                                 |")
    print(f"  |     Auth: {AUTH_MODE:<36} |")
    if AUTH_MODE == "password":
        print("  |     Login: email + password (SQLite)          |")
    if EMAIL_CONFIGURED:
        print(f"  |     Email: {SMTP_USER[:34]:<34} |")
    elif AUTH_MODE == "otp":
        print("  |     OTP: terminal (set SMTP or AUTH_MODE)     |")
    else:
        print("  |     Guest login enabled                       |")
    print(f"  |     ML model: {'loaded':<32} |" if ml_model is not None else "  |     ML model: not trained yet               |")
    print(f"  |     http://{HOST}:{PORT:<22} |")
    print("  +===============================================+")
    print()
    app.run(host=HOST, port=PORT, debug=False)
