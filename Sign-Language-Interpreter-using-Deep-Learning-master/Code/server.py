"""
Vani-Setu backend: password auth, translation, landmark collection, and ML classify.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import data as dataset
from features import landmarks_to_features
from suggest import Suggester
from models_user import User, db, init_db
from signs import OFFLINE_HI, dictionary

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
# Phrase and ISL run as JavaScript finger-pose rules in the browser; the server
# only hosts the letter SVM. Earlier builds carried a NumpyMLP scaffold for
# phrase/ISL networks that were never trained; it was removed in Phase 1.
HAND_TASK_PATH = os.path.join(APP_DIR, "hand_landmarker.task")
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
TARGET_SAMPLES_PER_LABEL = 25

# Only these files are served from the app folder. Everything else in it
# (users.db, .env, datasets, source code) must never be downloadable.
PUBLIC_FILES = {"index.html", "hand_landmarker.task"}


def _on_paas():
    return any(
        os.environ.get(k)
        for k in ("RENDER", "RAILWAY_ENVIRONMENT", "FLY_APP_NAME", "K_SERVICE")
    )


PORT = int(os.environ.get("PORT", "5001"))
HOST = os.environ.get("HOST", "0.0.0.0" if _on_paas() else "127.0.0.1").strip() or (
    "0.0.0.0" if _on_paas() else "127.0.0.1"
)

# Recording samples and retraining change the shared model, so they are on for
# localhost and off on a public host unless ALLOW_TRAINING=1 is set explicitly.
_allow = os.environ.get("ALLOW_TRAINING", "").strip().lower()
TRAINING_ENABLED = (_allow in ("1", "true", "yes")) if _allow else not _on_paas()

# The app translates the sentence into Hindi only; English is the source language.
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}

# Optional sentence polisher via a locally-running Ollama. Off by default so the
# Render deploy and anyone without Ollama is unaffected; set OLLAMA_URL to opt in.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct").strip()
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "15"))
POLISH_MAX_INPUT = 240
polish_cache = {}
polish_lock = Lock()

translation_cache = {}
cache_lock = Lock()
csv_lock = Lock()
train_lock = Lock()
# Protects (ml_model, ml_info) swaps so /api/classify never sees a non-None
# model paired with a reset info dict (feature_version would silently default to 1).
model_lock = Lock()
MAX_CACHE_SIZE = 500
ml_model = None       # the classifier (has predict_proba and classes_)
ml_info = {}          # feature version, threshold and headline metrics saved by the trainer
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
    """Load the trainer's bundle; an old bare pipeline still works with version-1 features.
    Writes to (ml_model, ml_info) are serialised through model_lock so /api/classify
    always reads a consistent pair."""
    global ml_model, ml_info
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH
    if not JOBLIB_OK or not os.path.exists(path):
        ml_model, ml_info = None, {}
        return
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            new_model = obj["model"]
            new_info = {
                "path": os.path.relpath(path, APP_DIR),
                "feature_version": int(obj.get("feature_version", 1)),
                "threshold": float(obj.get("threshold", DEFAULT_THRESHOLD)),
                "metrics": obj.get("metrics", {}),
            }
        else:
            new_model = obj
            new_info = {"path": os.path.relpath(path, APP_DIR), "feature_version": 1,
                        "threshold": DEFAULT_THRESHOLD, "metrics": {}}
        # Swap atomically from the reader's point of view (see /api/classify snapshot).
        with model_lock:
            ml_model = new_model
            ml_info = new_info
        print(f"  [OK] Loaded classifier: {ml_info['path']} (features v{ml_info['feature_version']})")
        suggester.set_confusion(obj.get("confusion") if isinstance(obj, dict) else None)
    except Exception as e:
        with model_lock:
            ml_model, ml_info = None, {}
        print(f"  [WARN] Could not load model: {e}")


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


# static_folder=None: do NOT let Flask auto-serve this directory. The directory
# contains users.db, .env, source and datasets. All browser-reachable files are
# served through the explicit index + PUBLIC_FILES routes below.
app = Flask(__name__, static_folder=None)
CORS(app)
ensure_hand_landmarker()
init_db(app)
load_ml_model()


def _bad(message, code=400, **extra):
    """Uniform error-response shape so the frontend's await res.json() never
    receives Flask's default HTML page."""
    body = {"success": False, "message": str(message)}
    body.update(extra)
    return jsonify(body), code


@app.errorhandler(404)
def _handle_404(e):
    return _bad("Not found", 404)


@app.errorhandler(405)
def _handle_405(e):
    return _bad("Method not allowed", 405)


@app.errorhandler(500)
def _handle_500(e):
    # Do not leak internals to the client; the server log keeps the traceback.
    return _bad("Server error", 500)


@app.route("/")
def index():
    # The whole app is one HTML file; browsers caching an older copy means a
    # fix pushed on the server never reaches the user. Tell the browser not to.
    resp = send_from_directory(APP_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


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
    # Preserved so the frontend's one-fetch probe still works; auth is now
    # email + password only, no OTP, no guest, no SMTP.
    return jsonify({"success": True, "mode": "password", "password_enabled": True})


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
    try:
        db.session.commit()
    except Exception:
        # Second thread wins the race; the first already created the account.
        db.session.rollback()
        return _bad("That email is already registered. Sign in instead.", 409)
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


@app.route("/api/translate", methods=["POST"])
def translate_text():
    data = request.get_json(silent=True)
    if not data or "text" not in data or "target" not in data:
        return jsonify({"success": False, "message": "text and target language are required"}), 400

    text = str(data.get("text") or "").strip()
    target = str(data.get("target") or "").strip()
    source = str(data.get("source") or "en").strip()

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


def _polish_available():
    return bool(OLLAMA_URL)


def _ollama_generate(prompt, num_predict=80, temperature=0.2):
    """POST a one-shot prompt to Ollama and return the model's reply string.
    Caller is responsible for building the prompt and handling URLError."""
    url = OLLAMA_URL.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    reply = (body.get("response") or "").strip()
    # Keep the first non-empty line, strip surrounding quotes - small models
    # often prefix their output with a quote or a filler line.
    for line in reply.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if line:
            return line
    return ""


POLISH_PROMPT_PLAIN = (
    "Rewrite the following sign-language gloss as one natural, grammatical "
    "English sentence. Keep the meaning. Do not add new facts. Reply with "
    "only the rewritten sentence, no quotes, no preamble.\n\nGloss: "
)
POLISH_PROMPT_CTX = (
    "You are polishing sign-language glosses into natural English. The user "
    "has already said the following, which you should treat as background "
    "context (do not repeat it):\n\n{ctx}\n\nNow rewrite THIS new gloss as "
    "one natural English sentence that reads coherently after the context. "
    "Keep the meaning, add no new facts, reply with only the sentence, no "
    "quotes, no preamble.\n\nGloss: {text}"
)


@app.route("/api/polish", methods=["POST"])
def polish():
    if not _polish_available():
        return _bad("Sentence polishing is disabled on this server.", 503,
                    enabled=False)
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    context = str(data.get("context") or "").strip()
    if not text:
        return _bad("text is required", 400)
    if len(text) > POLISH_MAX_INPUT:
        return _bad(f"text too long (max {POLISH_MAX_INPUT} chars)", 400)
    if len(context) > POLISH_MAX_INPUT * 2:
        context = context[-POLISH_MAX_INPUT * 2:]

    cache_key = (context + "||" + text) if context else text
    with polish_lock:
        cached = polish_cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "polished": cached, "source": "cache"})

    prompt = (POLISH_PROMPT_CTX.format(ctx=context, text=text)
              if context else POLISH_PROMPT_PLAIN + text)
    try:
        polished = _ollama_generate(prompt, num_predict=80, temperature=0.2)
    except urllib.error.URLError as e:
        return _bad(f"Ollama unreachable: {e.reason}", 503)
    except Exception as e:
        print(f"  [ERROR] polish failed: {e}")
        return _bad("polishing failed", 500)

    if not polished:
        return _bad("empty response from model", 502)

    with polish_lock:
        if len(polish_cache) >= 200:
            for k in list(polish_cache.keys())[:50]:
                del polish_cache[k]
        polish_cache[cache_key] = polished

    return jsonify({"success": True, "polished": polished,
                    "source": "ollama", "model": OLLAMA_MODEL})


CORRECT_WORD_PROMPT_PLAIN = (
    "A sign-language speller produced these letters, but one or two letters "
    "may be wrong because similar hand shapes get confused (M/N, V/K, U/R, "
    "D/F). Reply with the single English word they most likely meant - "
    "lowercase, no punctuation, no explanation. If the letters already form "
    "a word, reply with that word unchanged.\n\nLetters: {text}"
)
CORRECT_WORD_PROMPT_CTX = (
    "A sign-language speller produced these letters, but one or two letters "
    "may be wrong because similar hand shapes get confused (M/N, V/K, U/R, "
    "D/F). Here is what they have already said so far:\n\n{ctx}\n\nGiven "
    "that context, reply with the single English word they most likely "
    "meant - lowercase, no punctuation, no explanation.\n\nLetters: {text}"
)


@app.route("/api/correct-word", methods=["POST"])
def correct_word():
    """LLM-based spelling fix for a single word. Stronger than the dictionary
    suggester at rare words and names; off unless OLLAMA_URL is set."""
    if not _polish_available():
        return _bad("Word correction is disabled on this server.", 503,
                    enabled=False)
    data = request.get_json(silent=True) or {}
    letters = str(data.get("letters") or "").strip()
    context = str(data.get("context") or "").strip()
    if not letters:
        return _bad("letters is required", 400)
    if len(letters) > 40:
        return _bad("letters too long (max 40)", 400)
    if len(context) > POLISH_MAX_INPUT * 2:
        context = context[-POLISH_MAX_INPUT * 2:]

    cache_key = "cw:" + (context + "||" + letters.lower() if context
                         else letters.lower())
    with polish_lock:
        cached = polish_cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "word": cached, "source": "cache"})

    prompt = (CORRECT_WORD_PROMPT_CTX.format(ctx=context, text=letters)
              if context else CORRECT_WORD_PROMPT_PLAIN.format(text=letters))
    try:
        word = _ollama_generate(prompt, num_predict=16, temperature=0.1)
    except urllib.error.URLError as e:
        return _bad(f"Ollama unreachable: {e.reason}", 503)
    except Exception as e:
        print(f"  [ERROR] correct-word failed: {e}")
        return _bad("word correction failed", 500)

    # Keep just the first word token - strip punctuation and any trailing
    # commentary the model might slip in.
    word = re.split(r"[\s,.;:!?]+", word.strip(), maxsplit=1)[0]
    word = re.sub(r"[^A-Za-z'-]+", "", word).lower()
    if not word:
        return _bad("empty response from model", 502)

    with polish_lock:
        if len(polish_cache) >= 200:
            for k in list(polish_cache.keys())[:50]:
                del polish_cache[k]
        polish_cache[cache_key] = word

    return jsonify({"success": True, "word": word,
                    "source": "ollama", "model": OLLAMA_MODEL})


def _load_sign_shapes():
    path = os.path.join(DATA_DIR, "sign_shapes.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("shapes", {})
    except (OSError, ValueError):
        return {}


SIGN_SHAPES = _load_sign_shapes()


@app.route("/api/dictionary", methods=["GET"])
def get_dictionary():
    """How to make every letter and sign, in English and Hindi, with hand drawings."""
    return jsonify({"success": True, "dictionary": dictionary(SIGN_SHAPES)})


@app.route("/api/health", methods=["GET"])
def health():
    checks = {
        "server": True,
        "model_loaded": ml_model is not None,
        "dataset_letters": dataset.counts("letter"),
        "hand_model_file": os.path.exists(os.path.join(APP_DIR, "hand_landmarker.task")),
        "translator": TRANSLATOR_AVAILABLE,
        "polish_enabled": _polish_available(),
        "polish_model": OLLAMA_MODEL if _polish_available() else None,
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
        "target_per_label": TARGET_SAMPLES_PER_LABEL,
        "training_enabled": TRAINING_ENABLED,
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

    # Phase 1 removed phrase/ISL recording: those modes run as JavaScript
    # finger-pose rules and have no server-side model to train. Only "letter"
    # samples are accepted.
    kind = str(data.get("kind") or "letter").strip().lower()
    if kind != "letter":
        return _bad("Phrase and ISL are rule-based in this build; only letter samples are recorded.", 422)

    try:
        with csv_lock:
            hands = dataset.parse_hands(data)
            row = dataset.append_recording(label, hands, kind=kind,
                                            signer=data.get("signer", ""), split=data.get("split", ""))
            count = dataset.counts(kind)
            label_count = dataset.counts(kind, label)
    except ValueError as e:
        return _bad(str(e), 400)

    # Report the split actually written, not a separately-recomputed one.
    # (data.append_recording strips+lowercases; the old inline recompute did not.)
    return jsonify({
        "success": True,
        "samples": count,
        "label_count": label_count,
        "label": label,
        "kind": kind,
        "hands": len(hands),
        "split": row.get("split") or "train",
        "target": TARGET_SAMPLES_PER_LABEL,
        "path": os.path.relpath(dataset.RECORDINGS_CSV, APP_DIR),
    })


@app.route("/api/train", methods=["POST"])
def train_model():
    """Retrain the letter SVM from the browser and hot-reload, so a demo never
    needs a terminal. Phrase and ISL are JavaScript finger-pose rules and have
    no server-side training step."""
    if not TRAINING_ENABLED:
        return _training_disabled()
    body = request.get_json(silent=True) or {}
    kind = str(body.get("kind") or "letter").lower()
    if kind != "letter":
        return _bad("Phrase and ISL are rule-based in this build; nothing to train.", 422)
    if not train_lock.acquire(blocking=False):
        return jsonify({"success": False, "message": "Training already in progress"}), 409

    try:
        cmd = [sys.executable, os.path.join(APP_DIR, "train_classifier.py")]
        report = os.path.join(APP_DIR, "reports", "metrics.json")
        proc = subprocess.run(cmd, cwd=APP_DIR, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            lines = [l for l in out.strip().splitlines() if l.strip()]
            message = lines[-1] if lines else "Training failed"
            # Precondition failures are 422 (unprocessable), not 500 (server crash):
            # "no samples yet" is a perfectly ordinary state on a fresh clone.
            if ("samples yet" in out) or ("No letter samples found" in out):
                return _bad("No letter samples yet. Record some first.", 422)
            if "Need at least two" in out:
                return _bad(message, 422)
            return _bad(message, 500)

        load_ml_model()
        metrics = {}
        if os.path.exists(report):
            with open(report, encoding="utf-8") as f:
                metrics = json.load(f)
        test = metrics.get("test") or {}
        return jsonify({
            "success": True,
            "kind": "letter",
            "accuracy": metrics.get("accuracy", test.get("accuracy")),
            "n_samples": metrics.get("n_samples", sum((metrics.get("split") or {}).get(k, 0) for k in ("train", "validation", "test"))),
            "n_classes": metrics.get("n_classes", len(metrics.get("classes", []))),
            "model_loaded": ml_model is not None,
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
    """SVM over the first hand's 88-feature vector. Phrase and ISL are handled
    by the JavaScript rule classifiers in the browser and never reach here."""
    data = request.get_json(silent=True)
    if not data or ("landmarks" not in data and "hands" not in data):
        return jsonify({"success": False, "message": "landmarks required"}), 400
    mode = str(data.get("mode") or "letter").lower()
    if mode != "letter":
        return _bad("Phrase and ISL run as JavaScript rules; no server classify.", 422)

    start = time.perf_counter()
    try:
        hands = dataset.parse_hands(data)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    # Grab a consistent snapshot of (model, feature_version, threshold) under
    # model_lock. Without this, /api/train can rebind ml_model and ml_info
    # separately while a classify request is mid-flight, pairing a new
    # model with the OLD/empty info dict (feature_version defaults to 1 ->
    # a 63-length vector into an 88-feature pipeline -> sklearn ValueError).
    with model_lock:
        model = ml_model
        fv = int((ml_info or {}).get("feature_version", 1))
        threshold = float((ml_info or {}).get("threshold", DEFAULT_THRESHOLD))
    if model is None:
        return _bad("No trained model yet. Run train_classifier.py", 404)
    feats = landmarks_to_features(hands[0][0], fv)

    try:
        # The label always comes from the highest probability, so the label and
        # the confidence shown in the app can never disagree.
        probs = model.predict_proba([feats])[0]
        classes = [str(c) for c in model.classes_]
        order = sorted(range(len(classes)), key=lambda i: -probs[i])
        top = [{"label": classes[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]]
        return jsonify({
            "success": True,
            "mode": "letter",
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
    print("  |     Login: email + password (SQLite)          |")
    print(f"  |     ML model: {'loaded':<32} |" if ml_model is not None else "  |     ML model: not trained yet               |")
    print(f"  |     http://{HOST}:{PORT:<22} |")
    print("  +===============================================+")
    print()
    app.run(host=HOST, port=PORT, debug=False)
