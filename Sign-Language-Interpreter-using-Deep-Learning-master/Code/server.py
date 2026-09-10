"""
Vani-Setu OTP Authentication + Translation Backend
====================================================
Flask server providing:
  - POST /api/send-otp   → Generate & email a 6-digit OTP
  - POST /api/verify-otp → Verify OTP
  - POST /api/translate   → Translate text to target language
  - GET  /               → Serve index.html

OTP is stored in-memory with 5-minute expiry.
Email sending uses smtplib (Gmail App Password).
Translation uses deep-translator (Google Translate free API).
"""

import os
import random
import string
import time
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Lock

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Try to import deep_translator
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("  [WARN] deep-translator not installed. Run: pip install deep-translator")

# ==============================================================
# CONFIG
# ==============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Email config — Gmail App Password
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "shrivastavakartikeya82@gmail.com")
SMTP_PASS = os.environ.get("SMTP_PASS", "pcczcuxlgvhpqelq")
SENDER_NAME = "Vani-Setu"

OTP_LENGTH = 6
OTP_EXPIRY_SECONDS = 300  # 5 minutes
MAX_ATTEMPTS = 5
RATE_LIMIT_SECONDS = 60   # Min time between OTP sends to same email

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "ar": "Arabic",
    "bn": "Bengali",
    "ta": "Tamil",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh-CN": "Chinese (Simplified)",
    "ur": "Urdu",
    "mr": "Marathi",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "it": "Italian",
    "tr": "Turkish",
    "th": "Thai",
    "vi": "Vietnamese",
}

# ==============================================================
# OTP STORE (in-memory)
# ==============================================================
otp_store = {}  # email -> {otp, created_at, attempts}
store_lock = Lock()

# Translation cache to avoid repeated API calls
translation_cache = {}
cache_lock = Lock()
MAX_CACHE_SIZE = 500


def generate_otp():
    return ''.join(random.choices(string.digits, k=OTP_LENGTH))


def cleanup_expired():
    """Remove expired OTP entries."""
    now = time.time()
    expired = [k for k, v in otp_store.items() if now - v['created_at'] > OTP_EXPIRY_SECONDS]
    for k in expired:
        del otp_store[k]


def send_email_otp(email, otp):
    """Send OTP via email. Returns True on success."""
    if not SMTP_USER or not SMTP_PASS:
        print(f"\n{'='*50}")
        print(f"  OTP for {email}: {otp}")
        print(f"  (Email not configured - showing in console)")
        print(f"{'='*50}\n")
        return True  # Treat as success for demo

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🤟 Vani-Setu Login OTP: {otp}"
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = email

        html = f"""
        <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:480px;margin:0 auto;
                    background:#0a0a14;color:#e8e8f0;padding:40px;border-radius:16px;
                    border:1px solid rgba(168,85,247,0.3);">
            <div style="text-align:center;margin-bottom:24px;">
                <span style="font-size:36px;">🤟</span>
                <h1 style="background:linear-gradient(135deg,#a855f7,#06b6d4);
                           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                           font-size:28px;margin:8px 0 0;">Vani-Setu</h1>
                <p style="color:#888;font-size:13px;letter-spacing:2px;">SIGN LANGUAGE TRANSLATOR</p>
            </div>
            <div style="background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);
                        border-radius:12px;padding:24px;text-align:center;margin:20px 0;">
                <p style="color:#aaa;font-size:14px;margin:0 0 12px;">Your verification code:</p>
                <div style="font-size:36px;font-weight:900;letter-spacing:12px;color:#a855f7;">
                    {otp}
                </div>
                <p style="color:#666;font-size:12px;margin:12px 0 0;">Expires in 5 minutes</p>
            </div>
            <p style="color:#666;font-size:12px;text-align:center;">
                If you didn't request this code, please ignore this email.
            </p>
        </div>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, email, msg.as_string())

        print(f"  [OK] OTP email sent to {email}")
        print(f"  OTP for {email}: {otp}")
        return True

    except Exception as e:
        print(f"  [ERROR] Email send failed: {e}")
        print(f"  OTP for {email}: {otp} (showing in console as fallback)")
        return True  # Still return True so user can use console OTP


# ==============================================================
# FLASK APP
# ==============================================================
app = Flask(__name__, static_folder=APP_DIR)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(APP_DIR, filename)


@app.route("/api/send-otp", methods=["POST"])
def send_otp():
    data = request.get_json(silent=True)
    if not data or "email" not in data:
        return jsonify({"success": False, "message": "Email is required"}), 400

    email = data["email"].strip().lower()

    # Basic email validation
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return jsonify({"success": False, "message": "Invalid email format"}), 400

    with store_lock:
        cleanup_expired()

        # Rate limiting
        if email in otp_store:
            elapsed = time.time() - otp_store[email]['created_at']
            if elapsed < RATE_LIMIT_SECONDS:
                remaining = int(RATE_LIMIT_SECONDS - elapsed)
                return jsonify({
                    "success": False,
                    "message": f"Please wait {remaining}s before requesting a new OTP"
                }), 429

        otp = generate_otp()
        otp_store[email] = {
            "otp": otp,
            "created_at": time.time(),
            "attempts": 0
        }
        print(f"  [OTP] {email} -> {otp}", flush=True)

    # Send email (or print to console)
    send_email_otp(email, otp)

    return jsonify({
        "success": True,
        "message": "OTP sent successfully! Check your email."
    })


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

        # Check expiry
        if time.time() - entry["created_at"] > OTP_EXPIRY_SECONDS:
            del otp_store[email]
            return jsonify({"success": False, "message": "OTP has expired. Request a new one."}), 400

        # Check attempts
        if entry["attempts"] >= MAX_ATTEMPTS:
            del otp_store[email]
            return jsonify({"success": False, "message": "Too many attempts. Request a new OTP."}), 400

        # Verify
        if user_otp == entry["otp"]:
            del otp_store[email]  # One-time use
            return jsonify({"success": True, "message": "Login successful! Welcome to Vani-Setu."})
        else:
            entry["attempts"] += 1
            remaining = MAX_ATTEMPTS - entry["attempts"]
            return jsonify({
                "success": False,
                "message": f"Incorrect OTP. {remaining} attempts remaining."
            }), 400


@app.route("/api/translate", methods=["POST"])
def translate_text():
    """Translate text from English to target language."""
    data = request.get_json(silent=True)
    if not data or "text" not in data or "target" not in data:
        return jsonify({"success": False, "message": "text and target language are required"}), 400

    text = data["text"].strip()
    target = data["target"].strip()
    source = data.get("source", "en").strip()

    if not text:
        return jsonify({"success": True, "translated": ""})

    # If target is same as source, return as-is
    if target == source:
        return jsonify({"success": True, "translated": text})

    if target not in SUPPORTED_LANGUAGES:
        return jsonify({"success": False, "message": f"Unsupported language: {target}"}), 400

    if not TRANSLATOR_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Translation service not available. Install: pip install deep-translator"
        }), 500

    # Check cache
    cache_key = f"{source}:{target}:{text}"
    with cache_lock:
        if cache_key in translation_cache:
            return jsonify({"success": True, "translated": translation_cache[cache_key]})

    try:
        translator = GoogleTranslator(source=source, target=target)
        translated = translator.translate(text)

        # Cache result
        with cache_lock:
            if len(translation_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entries (simple FIFO)
                keys = list(translation_cache.keys())
                for k in keys[:100]:
                    del translation_cache[k]
            translation_cache[cache_key] = translated

        return jsonify({"success": True, "translated": translated})

    except Exception as e:
        print(f"  [ERROR] Translation failed: {e}")
        return jsonify({"success": False, "message": f"Translation error: {str(e)}"}), 500


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Return list of supported languages."""
    return jsonify({"success": True, "languages": SUPPORTED_LANGUAGES})


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    print()
    print("  +===============================================+")
    print("  |     Vani-Setu Authentication + Translation    |")
    print("  |     ----------------------------------------- |")
    if SMTP_USER:
        print(f"  |     Email: {SMTP_USER:<34} |")
    else:
        print("  |     No email configured (console OTP)        |")
    if TRANSLATOR_AVAILABLE:
        print(f"  |     Translation: {len(SUPPORTED_LANGUAGES)} languages available     |")
    else:
        print("  |     Translation: NOT AVAILABLE (install pkg)  |")
    print("  |     http://localhost:5000                     |")
    print("  +===============================================+")
    print()

    app.run(host="0.0.0.0", port=5000, debug=True)
