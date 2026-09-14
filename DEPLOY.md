# Deploying Vani-Setu (no terminal OTP)

The terminal OTP exists only for **local** development when email is not set.
On a public URL you should not read codes from a server log.

## Alternatives (pick one)

### 1. Guest button — recommended for a college showcase

No database. Visitors click **Continue as guest** and go to the camera.

In `Code/.env` (or the host’s environment variables):

```
AUTH_MODE=guest
HOST=0.0.0.0
PORT=5001
```

You can still add Gmail below so some users get a real email OTP.

### 2. Skip login completely

```
AUTH_MODE=none
HOST=0.0.0.0
```

The login page never appears. Simplest public demo. Anyone with the link can use the camera.

### 3. Email OTP (Gmail) — real codes, still no database

This is already built. You do **not** need a database URL. The app still does not store user accounts; it only emails a 6-digit code that lives in RAM for 5 minutes.

1. Turn on 2-Step Verification on the Google account.
2. Create an **App password** (Google Account → Security → App passwords).
3. Set:

```
AUTH_MODE=otp
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=yourname@gmail.com
SMTP_PASS=xxxx xxxx xxxx xxxx
HOST=0.0.0.0
```

Gmail will reject your normal account password. It must be an App password.

Other mail: any SMTP server works (`SMTP_HOST`, `SMTP_PORT`, user, pass). Services like SendGrid / Resend also speak SMTP.

### 4. What we did **not** add (and why)

| Idea | Why not for this project |
|------|--------------------------|
| Database URL / user table | Login is a gate, not an account system |
| Show the OTP on the website | Anyone could log in as anyone |
| SMS (Twilio) | Costs money and needs a phone number |
| Google Sign-In | Needs a Google Cloud OAuth client; can be a later add-on |

## Camera needs HTTPS

Browsers allow the webcam on `http://127.0.0.1` but **not** on a public `http://` URL. Deploy on a host that gives you **https://** (Render, Railway, Fly.io, Cloudflare Tunnel, etc.).

## Bind address

Locally Flask listens on `127.0.0.1`. On a VPS or PaaS you must set `HOST=0.0.0.0` or the site will not accept outside connections. Many platforms also inject `PORT` — leave that to them.

## Files to upload

- The `Vani-Setu` repo
- `Code/models/sign_rf.joblib` (gitignored — upload it or train on the server)
- `Code/hand_landmarker.task` (gitignored — upload it)
- Environment variables, never commit `.env`

## Quick local test of guest mode

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
AUTH_MODE=guest PORT=5001 .venv/bin/python server.py
```

Open http://127.0.0.1:5001 and click **Continue as guest**.
