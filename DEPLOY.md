# Deploying Vani-Setu (no terminal OTP)

The terminal OTP exists only for **local** development when `AUTH_MODE=otp` and email is not set.
On a public URL you should not read codes from a server log.

## Alternatives (pick one)

### 1. Email + password — default, uses a database

Create an account once, then sign in. Passwords are hashed (Werkzeug). No OTP.

**Local:** SQLite file `Code/users.db` is created automatically. Nothing extra to install.

**Hosted:** set `DATABASE_URL` to the Postgres URL from Render / Railway / Neon. Also `pip install psycopg2-binary` (or add it on the host). `postgres://` URLs are rewritten to `postgresql://`.

```
AUTH_MODE=password
HOST=0.0.0.0
PORT=5001
# DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

Do not commit `users.db` or `.env`.

### 2. Guest button — optional for a walk-up showcase

Password login stays, plus **Continue as guest**.

```
AUTH_MODE=guest
HOST=0.0.0.0
PORT=5001
```

You can still add Gmail below so some users get a real email OTP.

### 3. Skip login completely

```
AUTH_MODE=none
HOST=0.0.0.0
```

The login page never appears. Simplest public demo. Anyone with the link can use the camera.

### 4. Email OTP (Gmail) — codes instead of passwords

This does **not** store user accounts; it emails a 6-digit code that lives in RAM for 5 minutes.

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

## Camera needs HTTPS

Browsers allow the webcam on `http://127.0.0.1` but **not** on a public `http://` URL. Deploy on a host that gives you **https://** (Render, Railway, Fly.io, Cloudflare Tunnel, etc.).

## Bind address

Locally Flask listens on `127.0.0.1`. On a VPS or PaaS you must set `HOST=0.0.0.0` or the site will not accept outside connections. Many platforms also inject `PORT` — leave that to them.

## Files to upload

- The `Vani-Setu` repo
- `Code/models/sign_rf.joblib` (gitignored — upload it or train on the server)
- `Code/hand_landmarker.task` (gitignored — upload it)
- Environment variables, never commit `.env`

## Quick local test of password login

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
AUTH_MODE=password PORT=5001 .venv/bin/python server.py
```

Open http://127.0.0.1:5001 — **Create account**, then **Sign in**.
