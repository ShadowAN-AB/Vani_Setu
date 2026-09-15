# Deploying Vani-Setu

**Live demo:** [https://vani-setu.onrender.com](https://vani-setu.onrender.com)  
Continue as guest → Start camera. The Free instance may take a minute to wake.

**Localhost still works the same way.** From this folder:

```bash
./run.sh
```

Open **http://127.0.0.1:5001**, create an account or sign in. Deploy settings below do **not** change that unless you set them in `Code/.env`.

---

## What is different on a public URL

| | Localhost | Render (or similar) |
|--|-----------|---------------------|
| Address | `127.0.0.1:5001` | `https://….onrender.com` |
| Camera | allowed on `http://127.0.0.1` | **needs HTTPS** (Render gives this) |
| Bind | `127.0.0.1` | `0.0.0.0` (auto when `RENDER` is set) |
| Login | email + password (SQLite) | **Continue as guest** (`AUTH_MODE=guest`) so a wiped disk does not lock people out |
| Hand model | already on this Mac | downloaded on first boot if missing |
| Letter SVM | local file | shipped in git (`models/sign_rf.joblib`) |

Do not use terminal OTP on a host — nobody can read the server log.

---

## Deploy on Render (HTTPS)

1. Push this repo to GitHub (already: [ShadowAN-AB/Vani_Setu](https://github.com/ShadowAN-AB/Vani_Setu)).
2. Sign in at [https://render.com](https://render.com) with GitHub.
3. **New** → **Blueprint** → pick `Vani_Setu`.  
   Or **New** → **Web Service** → connect the repo, then:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `bash start.sh`
   - **Instance:** Free
4. Environment (Blueprint already sets these):

```
AUTH_MODE=guest
HOST=0.0.0.0
PYTHON_VERSION=3.12.8
MPLBACKEND=Agg
```

Leave `PORT` to Render.

5. After the first deploy finishes, open the `https://….onrender.com` URL.
6. Click **Continue as guest**, then **Start camera**, and allow the webcam.

Free Render apps **sleep after ~15 minutes**. The first request after sleep can take a minute. Camera still works after it wakes.

### Optional: real accounts on the host

Add a Render **Postgres** addon and set `DATABASE_URL` to the URL they give you. Change `AUTH_MODE` to `password`. Also `pip install psycopg2-binary` (or add it to `requirements_web.txt`). `postgres://` is rewritten to `postgresql://`.

---

## Other hosts

`Procfile` + `start.sh` also work on Railway. Set the same env vars. Bind is `0.0.0.0` when `RAILWAY_ENVIRONMENT` or `FLY_APP_NAME` is present.

---

## Auth modes (local or host)

| `AUTH_MODE` | Behaviour |
|-------------|-----------|
| `password` | Email + password. **Local default.** |
| `guest` | Password login plus **Continue as guest**. **Render default.** |
| `otp` | Email / terminal 6-digit code (local only unless SMTP is set). |
| `none` | Skip the login screen. |

Gmail OTP (optional): App password, not your normal Gmail password. See `Code/.env.example`.

---

## Files

- Never commit `.env` or `users.db`.
- `hand_landmarker.task` stays gitignored; the server downloads it if missing.
- `models/sign_rf.joblib` is in git so Spell mode works on a fresh host.

---

## Quick local test (unchanged)

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
AUTH_MODE=password PORT=5001 .venv/bin/python server.py
```

Open http://127.0.0.1:5001 — **Create account**, then **Sign in**.
