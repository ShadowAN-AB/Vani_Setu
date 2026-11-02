# Deploying Vani-Setu

**Live demo:** [https://vani-setu.onrender.com](https://vani-setu.onrender.com)
Create an account → Sign in → Start camera. The Free instance may take a minute to wake.

**Localhost works the same way:**

```bash
./run.sh
```

Open **http://127.0.0.1:5001**, create an account or sign in.

---

## What is different on a public URL

| | Localhost | Render (or similar) |
|--|-----------|---------------------|
| Address | `127.0.0.1:5001` | `https://….onrender.com` |
| Camera | allowed on `http://127.0.0.1` | **needs HTTPS** (Render gives this) |
| Bind | `127.0.0.1` | `0.0.0.0` (auto when `RENDER` is set) |
| Login | email + password (SQLite) | email + password (SQLite, or Postgres via `DATABASE_URL`) |
| Hand model | already on this Mac | downloaded on first boot if missing |
| Letter SVM | local file | shipped in git (`models/sign_svm.joblib`) |

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
HOST=0.0.0.0
PYTHON_VERSION=3.13
MPLBACKEND=Agg
```

Leave `PORT` to Render.

5. After the first deploy finishes, open the `https://….onrender.com` URL.
6. Create an account, sign in, click **Start camera**, allow the webcam.

Free Render apps **sleep after ~15 minutes**. The first request after sleep can take a minute. Camera still works after it wakes.

### Optional: Postgres for persistent accounts

Add a Render **Postgres** addon and set `DATABASE_URL` to the URL they give you. Also `pip install psycopg2-binary` (or add it to `requirements_web.txt`). `postgres://` is rewritten to `postgresql://`.

---

## Other hosts

`Procfile` + `start.sh` also work on Railway. Set the same env vars. Bind is `0.0.0.0` when `RAILWAY_ENVIRONMENT` or `FLY_APP_NAME` is present.

---

## Files

- Never commit `.env` or `users.db`.
- `hand_landmarker.task` is tracked in git for a fast cold start.
- `models/sign_svm.joblib` is in git so Spell mode works on a fresh host.
- Recording and retraining are off on public hosts. Set `ALLOW_TRAINING=1` only if you really want visitors to change the model.

---

## Quick local test

```bash
cd Sign-Language-Interpreter-using-Deep-Learning-master/Code
PORT=5001 .venv/bin/python server.py
```

Open http://127.0.0.1:5001 — **Create account**, then **Sign in**.
