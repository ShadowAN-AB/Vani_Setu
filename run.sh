#!/usr/bin/env bash
# One command to boot Vani-Setu locally. First run: creates venv, installs deps,
# optionally starts a local Ollama for the Polish button. Later runs: just starts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/Sign-Language-Interpreter-using-Deep-Learning-master/Code"
cd "$CODE"

if [[ ! -d .venv ]]; then
  echo "First-time setup: creating .venv ..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements_web.txt

OLLAMA_PID=""
cleanup() {
  if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "Stopping Ollama (pid $OLLAMA_PID)"
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Auto-enable the Polish button when Ollama is installed and OLLAMA_URL isn't
# already set by the caller. Starts ollama in the background, pulls the model
# on first use, and exports OLLAMA_URL so the server picks it up.
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b-instruct}"
if [[ -z "${OLLAMA_URL:-}" ]] && command -v ollama >/dev/null 2>&1; then
  if ! curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama in the background ..."
    ollama serve >/tmp/vanisetu-ollama.log 2>&1 &
    OLLAMA_PID=$!
    for _ in {1..20}; do
      curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
      sleep 0.5
    done
  fi
  if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$OLLAMA_MODEL"; then
    echo "Pulling Ollama model $OLLAMA_MODEL (first run only, ~2 GB) ..."
    ollama pull "$OLLAMA_MODEL"
  fi
  export OLLAMA_URL="http://127.0.0.1:11434"
  export OLLAMA_MODEL
  echo "Polish button: ON  (model=$OLLAMA_MODEL)"
else
  echo "Polish button: OFF (install Ollama + run again, or set OLLAMA_URL)"
fi

export PORT="${PORT:-5001}"
echo "Vani-Setu -> http://127.0.0.1:${PORT}"
echo "Login: email + password (stored in Code/users.db)"
python server.py
