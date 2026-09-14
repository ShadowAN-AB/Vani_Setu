#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/Sign-Language-Interpreter-using-Deep-Learning-master/Code"
cd "$CODE"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements_web.txt

export PORT="${PORT:-5001}"
echo "Vani-Setu → http://127.0.0.1:${PORT}"
echo "Auth: ${AUTH_MODE:-password}  (Create account / Sign in; AUTH_MODE=guest for a skip button)"
python server.py
