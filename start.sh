#!/usr/bin/env bash
# Public host start. Localhost still uses ./run.sh (127.0.0.1:5001).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/Sign-Language-Interpreter-using-Deep-Learning-master/Code"
export MPLBACKEND="${MPLBACKEND:-Agg}"
exec gunicorn -w 1 --threads 8 -b "0.0.0.0:${PORT:-5001}" --timeout 120 server:app
