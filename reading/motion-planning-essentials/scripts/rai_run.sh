#!/usr/bin/env bash
# The package's stand-in for the course's container runner: run a Python file from this folder with the
# folder's .venv/bin/python (or python3), in which `robotic` 0.4.1 must already be installed. See README.html#rai.
#   scripts/rai_run.sh modules/motion-planning/stage4_komo_rai.py [args]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
if ! "$PY" -c "import robotic" >/dev/null 2>&1; then
  echo "rai_run: $PY cannot import robotic; install robotic==0.4.1 in that environment (README.html#rai)" >&2
  exit 2
fi
exec "$PY" "$@"
