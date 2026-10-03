#!/bin/sh
# Prefer the private venv (has typesafe-sdk / system-one-adapter); fall back to system python.
PY="$HOME/.drift-guard/venv/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$(dirname "$0")/hook.py"
