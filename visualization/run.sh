#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${SPARSEOCC_VIZ_HOST:-0.0.0.0}"
PORT="${SPARSEOCC_VIZ_PORT:-8000}"

export SPARSEOCC_VIZ_HOST="$HOST"
export SPARSEOCC_VIZ_PORT="$PORT"

exec "$PYTHON_BIN" server.py
