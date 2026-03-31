#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"

if [ ! -x "${VENV_PY}" ]; then
  echo "Missing virtual environment at .venv/." >&2
  echo "Create it first:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

exec "${VENV_PY}" -m streamlit run "${ROOT_DIR}/src/dashboard.py"
