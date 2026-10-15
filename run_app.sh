#!/usr/bin/env bash
# Starts the Football Video Analysis app at http://localhost:8501
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "The virtual environment is missing. Run ./setup.sh first."
  exit 1
fi
if [ -z "$FOOTBALL_NO_BROWSER" ]; then
  (sleep 6 && .venv/bin/python -m webbrowser -t http://localhost:8501 >/dev/null 2>&1) &
fi
exec .venv/bin/python -m streamlit run app.py --server.port 8501 "$@"
