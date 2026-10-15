#!/usr/bin/env bash
# One-time setup for macOS and Linux: creates .venv and installs everything.
# Needs an internet connection; downloads about 1 GB of packages.
set -e
cd "$(dirname "$0")"

PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.11, 3.12 or 3.13 was not found. Install Python 3.13 from https://www.python.org/downloads/"
  exit 1
fi
echo "Using $PY"

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [ "$(uname -s)" = "Linux" ]; then
  # The default Linux PyTorch wheels include CUDA (several GB); the CPU build is much smaller.
  .venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_sample.py || echo "The sample video could not be downloaded; you can still upload your own video."

echo
echo "Setup finished. Start the app with ./run_app.sh"
