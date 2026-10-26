#!/usr/bin/env bash
# One-command launcher: pulls latest code (if this is a git clone), installs deps, opens the portal.
cd "$(dirname "$0")"
if [ -d .git ] && command -v git >/dev/null; then echo "Checking for updates..."; git pull --ff-only || echo "(update skipped)"; fi
python3 -m pip install -q -r requirements.txt
python3 -m streamlit run app.py
