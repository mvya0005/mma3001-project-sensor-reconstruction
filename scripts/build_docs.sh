#!/usr/bin/env bash
# Regenerate the HTML API documentation from the docstrings (Google style) into docs/api/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pdoc --docformat google --math --footer-text "sensorfill - MMA3001 project" -o docs/api sensorfill
echo "HTML documentation written to docs/api/index.html"
