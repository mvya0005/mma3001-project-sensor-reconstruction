#!/usr/bin/env bash
# Run the unit tests and docstring examples, and write an HTML + JUnit test report to reports/tests/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/tests
python -m pytest -v \
  --html=reports/tests/test_report.html --self-contained-html \
  --junitxml=reports/tests/junit.xml "$@"
