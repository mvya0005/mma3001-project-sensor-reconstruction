#!/usr/bin/env bash
# Run the test suite and write an HTML + JUnit test report to reports/tests/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/tests
python -m pytest tests -v \
  --html=reports/tests/test_report.html --self-contained-html \
  --junitxml=reports/tests/junit.xml "$@"
