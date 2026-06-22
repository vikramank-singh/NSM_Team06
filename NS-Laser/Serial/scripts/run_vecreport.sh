#!/bin/bash
# Build with vectorization-report flags and summarize vectorized vs. missed
# loops per source location. Run from NS-Laser/Serial/.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/env_fftw.sh

make vecreport

REPORT=bin/vecreport.txt
echo
echo "==== functions.h: vectorized loops ===="
grep "functions.h" "$REPORT" | grep -E "loop vectorized|LOOP WAS VECTORIZED" || echo "(none found by this compiler's report format)"
echo
echo "==== functions.h: missed-vectorization loops ===="
grep "functions.h" "$REPORT" | grep -iE "not vectorized|loop was not vectorized" || echo "(none found by this compiler's report format)"
echo
echo "Full report: $REPORT"
