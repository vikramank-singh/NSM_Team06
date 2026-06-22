#!/bin/bash
# Standardized Intel Advisor survey + roofline run for the NS-Laser solver.
# Run interactively on a node with the Intel oneAPI toolkit + Advisor on PATH.
#
# Usage: scripts/run_advisor.sh [nx ny nz n_iters]
#   Defaults to a small, fast-turnaround grid (51 51 10 50) — NOT the
#   production grid (201 201 20 1500), which is far too slow to instrument.
set -euo pipefail
cd "$(dirname "$0")/.."

NX="${1:-51}"
NY="${2:-51}"
NZ="${3:-10}"
ITERS="${4:-50}"

source scripts/env_fftw.sh
make CXX=icpx all
BIN=bin/A-NS

PROJECT_DIR=./advi
rm -rf "$PROJECT_DIR"

echo "==== Advisor survey (hotspots + vectorization) ===="
advisor --collect=survey --project-dir="$PROJECT_DIR" --search-dir src:r=./ -- \
	"$BIN" "$NX" "$NY" "$NZ" "$ITERS" --no-output

echo "==== Advisor trip counts + FLOP (for roofline) ===="
advisor --collect=tripcounts --flop --project-dir="$PROJECT_DIR" --search-dir src:r=./ -- \
	"$BIN" "$NX" "$NY" "$NZ" "$ITERS" --no-output

echo
echo "==== Summary ===="
advisor --report=survey --project-dir="$PROJECT_DIR"

echo
echo "Open the full report with: advisor-gui --project-dir=$PROJECT_DIR"
echo "Or export a roofline HTML: advisor --report=roofline --project-dir=$PROJECT_DIR --report-output=roofline.html"
