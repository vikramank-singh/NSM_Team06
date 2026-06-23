#!/bin/bash
# Standardized Intel VTune threading analysis for the NS-Laser OpenMP solver.
# Run interactively on a node with the Intel oneAPI toolkit + VTune on PATH.
#
# Usage: scripts/run_vtune.sh [nx ny nz n_iters]
#   Defaults to a small, fast-turnaround grid (51 51 10 50) — NOT the
#   production grid (201 201 20 1500), which is far too slow to instrument.
#   Thread count is taken from OMP_NUM_THREADS (defaults to all cores if unset).
set -euo pipefail
cd "$(dirname "$0")/.."

NX="${1:-51}"
NY="${2:-51}"
NZ="${3:-10}"
ITERS="${4:-50}"

source scripts/env_fftw.sh
make CXX=icpx all
BIN=bin/A-NS

RESULT_DIR=./vtune_threading
rm -rf "$RESULT_DIR"

echo "==== VTune threading analysis (OMP_NUM_THREADS=${OMP_NUM_THREADS:-<default>}) ===="
vtune -collect threading -r "$RESULT_DIR" -- \
	"$BIN" "$NX" "$NY" "$NZ" "$ITERS" --no-output

echo
echo "==== Summary ===="
vtune -report summary -r "$RESULT_DIR"

echo
echo "Open the full report with: vtune-gui $RESULT_DIR"
