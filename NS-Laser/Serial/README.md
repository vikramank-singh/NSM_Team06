Coupled Navier-Stokes - Paraxial Equation Solver

Incompressible Navier Stokes with Boussinesq approximations

References
[1] Parametric numerical study of laser–fluid thermal interaction in a closed duct, International Journal of Heat and Mass Transfer 265, 128808
https://doi.org/10.1016/j.ijheatmasstransfer.2026.128808

## Build

Requires FFTW3. If it isn't on the default include/lib path:

    FFTW_ROOT=/path/to/fftw3 make

| target          | binary                  | purpose                              |
|-----------------|-------------------------|---------------------------------------|
| `make`          | `bin/A-NS`              | release build (`-O3 -march=native`)   |
| `make debug`    | `bin/A-NS_debug`        | `-O0 -g` for correctness debugging    |
| `make vecreport`| `bin/A-NS_vecreport`    | `-O3` + compiler vectorization report |

`CXX` defaults to `icpx` if present on `PATH`, else `g++`. Override explicitly with `make CXX=g++` / `make CXX=icpx`.

## Running

    ./bin/A-NS                          # production grid: 201x201x20, 1500 iters, writes T/u/v/w/I/p.txt
    ./bin/A-NS 51 51 10 50 --no-output  # small grid, fast, no file output — for profiling/Advisor

`plot.py` expects the production-grid output files, so leave the size args off for a real run.

## Serial optimization workflow

This solver is tuned in three repeatable steps. Run from this directory (`NS-Laser/Serial/`).

### 1. Vectorization report

    scripts/run_vecreport.sh

Builds with the vectorizer's diagnostic flags (`-fopt-info-vec-*` for gcc, `-qopt-report` for icpx) and prints which loops in `functions.h` vectorized vs. which didn't, with the compiler's stated reason. Use this first — it's instant and catches missed-vectorization regressions (aliasing, non-restrict pointers, calls inside hot loops) before reaching for Advisor.

### 2. Intel Advisor survey + roofline

Needs the Intel oneAPI toolkit (`advisor`, `icpx`) on `PATH` — source `~/intel/oneapi/setvars.sh` first, or submit via SLURM:

    scripts/run_advisor.sh [nx ny nz n_iters]      # interactive, defaults to 51 51 10 50
    sbatch scripts/advisor.slurm [nx ny nz n_iters] # batch wrapper, same args

This builds with `icpx`, runs an Advisor `survey` pass (hotspots + vectorization status) and a `tripcounts --flop` pass (for the roofline model), then prints the survey summary. Always use a small grid here — Advisor's instrumentation overhead makes the production grid impractical. Open the full GUI report with:

    advisor-gui --project-dir=./advi

or export a standalone roofline chart:

    advisor --report=roofline --project-dir=./advi --report-output=roofline.html

### 3. Apply and re-check

Fix what Advisor/vec-report flag (non-restrict pointer aliasing, calls in hot loops, `pow()`/division in inner loops, misaligned arrays), then re-run step 1 to confirm the loop now vectorizes, and step 2 to confirm it moved on the roofline before re-profiling the next hotspot.

`scripts/env_fftw.sh` centralizes the FFTW environment (`FFTW_ROOT`, `LD_LIBRARY_PATH`, etc.) — both the Makefile and the Advisor scripts source it, so set `FFTW_ROOT` once if the default spack path doesn't apply to your environment.

## Notes on the current vectorization-friendly refactor

`functions.h`/`A-NS.cpp` already incorporate the fixes most commonly surfaced by the workflow above:
- `__restrict__`-qualified pointer parameters on every hot function (`prediction`, `correction`, `T_compute`, `divergence`, `poisson_solver_Neumann_bc`, `A_compute_ss`) so the compiler doesn't have to assume aliasing between field arrays.
- `advection`/`diffusion` are `inline` with `restrict` args, so they fuse into the caller's loop body instead of being emitted as an out-of-line scalar call per grid point.
- `pow(x,2)` replaced with `x*x`, and per-iteration divisions replaced with precomputed reciprocals hoisted outside the hot loops.
- Field arrays are 64-byte aligned (`alloc_field`, `posix_memalign`) instead of plain `new[]`.
- Grid size and iteration count are CLI-configurable (`./A-NS nx ny nz n_iters [--no-output]`) so Advisor/vec-report runs use a fast small case without a duplicated source file.

## Benchmark results

Wall-clock comparison of the pre-optimization revision (`e1b99d6`) vs. this branch's
vectorization rewrite (`fa84b68`), production grid (201x201x20, 1500 iters), single node,
SLURM `debug` partition (`bench/bench_orig.slurm`, `bench/bench_opt.slurm`):

| Revision      | Wall time (`real`) | `user` | `sys` |
|---------------|---------------------|--------|-------|
| Original      | 1534.497s (~25.6 min) | 1526.594s | 3.442s |
| Optimized     | 617.749s (~10.3 min)  | 614.026s  | 2.476s |

**~2.48x speedup (~60% reduction in wall-clock time).**
