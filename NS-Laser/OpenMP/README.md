Coupled Navier-Stokes - Paraxial Equation Solver (OpenMP port)

OpenMP parallelization of `NS-Laser/Serial` (vectorized + direct-DCT-Poisson
revision, see `../Serial/README.md` for that history). Same physics, same
output format — this directory differs only in thread-level parallelism.

References
[1] Parametric numerical study of laser–fluid thermal interaction in a closed duct, International Journal of Heat and Mass Transfer 265, 128808
https://doi.org/10.1016/j.ijheatmasstransfer.2026.128808

## Build

Requires FFTW3. If it isn't on the default include/lib path:

    FFTW_ROOT=/path/to/fftw3 make

| target          | binary                  | purpose                              |
|-----------------|-------------------------|---------------------------------------|
| `make`          | `bin/A-NS`              | release build (`-O3 -march=native`, OpenMP-enabled) |
| `make debug`    | `bin/A-NS_debug`        | `-O0 -g` for correctness debugging    |
| `make vecreport`| `bin/A-NS_vecreport`    | `-O3` + compiler vectorization report |

`CXX` defaults to `icpx` if present on `PATH`, else `g++`. Override explicitly with `make CXX=g++` / `make CXX=icpx`.

## Running

Thread count is controlled at runtime via `OMP_NUM_THREADS`, not at build time:

    OMP_NUM_THREADS=48 ./bin/A-NS                          # production grid: 201x201x20, 1500 iters
    OMP_NUM_THREADS=4 ./bin/A-NS 51 51 10 50 --no-output   # small grid, fast, no file output

## What's parallelized

The per-grid-point kernels called from `ns_compute()` (`prediction`, `correction`,
`divergence`, `T_compute`, `Neumann_bc`, `Dirichlet_bc`/`Dirichlet_bc3`, the DCT
Poisson solver's pack/unpack loops) are parallelized with `#pragma omp for`.
Critically, **the thread team is spawned once per `ns_compute()` call**, not
once per kernel: `ns_compute`'s entire body runs inside a single `#pragma omp
parallel` block (SPMD-style — every thread executes the same call sequence,
and each `#pragma omp for`/`single` inside it, including inside the kernels it
calls, divides the actual work). `poisson_solver_Neumann_fft`'s `fftw_execute`
calls run inside `#pragma omp single` since FFTW plan execution isn't safe to
call concurrently from multiple threads on the same plan/buffer.

`A_compute_ss` (the periodic laser-field propagation, called once per main-loop
iteration alongside `ns_compute`) is **not parallelized in this pass** — its
outer `k`-loop has a genuine sequential dependency through shared FFTW
`in`/`out` buffers. Threading it would need per-thread buffers/plans or FFTW's
own thread support (`libfftw3_threads`, confirmed available alongside
`libfftw3` in the FFTW install this project links against) — left as a
follow-up.

## Correctness validation

Validated bit-identical at `OMP_NUM_THREADS=1`, `4`, and `48` against each
other and against `../Serial/bin/A-NS` on the small grid (`51 51 10 50`) —
confirms no data races from the parallelization (every loop iteration writes a
unique array index, no cross-iteration dependency in the parallelized loops).

## Intel VTune threading analysis

Needs the Intel oneAPI toolkit (`vtune`, `icpx`) on `PATH` — source
`~/intel/oneapi/setvars.sh` first, or submit via SLURM (VTune's hardware
counters need a compute node, not the login node):

    scripts/run_vtune.sh [nx ny nz n_iters]      # interactive, defaults to 51 51 10 50
    sbatch scripts/vtune.slurm [nx ny nz n_iters] # batch wrapper, same args

Runs a `vtune -collect threading` pass and prints the CPU utilization /
thread-concurrency summary, mirroring `../Serial/scripts/run_advisor.sh`'s
survey+report pattern. Thread count defaults to the node's full core count
(`OMP_NUM_THREADS`); override per-submission with
`sbatch --export=OMP_NUM_THREADS=8 scripts/vtune.slurm`.

## Current performance status (production grid, 1500 iters)

| Revision                                          | Wall time (`real`) |
|----------------------------------------------------|---------------------|
| Serial, vectorized + DCT Poisson solve (baseline)   | 254.078s            |
| OpenMP, first pass (one `parallel for` per kernel call) | ~241s            |
| OpenMP, consolidated team + merged BC calls          | ~239-241s           |

**This OpenMP port is correct but not yet meaningfully faster than the serial
baseline** — only ~5% improvement despite running on 48 cores. `vtune -collect
hotspots` on the production grid size identified the cause directly: **~70%+
of CPU time is spent inside `libgomp` (thread-team fork/join and barrier
wait), not in actual compute.** `#pragma omp for`/`single` each carry an
implicit barrier, and `ns_compute()` still has ~19 such work-sharing
constructs per call (down from ~31 before merging the per-array `Dirichlet_bc`
calls into `Dirichlet_bc3`) — at 1500 timesteps, that's ~28,500 barrier
synchronizations among 48 threads, several of them over loops whose total
work (e.g. the thin boundary-condition planes) is too small to amortize the
synchronization cost.

**Next steps** (not yet done, follow-up work):
- Further reduce barrier count: merge `Neumann_bc`'s and `Dirichlet_bc3`'s
  three face-pair loops into fewer work-sharing regions, or use `nowait` where
  the dependency analysis allows it.
- Thread `A_compute_ss` (currently fully serial) — likely the next-largest
  remaining lever once `ns_compute`'s overhead is brought down further, via
  per-thread FFTW buffers/plans or `libfftw3_threads`.
- Re-run `scripts/vtune.slurm` after each change to confirm the libgomp
  fraction is actually dropping before re-benchmarking — VTune, not intuition,
  is what found the real bottleneck here (it was not what the initial port
  assumed).
