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

Thread count is controlled at runtime via `OMP_NUM_THREADS`, not at build time.
**Use 16-24 threads for the production grid, not the node's full 48** — see
"Current performance status" below; this isn't a typo, more threads measurably
hurts at this problem size.

    OMP_NUM_THREADS=24 ./bin/A-NS                          # production grid: 201x201x20, 1500 iters
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
iteration alongside `ns_compute`) is now also parallelized: it's wrapped in
its own `#pragma omp parallel` (SPMD-style, same pattern as `ns_compute`),
with `#pragma omp for collapse(2)` on the pack/phase-multiply/normalize-writeback
loops and `#pragma omp single` around each `fftw_execute` call (its `k`-loop
still has a genuine sequential dependency through the shared `in`/`out`
buffers, so the loop body itself can't be parallelized across `k` — only the
work *within* each `k`-layer is). FFTW is also configured to use its own
internal thread pool (`fftw_init_threads()` + `fftw_plan_with_nthreads()`,
linking `libfftw3_threads`) so the `fftw_execute` calls themselves — both here
and in `poisson_solver_Neumann_fft` — are multi-threaded even though only one
OpenMP thread issues each call. This turned out to be the dominant lever (see
"Current performance status" below) — `A_compute_ss` was previously fully
single-threaded and was blocking all other threads from useful work during
its phase each iteration, which wall-clock benchmarking exposed far more
sharply than CPU-time hotspot attribution did.

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

| Revision                                                    | Wall time (`real`) |
|---------------------------------------------------------------|---------------------|
| Serial, vectorized + DCT Poisson solve (baseline)              | 254.078s            |
| OpenMP, first pass (one `parallel for` per kernel call)         | ~241s (48 threads)  |
| OpenMP, consolidated team + merged `Dirichlet_bc3`              | ~239-241s (48 threads) |
| OpenMP, `collapse(2)` grain-size fix + dead-code removal, 48 threads | 236.879s       |
| OpenMP, same code, 24 threads (measured sweet spot)             | 233.561s        |
| OpenMP, `A_compute_ss` threaded + FFTW internal threading, 24 threads | **43.037s** |

**Latest result: 43.037s at 24 threads — ~5.9x faster than the serial
baseline, ~5.4x faster than the previous OpenMP revision.** Threading
`A_compute_ss` (previously fully serial) was the dominant lever, confirmed via
`scripts/verify_and_bench.slurm`: correctness PASS (bit-identical at
`OMP_NUM_THREADS=1/4/48` against each other and against `../Serial/bin/A-NS`),
and `user`/`real` time ratio (1156.380s / 43.037s ≈ 26.9) consistent with
genuine multi-core utilization at 24 threads. `vtune -collect hotspots` on the
production grid still shows `libgomp` barrier-wait dominating CPU-*time*
share (~87%, `func@0x1dfd0`), but this no longer correlates with wall-clock —
the earlier ~8%-faster ceiling was bottlenecked on `A_compute_ss` blocking all
threads serially each iteration, which CPU-time-based attribution
under-weighted relative to its actual wall-clock impact.

Two earlier rounds of profiling-driven fixes (still in effect, but superseded
as the dominant lever by `A_compute_ss` threading):

1. **Grain size**: `Neumann_bc`/`Dirichlet_bc`/`Dirichlet_bc3`'s bottom/top and
   left/right face-pair loops were driven by `nz` (20 for the production
   grid) — smaller than the thread count, so most threads got 0-1 work items.
   Added `collapse(2)` to flatten each face-pair's two loop dimensions
   together, giving every face-pair `nx*ny` or `ny*nz` work items regardless
   of `nz`. Also removed a dead `p_prime[i]=0.0` init loop (leftover from the
   old Gauss-Seidel solver, which needed an initial guess to iterate from —
   the direct DCT solver fully overwrites `p_prime` and never reads its prior
   value, so the zero-init and its barrier were pure waste).
2. **Thread-count sweep** (`bin/A-NS 201 201 20 100` at 1/2/4/8/12/16/24/32/48
   threads, via SLURM on an exclusive `debug`-partition node): wall time
   *improves* from 1 to ~16-24 threads, then **gets worse again at 48** —
   confirms the bottleneck is barrier-synchronization cost (which scales with
   thread count), not available parallel work. 16-24 threads is the measured
   sweet spot for this grid size; 48 threads is strictly worse.

Despite both fixes, `vtune -collect hotspots` still shows **~70%+ of CPU time
inside `libgomp`** (thread fork/join + barrier wait), not compute — confirming
the real lever is *barrier count*, not load balance within each barrier.
`ns_compute()` has ~24 work-sharing constructs per call (each with an implicit
barrier); at 1500 timesteps that's ~36,000 barrier synchronizations, several
of them (the boundary-condition loops) over inherently thin work that can
never fully amortize synchronization cost no matter how well the work is
balanced across threads.

**Next steps** (not yet done, follow-up work, in descending expected ROI):
- Re-sweep thread count (1/2/4/8/12/16/24/32/48) now that `A_compute_ss`
  threading + FFTW internal threading changed the workload balance — the
  previous 16-24 thread sweet spot was measured before this change and may no
  longer be optimal (FFTW's own thread pool adds a second layer of
  parallelism on top of OpenMP's).
- Eliminate `T_compute`'s copy-back loop (`T[n]=T_new[n]`, one more barrier
  per call) via a ping-pong pointer swap instead of a copy — requires
  threading a `double*&` reference through `ns_compute`/`A-NS.cpp`'s main
  loop.
- More aggressive barrier-count reduction (e.g. fusing `Neumann_bc`'s/
  `Dirichlet_bc3`'s three face-pair loops into fewer work-sharing regions)
  was considered but risks data races at shared edge/corner cells between
  face-pairs if done via `nowait` — not attempted.
- Re-run `scripts/vtune.slurm`/`scripts/verify_and_bench.slurm` after each
  change to confirm the libgomp fraction is actually dropping before
  re-benchmarking — VTune, not intuition, is what found the real bottleneck
  both times.

## Benchmarking notes

- **Always benchmark via SLURM on an exclusive node** (`scripts/verify_and_bench.slurm`
  or `bench/bench_opt.slurm`), never on the login node — it's shared with other
  users (load average from other sessions alone can exceed the core count) and
  gives meaningless timing.
- The `debug` partition's compute nodes have **no GCC/icpx C++ header
  toolchain** (`icpx` fails with "C++ header location not resolved" if asked
  to compile there) — build `bin/A-NS` on the login node first, then only
  *run* the pre-built binary via SLURM. `scripts/verify_and_bench.slurm`
  follows this pattern.
