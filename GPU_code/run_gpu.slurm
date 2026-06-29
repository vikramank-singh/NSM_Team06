#!/bin/bash
# ===========================================================================
# Single-GPU OpenMP target-offload run (NVIDIA A100) for the coupled
# thermal-laser solver. The NS/Poisson/T stencils run on the GPU; the laser
# BPM (FFTW) runs on the host CPU, so we also give the rank a few CPU cores.
#
#   sbatch run_gpu.slurm
# ===========================================================================
#SBATCH --job-name=ns-laser-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16            # host threads for the FFTW laser march
#SBATCH --time=10:30:00
#SBATCH --partition=gpu               # <-- set your GPU partition

cd "${SLURM_SUBMIT_DIR:-.}"

# ---- toolchain: NVHPC for nvc++ -mp=gpu (or load LLVM for the clang build) --
source ~/source/spack/share/spack/setup-env.sh
spack load nvhpc@25.11 +mpi

export FFTW_ROOT=/home/samirs/source/spack/opt/spack/linux-cascadelake/fftw-3.3.10-p7gx6r42mzcw4t3fgneys3dhyyqfs4vj
export LD_LIBRARY_PATH=$FFTW_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$FFTW_ROOT/include:$CPATH
export PATH=$FFTW_ROOT/bin:$PATH



# host threads for the laser FFT march
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Fail loudly if the kernels can't actually run on the GPU instead of silently
# falling back to the host (which would look "correct" but slow).
export OMP_TARGET_OFFLOAD=MANDATORY

NX=301; NY=301; NZ=20; ITERS=500
LX=5.0; LY=5.0; LZ=200.0
METHOD=jacobi                         # jacobi | rbgs
export POISSON_SWEEPS=150

echo "=== GPU run: ${NX}x${NY}x${NZ} iters=${ITERS} method=${METHOD} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

#nsys profile \
#  --trace=cuda,nvtx,osrt \
#  --stats=true \
#  --force-overwrite=true \
#  -o nsys_ans_gpu \
#  ./A-NS-gpu 301 301 20 10 5.0 5.0 200.0 jacobi

#exit

nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o nsys_ans_gpu \
  ././A-NS-gpu-cufft 301 301 20 10 5.0 5.0 200.0 jacobi

exit

./A-NS-gpu-cufft ${NX} ${NY} ${NZ} ${ITERS} ${LX} ${LY} ${LZ} ${METHOD}
echo "=== exit code: $? ==="

# Expect the program to print 'omp_get_num_devices()=1' (or more) and
# 'device=GPU ...'. If you instead see device count 0 with MANDATORY set, the
# offload toolchain/runtime isn't finding the GPU -- check the module and the
# -gpu=cc80 build flag.

