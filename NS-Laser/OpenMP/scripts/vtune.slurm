#!/bin/bash
#SBATCH --job-name=ns-laser-vtune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p hm
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclusive

# Batch wrapper around scripts/run_vtune.sh for the SLURM cluster.
# Submit from NS-Laser/OpenMP/: sbatch scripts/vtune.slurm [nx ny nz iters]
#
# Thread count defaults to the node's full core count; override per-submission
# with: sbatch --export=OMP_NUM_THREADS=8 scripts/vtune.slurm

ulimit -s unlimited
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-48}}"

source ~/intel/oneapi/setvars.sh

cd "${SLURM_SUBMIT_DIR}"
./scripts/run_vtune.sh "$@"
