#!/bin/bash
#SBATCH --job-name=ns-laser-advisor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p hm
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclusive

# Batch wrapper around scripts/run_advisor.sh for the SLURM cluster.
# Submit from NS-Laser/Serial/: sbatch scripts/advisor.slurm [nx ny nz iters]

ulimit -s unlimited
export OMP_NUM_THREADS=1

source ~/intel/oneapi/setvars.sh

cd "${SLURM_SUBMIT_DIR}"
./scripts/run_advisor.sh "$@"
