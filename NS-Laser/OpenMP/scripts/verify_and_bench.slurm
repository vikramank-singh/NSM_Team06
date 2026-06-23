#!/bin/bash
#SBATCH --job-name=ns-laser-verify-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p debug
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclusive

source ~/intel/oneapi/setvars.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env_fftw.sh
# bin/A-NS is built ahead of time on the login node (debug-partition compute
# nodes have no GCC/icpx C++ header toolchain -- icpx fails with "C++ header
# location not resolved" if asked to compile there). This job only runs the
# already-built binary.

echo "==== correctness: small grid, 1/4/48 threads ===="
mkdir -p /tmp/verify_$$/{omp1,omp4,omp48,serial}
for nt in 1 4 48; do
  ( cd /tmp/verify_$$/omp$nt && OMP_NUM_THREADS=$nt "${SLURM_SUBMIT_DIR}/bin/A-NS" 51 51 10 50 > /dev/null )
done
( cd /tmp/verify_$$/serial && "${SLURM_SUBMIT_DIR}/../Serial/bin/A-NS" 51 51 10 50 > /dev/null )
python3 -c "
base='/tmp/verify_$$'
fields = ['T','u','v','w','I','p']
def maxdiff(a_dir, b_dir, f):
    a=[float(x) for x in open(f'{base}/{a_dir}/{f}.txt')]
    b=[float(x) for x in open(f'{base}/{b_dir}/{f}.txt')]
    return max(abs(x-y) for x,y in zip(a,b))
ok = True
for f in fields:
    d4 = maxdiff('omp1','omp4',f)
    d48 = maxdiff('omp1','omp48',f)
    ds = maxdiff('omp1','serial',f)
    print(f, 'omp1-vs-omp4', d4, 'omp1-vs-omp48', d48, 'omp1-vs-serial', ds)
    if d4 != 0 or d48 != 0 or ds != 0:
        ok = False
print('CORRECTNESS:', 'PASS' if ok else 'FAIL')
"
rm -rf /tmp/verify_$$

echo
echo "==== hotspots profile (production grid size, 100 iters) ===="
rm -rf vtune_hs
OMP_NUM_THREADS=48 vtune -collect hotspots -r vtune_hs -- bin/A-NS 201 201 20 100 --no-output
vtune -report hotspots -r vtune_hs -group-by function -report-knob show-issues=false > vtune_hs_report.txt || true
head -20 vtune_hs_report.txt
rm -rf vtune_hs vtune_hs_report.txt

echo
echo "==== production benchmark (full grid, 1500 iters, OMP_NUM_THREADS=48) ===="
cp bin/A-NS bench/opt_out/A-NS
cd bench/opt_out
rm -f run.log time.log status.txt
export OMP_NUM_THREADS=48
TIMEFORMAT='real %R
user %U
sys %S'
{ time ./A-NS > run.log ; } 2> time.log
echo "DONE" > status.txt
cat time.log
