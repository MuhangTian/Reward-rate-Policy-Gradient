#!/bin/bash
#SBATCH --job-name=large_context
#SBATCH --partition=cpu
#SBATCH --time=11:00:00
#SBATCH -N 1 -n 1 -c 32 --mem=64G
#SBATCH --array=0-63%24
#SBATCH --output=logs/large_context_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"     # submit from bandit/
mkdir -p logs
FAMS=(E1_indep E2_poscorr E3_negcorr E4_lognorm)
KS=(70 100 130 160); XS=(8 16 32 64)
i=$SLURM_ARRAY_TASK_ID
F=${FAMS[$(( i % 4 ))]}; r=$(( i / 4 ))
K=${KS[$(( r % 4 ))]}; X=${XS[$(( r / 4 ))]}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "large-context task $i -> $F K=$K X=$X  (lambda every step, 30 seeds, exact GT)"
srun python3 run_large_context.py "$F" "$K" "$X" "$SLURM_CPUS_PER_TASK"
