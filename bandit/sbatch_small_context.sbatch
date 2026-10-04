#!/bin/bash
#SBATCH --job-name=small_context
#SBATCH -p cpu -t 11:00:00 -N 1 -n 1 -c 32 --mem=64G
#SBATCH --array=0-15%16
#SBATCH -o logs/small_context_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"     # submit from bandit/
mkdir -p logs
FAMS=(E1_indep E2_poscorr E3_negcorr E4_lognorm)
KS=(30 20 10 5)          # LONGEST FIRST: K=30 enumerates 31^4 every step (~3.5h)
i=$SLURM_ARRAY_TASK_ID
F=${FAMS[$(( i % 4 ))]}; K=${KS[$(( i / 4 ))]}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "small-context task $i -> $F K=$K"
srun python3 run_small_context.py "$F" "$K" "$SLURM_CPUS_PER_TASK"
