#!/bin/bash
# One run of the NanoGPT speedrun experiment on a SLURM allocation: the first
# node runs the trainer (PPO on NGPU GPUs + the job queue), every other node
# runs a grader that executes the proposed attempts on 8 GPUs. The paper used
# 1 trainer + 5 graders per run.
#
#   sbatch scripts/run.sbatch rr
#   sbatch scripts/run.sbatch vanilla
#
# Adjust --nodes / --gres / --time (and add --partition / --account) for your
# cluster; scripts/run.sh lists the environment variables it reads.
#SBATCH --job-name=speedrun
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/speedrun-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OBJECTIVE=${1:?"usage: sbatch scripts/run.sbatch rr|vanilla"}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
if (( ${#nodes[@]} < 2 )); then
    echo "need at least 2 nodes (1 trainer + graders)" >&2
    exit 1
fi
export QUEUE_URL="http://${nodes[0]}:${QUEUE_PORT:-8377}"
echo "trainer ${nodes[0]}; graders ${nodes[*]:1} -> $QUEUE_URL"

srun --nodes=1 --ntasks=1 -w "${nodes[0]}" bash scripts/run.sh trainer "$OBJECTIVE" &
TRAINER=$!
srun --nodes=$(( ${#nodes[@]} - 1 )) --ntasks-per-node=1 --exclude="${nodes[0]}" \
    bash scripts/run.sh grader &
GRADERS=$!

# the run ends when the trainer does; the graders would otherwise wait forever
rc=0
wait "$TRAINER" || rc=$?
kill "$GRADERS" 2>/dev/null || true
wait "$GRADERS" 2>/dev/null || true
exit "$rc"
