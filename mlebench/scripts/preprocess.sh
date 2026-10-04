#!/bin/bash
# Build the train/test parquet for every competition in scripts/tasks.tsv,
# serially, or for one competition when its id is given.
#
#   bash scripts/preprocess.sh                       # all 22
#   bash scripts/preprocess.sh leaf-classification   # one
#
# MLE_BENCH_DATA must point at your prepared mle-bench tree: the prompts refer
# to the data through that one placeholder. Output: $DATA_ROOT/<competition_id>/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

: "${MLE_BENCH_DATA:?set MLE_BENCH_DATA to the root of your prepared mle-bench tree}"
DATA_ROOT=${DATA_ROOT:-data}
PROMPTS=examples/data_preprocess
ONLY=${1:-}

found=0
while IFS=$'\t' read -r comp _; do
    if [[ -z "$comp" || "$comp" == \#* ]]; then continue; fi
    if [[ -n "$ONLY" && "$ONLY" != "$comp" ]]; then continue; fi
    found=1
    echo "=== $comp -> $DATA_ROOT/$comp"
    mkdir -p "$DATA_ROOT/$comp"
    python examples/data_preprocess/mlebench.py \
        --competition_id "$comp" \
        --local_dir "$DATA_ROOT/$comp" \
        --prompt-path "$PROMPTS/$comp.txt" \
        --self-improve-prompt-path "$PROMPTS/self-improve/pure_code/$comp.txt"
done < scripts/tasks.tsv

if [[ "$found" == 0 ]]; then
    echo "unknown competition '$ONLY' -- see scripts/tasks.tsv" >&2
    exit 1
fi
