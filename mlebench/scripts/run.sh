#!/bin/bash
# Train RPG and vanilla PPO on every competition in scripts/tasks.tsv
#
#   bash scripts/run.sh                             # 22 competitions x {rpg, vanilla}
#   bash scripts/run.sh leaf-classification         # one competition, both arms
#   bash scripts/run.sh leaf-classification rpg     # one competition, one arm
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ONLY_TASK=${1:-}
ONLY_ARM=${2:-}
if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != rpg && "$ONLY_ARM" != vanilla ]]; then
    echo "arm must be 'rpg' or 'vanilla', got '$ONLY_ARM'" >&2
    exit 1
fi

MODEL_DIR=${MODEL_DIR:-$REPO/verl/models}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs}
REPS=${REPS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-rpg-mlebench}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-paper-fleet}
if [[ -z "${N_GPUS:-}" ]]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}

# The actor and reference policy load a text-only view of the checkpoint so
# FSDP training and the vLLM rollout use the same model class; the critic
# (token-classification head) loads the full checkpoint.
TEXT_MODEL=$MODEL_DIR/Qwen3.5-4B-text
FULL_MODEL=$MODEL_DIR/Qwen3.5-4B

# On a single GPU the paper used a reference log-prob micro-batch of 8, and 4
# with two GPUs. Micro-batching changes memory, not the update.
REF_MICRO_BATCH=$([[ "$N_GPUS" == 1 ]] && echo 8 || echo 4)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_usage_threshold=1.0
export RAY_heartbeat_timeout_milliseconds=300000
export RAY_gcs_node_heartbeat_timeout_milliseconds=300000
export HYDRA_FULL_ERROR=1
export WANDB_INIT_TIMEOUT=600
export WANDB_X_FILE_STREAM_RETRY_MAX=60
export WANDB_X_GRAPHQL_RETRY_MAX=20

# ---------------------------------------------------------------- trainer overrides
common_overrides() {   # $1 data dir, $2 timeout, $3 use_leaderboard_score, $4 name
    local data=$1 timeout=$2 leaderboard=$3 name=$4
    local ckpt=$OUTPUT_DIR/checkpoints/$name
    local actor=$TEXT_MODEL critic=$FULL_MODEL
    if [[ -d "$ckpt/actor" ]]; then          # resume an interrupted run
        actor=$ckpt/actor
        critic=$ckpt/critic
    fi
    cat <<EOF
data.train_files=$data/train.parquet
data.timeout=$timeout
data.train_batch_size=128
data.max_prompt_length=2560
data.max_response_length=2500
actor_rollout_ref.model.path=$actor
actor_rollout_ref.model.ref_path=$TEXT_MODEL
actor_rollout_ref.model.enable_gradient_checkpointing=True
+actor_rollout_ref.model.attn_implementation=sdpa
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size=4
actor_rollout_ref.actor.ppo_epochs=100
actor_rollout_ref.actor.grad_clip=1.0
actor_rollout_ref.actor.clip_ratio=0.2
actor_rollout_ref.actor.entropy_coeff=0.001
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Qwen3_5DecoderLayer']
actor_rollout_ref.ref.log_prob_micro_batch_size=$REF_MICRO_BATCH
actor_rollout_ref.ref.fsdp_config.param_offload=True
+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Qwen3_5DecoderLayer']
actor_rollout_ref.rollout.name=vllm
+actor_rollout_ref.rollout.load_path=$FULL_MODEL
actor_rollout_ref.rollout.temperature=0.7
actor_rollout_ref.rollout.top_k=20
actor_rollout_ref.rollout.top_p=0.8
+actor_rollout_ref.rollout.presence_penalty=1.5
actor_rollout_ref.rollout.dtype=bfloat16
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.rollout.ignore_eos=False
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS
actor_rollout_ref.rollout.log_prob_micro_batch_size=4
critic.model.path=$critic
critic.model.enable_gradient_checkpointing=True
+critic.model.attn_implementation=sdpa
critic.model.fsdp_config.optimizer_offload=True
critic.optim.lr=1e-5
critic.ppo_micro_batch_size=4
critic.forward_micro_batch_size=4
critic.grad_clip=1.0
critic.cliprange_value=0.5
algorithm.adv_estimator=gae
algorithm.gamma=1.0
algorithm.lam=1.0
algorithm.kl_ctrl.kl_coef=0.001
reward.self_improve_type=pure_code
reward.use_leaderboard_score=$leaderboard
reward.buffer_size=10
reward.num_cpus_per_sample=1
reward.code_mem_limit_gb=16
reward.scalar_fn_name=identity
reward.b_score=0.0
trainer.use_topk_buffer=True
trainer.self_improve_bonus=0.5
trainer.topk_buffer_capacity_mult=10
trainer.use_submission_rate_reweighting=False
trainer.multi_step=1
trainer.total_epochs=100
trainer.save_freq=5
trainer.logger=['wandb']
trainer.project_name=$WANDB_PROJECT
trainer.experiment_name=$name
trainer.default_local_dir=$ckpt
trainer.default_hdfs_dir=null
trainer.workspace_dir=$OUTPUT_DIR/workspace/$name
trainer.n_gpus_per_node=$N_GPUS
trainer.nnodes=1
EOF
}

rpg_overrides() {      # RPG: time penalty priced at the NIW reward-rate estimate
    cat <<EOF
actor_rollout_ref.actor.use_reward_rate_penalty=True
+actor_rollout_ref.actor.reward_rate_penalty_coeff=1
actor_rollout_ref.actor.reward_rate_div_avg_time=constant
+actor_rollout_ref.actor.divide_avg_time=True
+actor_rollout_ref.actor.expected_reward_rate=True
actor_rollout_ref.actor.use_niw_reward_rate=True
actor_rollout_ref.actor.niw_forgetting_factor=0.3
actor_rollout_ref.actor.reward_rate_exclude_invalid=False
actor_rollout_ref.actor.penalize_invalid_time=True
reward.invalid_zero_reward_full_time=True
reward.no_submission_penalty=0.0
reward.invalid_reward_rate_penalty=-10.0
reward.baseline_score=null
EOF
}

vanilla_overrides() {  # vanilla: reward only; the reward rate is logged, not used
    cat <<EOF
actor_rollout_ref.actor.use_reward_rate_penalty=False
actor_rollout_ref.actor.log_reward_rate=True
+actor_rollout_ref.actor.divide_avg_time=False
reward.no_submission_penalty=-10.0
+reward.baseline_score=null
EOF
}

# ---------------------------------------------------------------- one run
start_ray() {          # a single-node Ray head for this run
    local attempt base
    for attempt in 1 2 3 4 5; do
        base=$(shuf -i 10000-19000 -n 1)
        export RAY_ADDRESS="127.0.0.1:$base"
        # kept short on purpose: Ray puts Unix sockets under it, and their paths
        # may not exceed 107 bytes (a competition name here would overflow)
        export RAY_TEMP_DIR="/tmp/$USER/ray/$$-$attempt"
        # cap the object store (default 40 GB); left uncapped Ray takes 30% of
        # memory and fills toward it over a multi-day run
        local store=(--object-store-memory=$(( ${RAY_OBJECT_STORE_MEM_GB:-40} * 1024 ** 3 )))
        if ray start --head --node-ip-address=127.0.0.1 --port="$base" \
                --node-manager-port=$((base + 1)) --object-manager-port=$((base + 2)) \
                --min-worker-port=$((base + 100)) --max-worker-port=$((base + 199)) \
                --dashboard-agent-listen-port=0 --include-dashboard=False \
                --num-gpus="$N_GPUS" --num-cpus="$NUM_CPUS" "${store[@]}" \
                --temp-dir="$RAY_TEMP_DIR"; then
            return 0
        fi
        ray stop --force >/dev/null 2>&1 || true      # port clash: retry
        sleep $((RANDOM % 10 + 1))
    done
    echo "could not start a Ray head after 5 attempts" >&2
    return 1
}

run_one() {            # $1 competition, $2 timeout, $3 leaderboard, $4 arm, $5 rep
    local comp=$1 timeout=$2 leaderboard=$3 arm=$4 rep=$5
    RUN_NAME="${comp}_${arm}_rep${rep}"
    local overrides
    overrides=$( common_overrides "$DATA_ROOT/$comp" "$timeout" "$leaderboard" "$RUN_NAME"
                 "${arm}_overrides" )
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "### $RUN_NAME"
        echo "$overrides"
        return 0
    fi
    if [[ ! -f "$DATA_ROOT/$comp/train.parquet" ]]; then
        echo "missing $DATA_ROOT/$comp/train.parquet -- run scripts/preprocess.sh $comp" >&2
        return 1
    fi
    export WANDB_DIR="/tmp/$USER/wandb/$RUN_NAME"
    mkdir -p "$WANDB_DIR" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/workspace"
    echo "=== $RUN_NAME  (timeout ${timeout}s, leaderboard=$leaderboard, $N_GPUS GPU)"
    start_ray
    local status=0
    mapfile -t args <<< "$overrides"
    python3 -m verl.trainer.main_ppo "${args[@]}" 2>&1 || status=$?
    ray stop --force >/dev/null 2>&1 || true
    return "$status"
}

# ---------------------------------------------------------------- the sweep
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
found=0
failed=()
while IFS=$'\t' read -r comp timeout leaderboard; do
    if [[ -z "$comp" || "$comp" == \#* ]]; then continue; fi
    if [[ -n "$ONLY_TASK" && "$ONLY_TASK" != "$comp" ]]; then continue; fi
    found=1
    for arm in rpg vanilla; do
        if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != "$arm" ]]; then continue; fi
        for ((rep = 1; rep <= REPS; rep++)); do
            run_one "$comp" "$timeout" "$leaderboard" "$arm" "$rep" \
                || failed+=("${comp}_${arm}_rep${rep}")
        done
    done
done < scripts/tasks.tsv

if [[ "$found" == 0 ]]; then
    echo "unknown competition '$ONLY_TASK' -- see scripts/tasks.tsv" >&2
    exit 1
fi
if (( ${#failed[@]} )); then
    echo "FAILED runs: ${failed[*]}" >&2
    exit 1
fi
