#!/bin/bash
# Launch one role of the NanoGPT speedrun experiment.
#
#   bash scripts/run.sh trainer rr        # or: trainer vanilla
#   bash scripts/run.sh grader
#
# The trainer serves an HTTP job queue and runs PPO; each grader pulls the
# attempts it proposes and runs them on its own 8 GPUs. The roles may live on
# different machines: a grader only needs QUEUE_URL (or QUEUE_HOST) pointing at
# the trainer. scripts/run.sbatch places one trainer and N graders on a SLURM
# allocation.
#
# Environment:
#   MODEL_DIR      Qwen3.5-4B (critic, rollout) and Qwen3.5-4B-text (actor, ref)
#                                                              [verl/models]
#   FINEWEB_DIR    the fineweb10B shards the attempts train on (graders)
#                                                     [data/fineweb10B]
#   OUTPUT_DIR     buffers, attempt archive, checkpoints              [outputs]
#   QUEUE_URL      graders: the trainer's queue, e.g. http://node01:8377
#   QUEUE_HOST     graders: alternatively just the trainer's hostname
#   QUEUE_PORT     queue port                                            [8377]
#   GRADER_DIR     grader-local scratch for attempt sandboxes
#                                                 [/tmp/$USER/speedrun_grader]
#   Implementer and judge (the paper used gpt-5.5):
#   IMPL_API_KEY (or OPENAI_API_KEY), IMPL_API_BASE   an OpenAI-compatible endpoint
#   SPEEDRUN_IMPLEMENTER_MODEL, GOLF_JUDGE_MODEL                     [gpt-5.5]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1

ROLE=${1:?"usage: run.sh trainer rr|vanilla, or run.sh grader"}
OBJECTIVE=${2:-}
MODEL_DIR=${MODEL_DIR:-$REPO/verl/models}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs}
PORT=${QUEUE_PORT:-8377}
export GOLF_WALL_SECONDS=1500      # per-attempt wall clock (grader kill, trainer charge)

# =============================================================== grader
if [[ "$ROLE" == grader ]]; then
    if [[ -z "${QUEUE_URL:-}" ]]; then
        : "${QUEUE_HOST:?set QUEUE_URL or QUEUE_HOST to the trainer host}"
        QUEUE_URL="http://$QUEUE_HOST:$PORT"
    fi
    export GOLF_QUEUE_URL=$QUEUE_URL
    export GOLF_ATTEMPT_DATA_PATH=${FINEWEB_DIR:-$REPO/data/fineweb10B}
    export GOLF_SANDBOX_NETNS=0
    # node-local scratch for the attempts' sandboxes (never shared between hosts)
    export GOLF_QUEUE_DIR=${GRADER_DIR:-/tmp/$USER/speedrun_grader}
    mkdir -p "$GOLF_QUEUE_DIR"
    echo "[speedrun] grader on $(hostname) -> $GOLF_QUEUE_URL"
    # the trainer loads a 4B model before its queue comes up
    python3 -m speedrun.exec.netqueue wait --timeout-s 3600
    # exit 3 = "GPUs left dirty, restart me"
    while :; do
        rc=0
        python3 -m speedrun.exec.worker || rc=$?
        if [[ "$rc" -eq 3 ]]; then
            echo "[speedrun] worker exit 3 (dirty GPUs) -- restarting"
            sleep 10
            continue
        fi
        exit "$rc"
    done
fi
[[ "$ROLE" == trainer ]] || { echo "unknown role '$ROLE' (trainer | grader)" >&2; exit 2; }

# =============================================================== trainer
# The two arms differ only in the objective and its reward shaping.
case "$OBJECTIVE" in
    rr)      OBJ_ARGS=(golf.use_reward_rate=True golf.reward_mode=quality
                       golf.quality_cross_bonus=5.0 golf.time_below_target_weight=0.0) ;;
    vanilla) OBJ_ARGS=(golf.use_reward_rate=False golf.reward_mode=time_to_target
                       golf.quality_cross_bonus=0.0 golf.time_below_target_weight=20000.0) ;;
    *) echo "trainer objective must be 'rr' or 'vanilla'" >&2; exit 2 ;;
esac

export GOLF_QUEUE_DIR=$OUTPUT_DIR/queue_$OBJECTIVE
export GOLF_QUEUE_URL=http://127.0.0.1:$PORT   # the trainer talks to its own queue
export GOLF_JUDGE=1
export GOLF_CLAIM_TIMEOUT_S=7200
export SPEEDRUN_IMPLEMENTER_MODEL=${SPEEDRUN_IMPLEMENTER_MODEL:-gpt-5.5}
export GOLF_JUDGE_MODEL=${GOLF_JUDGE_MODEL:-$SPEEDRUN_IMPLEMENTER_MODEL}
export FLA_DISABLE_BACKEND_DISPATCH=1   # read by flash-linear-attention, when installed
export VERL_NO_META_INIT=1
SEED_ARCHIVE=${SEED_ARCHIVE:-$REPO/speedrun/seed_archive_rescored_1500s}
BUFFER_PATH=$GOLF_QUEUE_DIR/buffer.jsonl
NGPU=4
mkdir -p "$GOLF_QUEUE_DIR"

ARGS=(
    golf.batch_size=64
    golf.rollout_n=1
    golf.use_delta_specs=True
    golf.implementer_workers=16
    golf.buffer_path="$BUFFER_PATH"
    golf.allow_empty_state=False
    golf.timeout="$GOLF_WALL_SECONDS"
    golf.quality_miss_loss=10.0
    golf.quality_invalid_reward=-10.0
    golf.quality_below_target_weight=100.0
    golf.quality_below_target_floor=3.25
    golf.explore_edit_frac=0.5
    golf.explore_edit_min_lines=20
    golf.tie_break_seed=0
    "${OBJ_ARGS[@]}"
    data.train_batch_size=64
    data.max_prompt_length=40960
    data.max_response_length=2560
    actor_rollout_ref.model.path="$MODEL_DIR/Qwen3.5-4B-text"
    actor_rollout_ref.model.ref_path="$MODEL_DIR/Qwen3.5-4B-text"
    actor_rollout_ref.model.attn_implementation=sdpa
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    "actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Qwen3_5DecoderLayer']"
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size=$NGPU          # 1 per rank
    actor_rollout_ref.actor.ppo_epochs=2
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.logits_to_keep_response_only=True
    actor_rollout_ref.ref.log_prob_micro_batch_size=$((4 * NGPU))   # 4 per rank
    actor_rollout_ref.ref.logits_to_keep_response_only=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    "actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Qwen3_5DecoderLayer']"
    actor_rollout_ref.rollout.load_path="$MODEL_DIR/Qwen3.5-4B"
    actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.top_p=0.8
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.min_p=0.0
    actor_rollout_ref.rollout.presence_penalty=1.5
    actor_rollout_ref.rollout.repetition_penalty=1.0
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$((4 * NGPU))
    critic.ppo_mini_batch_size=64
    critic.model.path="$MODEL_DIR/Qwen3.5-4B"
    critic.model.attn_implementation=sdpa
    critic.model.enable_gradient_checkpointing=True
    critic.model.fsdp_config.optimizer_offload=True
    "critic.model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=['Qwen3_5DecoderLayer']"
    critic.optim.lr=1e-5
    critic.grad_clip=1.0
    critic.cliprange_value=0.5
    critic.ppo_micro_batch_size=$NGPU
    critic.forward_micro_batch_size=$((4 * NGPU))
    algorithm.adv_estimator=gae
    algorithm.gamma=1.0
    algorithm.lam=1.0
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.n_gpus_per_node=$NGPU
    trainer.nnodes=1
    trainer.logger=[wandb]
    trainer.project_name="${WANDB_PROJECT:-nanogpt-speedrun}"
    trainer.experiment_name="speedrun-4b-$OBJECTIVE-b64"
    trainer.save_freq=0
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints/speedrun-$OBJECTIVE"
)

# Seed the buffer once: the archive stores repo-relative code paths.
if [[ ! -e "$BUFFER_PATH" ]]; then
    python3 - "$SEED_ARCHIVE/buffer.jsonl" "$BUFFER_PATH" "$REPO" <<'PY'
import json, os, sys
src, dst, root = sys.argv[1:4]
with open(dst, "w") as out:
    for line in open(src):
        if not line.strip():
            continue
        n = json.loads(line)
        cp = n.get("code_path")
        if cp and not os.path.isabs(cp):
            n["code_path"] = os.path.join(root, cp)
        if n.get("code_path") and n.get("_event") != "expand" \
                and not os.path.exists(n["code_path"]):
            raise SystemExit(f"seed code missing: {n['code_path']}")
        out.write(json.dumps(n) + "\n")
print(f"[speedrun] seeded {dst} from {src}")
PY
fi

python3 -m speedrun.preflight "${ARGS[@]}"

# The queue graders pull from, and a GPU heartbeat while the trainer waits on them.
python3 -m speedrun.exec.netqueue serve --port "$PORT" &
NETQ_PID=$!
python3 -m speedrun.tools.gpu_heartbeat &
HB_PID=$!
trap 'kill $NETQ_PID $HB_PID 2>/dev/null || true' EXIT
python3 -m speedrun.exec.netqueue wait --timeout-s 60
echo "[speedrun] trainer on $(hostname): graders should use http://$(hostname):$PORT"

rc=0
python3 -m speedrun.rl.main_golf "${ARGS[@]}" || rc=$?
exit $rc
