"""
Entry point for the modded-nanogpt speedrun RL trainer.

Mirrors verl/trainer/main_ppo.py (ray init, FSDP worker mapping, worker set
actor+rollout / ref policy / critic, resume from global_step.txt) with two
differences:
  * config = base ppo_trainer.yaml + golf_trainer.yaml + speedrun/config.yaml
    + CLI dotlist overrides (key=value)
  * no RewardManager: grading happens in the external exec worker
    (speedrun/exec/worker.py) against the modded-nanogpt speedrun harness.
  * trainer = GolfPPOTrainer (speedrun/rl/verl_golf.py).

Run through scripts/run.sh (trainer role), which sets the environment and the
full argument list of the paper's runs.
"""
from __future__ import annotations

import os
import sys

import ray
from omegaconf import OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASE_YAML = os.path.join(_REPO_ROOT, "verl", "trainer", "config", "ppo_trainer.yaml")
_GOLF_YAML = os.path.join(os.path.dirname(__file__), "config", "golf_trainer.yaml")
_SPEEDRUN_YAML = os.path.join(_REPO_ROOT, "speedrun", "config.yaml")


def build_config(argv=None):
    dotlist = OmegaConf.from_dotlist(
        list(argv if argv is not None else sys.argv[1:]))
    # the CLI dotlist is merged last so it always wins
    config = OmegaConf.merge(
        OmegaConf.load(_BASE_YAML),
        OmegaConf.load(_GOLF_YAML),
        OmegaConf.load(_SPEEDRUN_YAML),
        dotlist,
    )
    # keep rollout.n in sync with golf.rollout_n (it sets the batch shape)
    config.actor_rollout_ref.rollout.n = config.golf.rollout_n
    return config


def main():
    config = build_config()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    # connect to a pre-started ray cluster if one exists, else start one
    addr = "auto" if os.environ.get("RAY_ADDRESS") or \
        os.environ.get("GOLF_RAY_EXISTING") else None
    info = ray.init(
        _temp_dir=f"/tmp/{os.environ.get('USER', 'ray')}/ray_tmp/{job_id}",
        address=addr,
        # avoid port collisions with other ray heads on the same node
        include_dashboard=False,
        runtime_env={
            "working_dir": ".",
            "excludes": [
                ".git/", ".github/", "cache/", "checkpoints/", "data/",
                "docker/", "docs/", "examples/", "logs/", "outputs/",
                "patches/", "reports/", "scripts/", "tests/",
                # weight dirs only; verl/models/ is also a python package
                "verl/models/Qwen*/", "verl/models/warm_start/",
                "verl.egg-info/", "wandb/",
                "workspace/", "__pycache__/",
            ],
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                # pass these through to worker processes explicitly
                **{k: os.environ[k] for k in
                   ("IMPL_API_BASE", "IMPL_API_KEY",
                    "GATEWAY_CLI_BIN", "SPEEDRUN_IMPL_BACKEND",
                    "SPEEDRUN_IMPLEMENTER_THINKING",
                    "SPEEDRUN_IMPLEMENTER_MODEL",
                    "PYTORCH_CUDA_ALLOC_CONF", "VERL_MEM_SNAPSHOT_DIR",
                    "CUDA_LAUNCH_BLOCKING", "PYTHONPATH",
                    "FLA_DISABLE_BACKEND_DISPATCH",
                    "TRITON_AUTOTUNE_FIRST_CONFIG",
                    "VERL_FORCE_MICRO_BSZ", "VERL_VLLM_SLEEP_LEVEL",
                    "VERL_NO_META_INIT", "VERL_SYNC_OFFLOAD_CPU",
                    "VERL_SYNC_GATHER_BEFORE_WAKE",
                    "GOLF_JUDGE", "GOLF_JUDGE_BACKEND", "GOLF_JUDGE_MODEL",
                    "GOLF_JUDGE_WORKERS", "GOLF_JUDGE_TIMEOUT_S")
                   if k in os.environ},
            },
        },
    )
    print(f"Dashboard: {info.dashboard_url}")
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from pprint import pprint
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    from verl.utils.fs import copy_local_path_from_hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    local_path = os.path.join(local_path, "model")

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    assert config.actor_rollout_ref.actor.strategy == "fsdp", \
        "golf trainer only supports the fsdp worker path"
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.main_ppo import init_start_global_step
    from speedrun.rl.verl_golf import GolfPPOTrainer

    # Worker set: actor+rollout, a ref policy for the KL-on-reward penalty,
    # and a critic for GAE. No reward model; grading lives in the exec worker.
    assert config.algorithm.adv_estimator == "gae", (
        "the speedrun trainer uses a critic with GAE; got adv_estimator="
        f"{config.algorithm.adv_estimator!r}")
    from verl.workers.fsdp_workers import CriticWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
        Role.Critic: global_pool_id,
    }
    print(f"[golf] critic path={config.critic.model.path} "
          f"ref path={config.actor_rollout_ref.model.ref_path} "
          f"kl_coef={config.algorithm.kl_ctrl.kl_coef}", flush=True)

    start_global_step = init_start_global_step(
        start_global_step=1,
        model_path=config.actor_rollout_ref.model.path,
    )

    trainer = GolfPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=None,        # grading via $GOLF_QUEUE_DIR, not a RewardManager
        val_reward_fn=None,
        start_global_step=start_global_step,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
