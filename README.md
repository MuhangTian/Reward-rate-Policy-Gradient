# Reward-rate Policy Gradient for Efficient Machine Learning Engineering Agents
[![arxiv badge](https://img.shields.io/badge/arXiv-2609.36393-red)](https://arxiv.org/abs/2609.36393)

This repository contains the code implementation for the experiments in our paper. We train language-model agents with *Reward-rate Policy Gradient (RPG)*, which prices every second an attempt takes at the agent's own reward rate, and compare it with vanilla RL on a contextual bandit simulation, MLE-Bench Lite, and the NanoGPT speedrun.

## Requirements 🛠️
* Python 3.11. Install all packages with
```bash
pip install -r requirements.txt
```
* The two training stacks under `mlebench/verl` and `nanogpt/verl` are modified copies of [verl](https://github.com/volcengine/verl) and are not interchangeable. They are not installed; each launch script puts its own subtree on `PYTHONPATH`.
* The policy is Qwen3.5-4B. `python prepare_models.py <model_dir>` downloads it and derives the text-only checkpoint used by the actor and reference policy. Pass `MODEL_DIR=<model_dir>` to the launch scripts (default: `<subtree>/verl/models`).

## Code Structure 📚
* `bandit/`: contextual bandit simulation
    * `ctx_envs.py`: environments (E1–E4 families) and their exact optimal reward rates
    * `ctx_agents.py`: C-UCB (theory and tuned), NPG-NIW, and SPG-NIW
    * `run_small_context.py`, `run_large_context.py`: tune and evaluate every agent on one instance; `sbatch_*.sbatch` run all instances on SLURM
* `mlebench/`: MLE-Bench Lite experiments
    * `scripts/preprocess.sh`, `scripts/run.sh`, `scripts/tasks.tsv`: data preparation, training, and the per-competition settings
    * `verl/trainer/ppo/ray_trainer.py`: PPO with the reward-rate penalty; `verl/utils/reward_rate_niw.py`: the NIW reward-rate estimator
    * `verl/utils/reward_score/`: sandboxed execution and grading; `examples/data_preprocess/`: prompts
* `nanogpt/`: NanoGPT speedrun experiments
    * `scripts/run.sh`, `scripts/run.sbatch`: launch the trainer and the graders
    * `speedrun/reward.py`: rewards of both arms; `speedrun/rl/verl_golf.py`: trainer and reward-rate estimate
    * `speedrun/exec/`, `speedrun/grader.py`, `speedrun/sandbox_profile.py`: attempt execution and grading
    * `speedrun/implementer.py`, `speedrun/judge.py`, `speedrun/prompts/`: implementer, judge, and prompts
    * `speedrun/seed_archive/`, `speedrun/seed_archive_rescored_1500s/`: leaderboard records that seed the buffer

## Data 📊
* MLE-Bench: install [mle-bench](https://github.com/openai/mle-bench) (included in `requirements.txt`), prepare the 22 competitions in `mlebench/scripts/tasks.tsv` with `mlebench prepare -c <competition>` (Kaggle credentials required), and set `MLE_BENCH_DATA=~/.cache/mle-bench`, the directory that holds mle-bench's `data/`.
* NanoGPT: download the FineWeb-10B token shards with `data/cached_fineweb10B.py` from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and set `FINEWEB_DIR` to the directory with the `fineweb_{train,val}_*.bin` files.

## Experiments 🔬
### Contextual Bandits
From `bandit/`, run one instance with
```bash
python run_small_context.py E1_indep 10 32        # family, K, CPUs (|X| = 4)
python run_large_context.py E1_indep 100 16 32    # family, K, |X|, CPUs
```
or all instances with `sbatch sbatch_small_context.sbatch` and `sbatch sbatch_large_context.sbatch` (submitted from `bandit/`). Each run tunes the hyperparameters of every agent on 30 seeds, evaluates on 100 held-out seeds, and saves the regret curves to `results_small_context/` or `results_large_context/`.

### MLE-Bench Lite
From `mlebench/`,
```bash
bash scripts/preprocess.sh                       # build the prompts and parquet files
bash scripts/run.sh                              # 22 competitions x {rpg, vanilla}, one run after another
bash scripts/run.sh leaf-classification rpg      # or a single competition and arm
```
`scripts/run.sh` holds the complete training configuration; `REPS=3` repeats each run three times, as in the paper. Runs use all visible GPUs (two A100s in the paper).

### NanoGPT Speedrun
From `nanogpt/`, one arm runs a trainer (4 GPUs) and graders (8 GPUs each) that execute the attempts:
```bash
sbatch scripts/run.sbatch rr                      # or vanilla; 1 trainer + 5 graders on SLURM
bash scripts/run.sh trainer rr                    # without SLURM: on the trainer machine,
QUEUE_HOST=<trainer-host> bash scripts/run.sh grader   # and on each grader machine
```
The implementer and judge call an OpenAI-compatible endpoint: set `IMPL_API_KEY` (or `OPENAI_API_KEY`), and `IMPL_API_BASE` for a non-OpenAI endpoint. The seed records were timed on 8×A100-80GB graders; on other hardware, re-time them with `speedrun/rescore_seed_archive.py` and pass the result as `SEED_ARCHIVE`.

Training is logged to Weights & Biases (`wandb login` first).
