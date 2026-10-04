import os
import gc
import json
import re
import functools
import black
import subprocess
from pathlib import Path
import uuid
from verl.utils.reward_score.interpreter import Interpreter
import ast
import sys
import ray
import torch
import asyncio
from verl import DataProto
import numpy as np
import time
import pickle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

allowed_ml_libs = {
    'statsmodels', 'pandas', 'timm', 'bayes_opt', 
    'sklearn', 'xgboost', 'numpy', 'torch', 
    'torchvision', 'lightgbm', 'torch_geometric',
}
global_instrumentor = None


# Clamp bound for diagnostic scores, below the float32 max (~3.4e38).
_F32_LIMIT = 1e38


def _finite_or_clamped(value, uid=None):
    """
    Keep a diagnostic score assignable to a float32 tensor.

    Grader metrics such as RMSE are unbounded, so NaN/inf/oversized values are
    clamped and logged.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.error("REWARD: non-numeric raw_grader_score %r (uid=%s)", value, uid)
        return 0.0
    if v != v:                                   # NaN
        logger.error("REWARD: NaN raw_grader_score (uid=%s) -> 0.0", uid)
        return 0.0
    if v > _F32_LIMIT or v < -_F32_LIMIT:
        clamped = _F32_LIMIT if v > 0 else -_F32_LIMIT
        logger.error("REWARD: raw_grader_score %r overflows float32 (uid=%s) -> %g",
                     v, uid, clamped)
        return clamped
    return v


# sys.stdlib_module_names is available in Python 3.10+
standard_libs = getattr(sys, "stdlib_module_names", set())


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def compute_instrumented_score(
    term_out: list[str], 
    step_score: float = 0.1,
    no_ml_penalty: float = -10.0,
    enable_env_reward: bool = True,
    ) -> tuple[float, str, int]:
    
    ml_algo, num_features = "-1", -1
    ml_algo_regex = r"MODEL APPROACH:\s*(.*?)(?=,\s*NUM FEATURES)"
    num_features_regex = r"NUM FEATURES:\s*(\d+)"
    
    if 'no model training detected!' in term_out:
        return no_ml_penalty, ml_algo, num_features
    
    milestones = {
        "libraries imported!": False,
        "data loaded!": False,
        "model defined!": False,
        "model trained!": False,
        "predictions made!": False
    }
    
    score = 0.0
    
    for line in term_out:
        
        if enable_env_reward:
            for phrase, already_scored in milestones.items():
                if not already_scored and re.search(phrase, line):
                    score += step_score
                    milestones[phrase] = True # Mark as claimed
        
        ml_match = re.search(ml_algo_regex, line)
        if ml_match:
            ml_algo = ml_match.group(1).strip()
        
        num_features_match = re.search(num_features_regex, line)
        if num_features_match:
            num_features = int(num_features_match.group(1).strip())
            
    return score, ml_algo, num_features


def code_uses_valid_imports(code: str) -> bool:
    """check if the code only imports from allowed_ml_libs or standard_libs"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False # If code doesn't parse, it's invalid anyway

    for node in ast.walk(tree):
        root_module = None
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split('.')[0]
        
        # Check the root module against our lists
        if root_module:
            if root_module not in allowed_ml_libs and root_module not in standard_libs:
                print(f"Library '{root_module}' is strictly prohibited.")
                return False
    
    return True


_EXC_MSG_CAP = 150     # chars of exception message shown to the model
_EXC_SRC_CAP = 100     # chars of the offending source line shown


def format_no_submission_error(res, raw_code: str | None) -> str:
    """
    Self-contained one-line error for the self-improve prompt: exception
    type, the line in the generated script where it surfaced, that line's
    source text, and a capped message -- actionable even when the shown
    previous code was truncated around that region.
    """
    parts = [f"no submission.csv, the error is {res.exc_type}"]
    line_no = getattr(res, "exc_line", None)
    if line_no:
        parts.append(f" at line {line_no}")
        src_lines = (raw_code or "").splitlines()
        if 0 < line_no <= len(src_lines):
            src = src_lines[line_no - 1].strip()
            if src:
                parts.append(f" (`{src[:_EXC_SRC_CAP]}`)")
    msg = getattr(res, "exc_msg", None)
    if msg:
        parts.append(f": {msg[:_EXC_MSG_CAP]}")
    return "".join(parts)


def is_nan(val):
    # If x is not equal to itself, it's a NaN
    return val != val


@functools.lru_cache(maxsize=None)
def _load_leaderboard_scores(competition_id: str) -> tuple:
    """
    Loads and caches the real Kaggle leaderboard scores for a competition
    """
    from mlebench.data import get_leaderboard
    from mlebench.registry import registry

    competition = registry.get_competition(competition_id)
    leaderboard = get_leaderboard(competition)
    return tuple(leaderboard["score"].dropna().to_numpy())


def compute_leaderboard_score(raw_score: float, competition_id: str, lower_is_better: bool) -> float:
    """
    True percentile rank of `raw_score` against the competition's real
    Kaggle leaderboard (mle-bench's `get_leaderboard`)
    """
    scores = np.array(_load_leaderboard_scores(competition_id))
    x = raw_score
    if lower_is_better:
        scores = -scores
        x = -x
    less = np.sum(scores < x)
    equal = np.sum(scores == x)
    return float((less + 0.5 * equal) / len(scores))


def compute_score(
    code: str, 
    raw_code: str,
    ground_truth: str, 
    workspace_dir: str,
    step: int,
    reward_config,
    timeout: int,
    k: int,
    scalar_fn: object,
    code_mem_limit: int,
    uid: str,
    steps_for_saving: int=1,
    baseline_score: float=None,
    num_cpus: int=1,
    ) -> dict:
    # runs the code in run_dir, which is unique to each sample in the batch
    run_dir = Path(workspace_dir) / uid
    
    interpreter = Interpreter(
        working_dir=run_dir,
        steps_for_saving=steps_for_saving,
        timeout=timeout,
        code_mem_limit=code_mem_limit,
        num_cpus=num_cpus,
        mode="subprocess",
    )

    # run the code with an interpreter, LLM's generated code should save a submission.csv
    res = interpreter.run(code, step)
    interpreter.cleanup_session()

    submission_file = run_dir / "submission.csv"

    if res.exc_type == "TimeoutError":
        ins_score, ml_algo_str, num_features = compute_instrumented_score(
            res.term_out,
            reward_config.env_step_score,
            reward_config.no_ml_penalty,
            reward_config.enable_env_reward,
        )
        return {
            "test_score": 0,
            "raw_grader_score": 0.0,
            "env_ins_score": reward_config.no_submission_penalty + ins_score,
            "exec_time": res.exec_time,
            "min_time": reward_config.min_time,
            "valid_submission": 0,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": raw_code,
            "uid": uid,
            "error": f"The script reached a timeout of {res.exec_time} seconds",
        }
    
    # get score for environment instrumentation
    ins_score, ml_algo_str, num_features = compute_instrumented_score(
        res.term_out, 
        reward_config.env_step_score, 
        reward_config.no_ml_penalty,
        reward_config.enable_env_reward,
    )
    
    # no submission file
    if not submission_file.exists():
        logger.info('REWARD: no submission.csv')
        return {
            "test_score": 0,
            "raw_grader_score": 0.0,
            "env_ins_score": reward_config.no_submission_penalty + ins_score,
            "exec_time": res.exec_time,
            "min_time": reward_config.min_time,
            "valid_submission": 0,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": raw_code,
            "uid": uid,
            "error": format_no_submission_error(res, raw_code),
        }

    cmd = f"mlebench grade-sample {submission_file} {ground_truth}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, close_fds=True)
    
    try:
        result = '{' + result.stderr.split('{')[1]
        result = json.loads(result)
    except:
        logger.error("REWARD: error parsing grade.json")
        return {
            "test_score": 0,
            "raw_grader_score": 0.0,
            "env_ins_score": reward_config.no_submission_penalty + ins_score,
            "exec_time": res.exec_time,
            "min_time": reward_config.min_time,
            "valid_submission": 0,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": raw_code,
            "uid": uid,
            "error": "error parsing grade.json, usually caused by mlebench prepare issues, not your fault",
        }

    # if there is a valid score (performance on test set), return the score
    if result["score"] and not is_nan(result["score"]):
        logger.info(f"REWARD: {result['score']}")

        raw_reward_score = result["score"]
        if getattr(reward_config, "use_leaderboard_score", False):
            try:
                raw_reward_score = compute_leaderboard_score(
                    result["score"], ground_truth, result["is_lower_better"],
                )
            except Exception as e:
                logger.warning(f"[leaderboard score] falling back to raw score: {e}")
                raw_reward_score = result["score"]

        test_score = scalar_fn(raw_reward_score)

        if baseline_score is not None and test_score < baseline_score:
            ins_score = ins_score - test_score + reward_config.no_submission_penalty

        return {
            "test_score": raw_reward_score,
            "scaled_test_score": test_score,
            "raw_grader_score": result["score"],
            "env_ins_score": ins_score,
            "exec_time": res.exec_time,
            "min_time": reward_config.min_time,
            "valid_submission": 1,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": raw_code,
            "uid": uid,
            "error": f"No error detected, you achieved a test score of {result['score']}",
        }
    
    logger.info("REWARD: has_submission OR valid_submission")
    return {
        "test_score": 0,
        "raw_grader_score": 0.0,
        "env_ins_score": reward_config.no_submission_penalty + ins_score,
        "exec_time": res.exec_time,
        "min_time": reward_config.min_time,
        "valid_submission": 0,
        "ml_algo_str": ml_algo_str,
        "num_features": num_features,
        "raw_code": raw_code,
        "uid": uid,
        "error": "The submission is invalid, check whether the predictions are in the correct format, and that the number of rows match",
    }


@ray.remote
class GeminiActor:
    
    def __init__(self):
        from verl.utils.reward_score.instrumentor import make_instrumentor
        self.instrumentor = make_instrumentor()
        logger.info(f"Worker {os.getpid()} initialized instrumentor "
                    f"{type(self.instrumentor).__name__} "
                    f"(model={self.instrumentor.model_name})")
    
    
    def process_code(self, code: str) -> str:
        return self.instrumentor.process_script(code)
        

@ray.remote
def extract_code_get_reward(
    data_item: DataProto, 
    reward_config,
    tokenizer,
) -> tuple[dict, str, str, str, str]:
    # Extract relevant pieces
    prompt_ids = data_item.batch['prompts']
    uid = data_item.non_tensor_batch['uid']
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses']
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

    # Decode
    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    solution_str = tokenizer.decode(sequences)
    
    ml_algo_str, num_features = "-1", -1

    # Remove everything before the first "Assistant:", which are basically the prompts
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        logger.warning("[WARN] wrong assistant format")
        return {
            "test_score": 0,
            "raw_grader_score": 0.0,
            "env_ins_score": reward_config.no_submission_penalty,
            "exec_time": reward_config.invalid_submission_time,
            "min_time": reward_config.min_time,
            "valid_submission": 0,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": "",
            "uid": uid,
            "error": "wrong formatting for the output",
        }, "", ground_truth, uid, ""

    # extract code and natural language text from model output
    raw_code = extract_code(solution_str)
    nl_text = extract_text_up_to_code(solution_str)

    if not raw_code or not nl_text:
        logger.warning("[WARN] wrong plan + code format")
        return {
            "test_score": 0,
            "raw_grader_score": 0.0,
            "env_ins_score": reward_config.no_submission_penalty,
            "exec_time": reward_config.invalid_submission_time,
            "min_time": reward_config.min_time,
            "valid_submission": 0,
            "ml_algo_str": ml_algo_str,
            "num_features": num_features,
            "raw_code": "",
            "uid": uid,
            "error": "wrong formatting for the output, no code or no description detected",
        }, "", ground_truth, uid, ""
    
    return None, raw_code, ground_truth, uid, nl_text
    

@ray.remote
def compute_single_reward(
    code,
    raw_code,
    ground_truth,
    workspace_dir,
    global_step,
    reward_config,
    baseline_score,
    timeout,
    code_mem_limit,
    k,
    scalar_fn,
    uid,
    num_cpus=1,
):
    return compute_score(
        code=code,
        raw_code=raw_code,
        ground_truth=ground_truth,
        step=global_step,
        workspace_dir=workspace_dir,
        reward_config=reward_config,
        baseline_score=baseline_score,
        timeout=timeout,
        k=k,
        scalar_fn=scalar_fn,
        code_mem_limit=code_mem_limit,
        uid=uid,
        num_cpus=num_cpus,
    )


class RewardManager():
    """The reward manager.
    """
    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        workspace_dir,
        reward_config,
        code_mem_limit_gb=16,
        timeout=300,
        baseline_score=None,
        buffer_size=None,
        num_cpus_per_sample: int = 1,
        num_gemini_actors: int = 2,
        scalar_fn_name: str = "identity",
        b_score: float = 0.0,
        b_start_step: int = 1,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # times to print for each data source
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self.scalar_fn = ScalarizingFunction(scalar_fn_name)
        self.reward_config = reward_config
        self.baseline_score = baseline_score
        self.buffer_size = buffer_size
        self.num_cpus_per_sample = num_cpus_per_sample
        self.num_gemini_actors = num_gemini_actors
        self.code_mem_limit_bytes = code_mem_limit_gb * 1024 ** 3
        
        if os.environ.get("MLE_INSTRUMENTOR", "genai").strip().lower() == "gateway":
            self.num_gemini_actors = int(os.environ.get("MLE_GATEWAY_ACTORS", "16"))
            _actor_cpus = 0.1
        else:
            _actor_cpus = 1
        self.gemini_actors = [
            GeminiActor.options(
                num_cpus=_actor_cpus,
                max_restarts=-1,       # -1 means restart infinitely if it crashes
            ).remote()
            for _ in range(self.num_gemini_actors)
        ]
        
        if self.baseline_score == "90th-percentile":
            assert self.buffer_size is not None, "must have buffer_size"
            self.reward_buffer = []
            self.start_step = b_start_step
            self.b_score = b_score
        else:
            self.b_score = None
            self.reward_buffer = None
            self.start_step = None
    
    
    def clip_time(self, exec_time):
        if self.reward_config.clip_time:
            return max(exec_time, self.reward_config.clip_time)
        else:
            return exec_time


    def penalty_time(self, result_dict, is_invalid_zeroed):
        """
        Time used to size the reward-rate penalty for this sample.
        """
        charged = result_dict["exec_time"]
        if is_invalid_zeroed:
            charged = self.timeout + (
                result_dict["exec_time"]
                if getattr(self.reward_config, 'invalid_penalty_add_exec_time', False)
                else 0.0)
        return self.clip_time(charged + result_dict["min_time"])
    
    
    def get_baseline_score(self):
        if self.baseline_score == '90th-percentile':
            return self.b_score
        elif isinstance(self.baseline_score, float):
            return self.baseline_score
        else:
            return None
    
    
    def update_reward_buffer(self, reward):
        if self.baseline_score == '90th-percentile':
            self.reward_buffer.append(reward)
    
    
    def clean_reward_buffer_and_update_b_score(self, global_step):
        if self.baseline_score == '90th-percentile':
            if global_step >= self.start_step + self.buffer_size:
                self.b_score = np.percentile(self.reward_buffer, 90, method='closest_observation')
                self.reward_buffer = []
                self.start_step = global_step
    
    
    def save_reward_buffer(self, path):
        data_dict = {
            'b_score': self.b_score,
            'reward_buffer': self.reward_buffer,
            'start_step': self.start_step,
        }
        with open(path, 'wb') as f:
            pickle.dump(data_dict, f)
            
    
    def load_reward_buffer(self, path):
        try:
            with open(path, 'rb') as f:
                data_dict = pickle.load(f)
            self.b_score = data_dict['b_score']
            self.reward_buffer = data_dict['reward_buffer']
            self.start_step = data_dict['start_step']

            logger.info(f"[Reward buffer] Loaded reward buffer from {path}, b_score: {self.b_score}, start_step: {self.start_step}, buffer size: {len(self.reward_buffer)}")
        except Exception as e:

            logger.warning("[Reward buffer] Failed to load reward buffer, starting fresh.")
    
    
    def get_error_result(self, type, uid, raw_code, exec_time=None) -> dict:
        ml_algo_str = "-1"
        num_features = -1
        
        if type == 'exception':
            return {
                "test_score": 0.0,
                "raw_grader_score": 0.0,
                "env_ins_score": self.reward_config.no_submission_penalty,
                "exec_time": self.reward_config.invalid_submission_time,
                "min_time": self.reward_config.min_time,
                "valid_submission": 0,
                "ml_algo_str": ml_algo_str,
                "num_features": num_features,
                "b_score": self.b_score,
                "raw_code": raw_code,
                "uid": uid,
                "error": "Exception is raised during reward computation",
            }
        elif type == "ray_task_error":
            return {
                "test_score": 0.0,
                "raw_grader_score": 0.0,
                "env_ins_score": self.reward_config.no_submission_penalty,
                "exec_time": self.reward_config.invalid_submission_time,
                "min_time": self.reward_config.min_time,
                "valid_submission": 0,
                "ml_algo_str": ml_algo_str,
                "num_features": num_features,
                "b_score": self.b_score,
                "raw_code": raw_code,
                "uid": uid,
                "error": "Ray task error",
            }
        else:
            raise ValueError()


    def select_prev_solution(self, raw_code, nl_text):
        """
        Selects what gets shown to the model as {previous_plan_code} in the
        next self-improve round
        """
        if self.reward_config.self_improve_type == "pure_code":
            return raw_code
        return nl_text


    def gemini_actor_idx(self, i):
        return i % self.num_gemini_actors
    
    
    def extract_code(self, data: DataProto) -> list[tuple]:
        task_refs = []
        # launch all tasks in parallel to extract code and compute failure rewards
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            ref = extract_code_get_reward.options(num_cpus=self.num_cpus_per_sample).remote(
                data_item,
                self.reward_config,
                self.tokenizer,
            )
            task_refs.append(ref)
        
        results = ray.get(task_refs)
        return results
    
    
    def gemini_comment_code(self, code_results: list[tuple]) -> list[tuple]:
        task_refs = {}
        results = [None] * len(code_results)
        
        for i, (failure_result, raw_code, ground_truth, uid, nl_text) in enumerate(code_results):
            if failure_result is not None:
                results[i] = (failure_result, "", "", ground_truth, uid, "")
            else:
                ref = self.gemini_actors[self.gemini_actor_idx(i)].process_code.remote(raw_code)
                task_refs[ref] = i
        
        unfinished_refs = list(task_refs.keys())
        while unfinished_refs:
            ready_refs, unfinished_refs = ray.wait(unfinished_refs, num_returns=1, timeout=1.0)
            for ref in ready_refs:
                idx = task_refs[ref]
                processed_code = ray.get(ref)
                _, raw_code, ground_truth, uid, nl_text = code_results[idx]
                results[idx] = (None, raw_code, processed_code, ground_truth, uid, nl_text)
            
        return results


    def compute_rewards(
        self, 
        num_data: int, 
        code_results: list[tuple],
        global_step: int, 
        baseline_score: float,
        k: int,
    ) -> list[dict]:        
        task_refs = {}
        results = []
        
        # Launch all tasks in parallel to compute reward
        for i in range(num_data):
            failure_result, raw_code, code, ground_truth, uid, nl_text = code_results[i]
            
            if failure_result is not None:
                results.append((i, failure_result))
            else:
                ref = compute_single_reward.options(
                        num_cpus=self.num_cpus_per_sample,
                        max_retries=0,  # no retry, we will handle errors in the main loop
                    ).remote(
                    code=code,
                    raw_code=self.select_prev_solution(raw_code, nl_text),
                    ground_truth=ground_truth,
                    workspace_dir=self.workspace_dir,
                    global_step=global_step,
                    reward_config=self.reward_config,
                    baseline_score=baseline_score,
                    timeout=self.timeout,
                    k=k,
                    scalar_fn=self.scalar_fn,
                    uid=uid,
                    code_mem_limit=self.code_mem_limit_bytes,
                    num_cpus=self.num_cpus_per_sample,
                )
                task_refs[ref] = i

        timeout_error_count = 0
        crashed_error_count = 0
        unknown_error_count = 0
        unfinished_refs = list(task_refs.keys())

        while unfinished_refs:
            ready_refs, unfinished_refs = ray.wait(
                unfinished_refs,
                num_returns=1,
                timeout=1.0
            )

            for ref in ready_refs:
                i = task_refs[ref]
                try:
                    result_dict = ray.get(ref)
                    results.append((i, result_dict))
                    if 'reached a timeout' in str(result_dict.get('error', '')):
                        timeout_error_count += 1

                except ray.exceptions.OutOfMemoryError as e:
                    logger.warning(f"Task exceeded memory limit ({self.code_mem_limit_bytes} bytes) at index {i}.")
                    _, raw_code, _, _, uid, nl_text = code_results[i]
                    results.append((i, self.get_error_result("ray_task_error", uid, self.select_prev_solution(raw_code, nl_text))))
                    crashed_error_count += 1

                except ray.exceptions.RayTaskError as e:
                    logger.warning(f"Ray Actor crashed or task failed with error: {e} for index {i}.")
                    _, raw_code, _, _, uid, nl_text = code_results[i]
                    results.append((i, self.get_error_result("ray_task_error", uid, self.select_prev_solution(raw_code, nl_text))))
                    crashed_error_count += 1

                except Exception as e:
                    logger.warning(f"Unknown error: {e} for index {i}.")
                    _, raw_code, _, _, uid, nl_text = code_results[i]
                    results.append((i, self.get_error_result("exception", uid, self.select_prev_solution(raw_code, nl_text))))
                    unknown_error_count += 1

        return results, crashed_error_count, unknown_error_count, timeout_error_count

    
    def init_gpu_heartbeat_actor(self):
        # Find the GPU node for the heartbeat actor
        gpu_node_id = None
        max_retries = 60  # Wait up to 60 seconds for the network to sync
        
        logger.info("Scanning Ray cluster for GPU node...")
        for attempt in range(max_retries):
            for node in ray.nodes():
                if node.get("Resources", {}).get("GPU", 0) > 0 and node.get("Alive"):
                    gpu_node_id = node["NodeID"]
                    break
            
            if gpu_node_id:
                logger.info(f"GPU node found on attempt {attempt + 1}. NodeID: {gpu_node_id}")
                break
                
            time.sleep(1)  # Pause and query again

        if gpu_node_id:
            # Force the heartbeat onto the GPU node using 0 Ray GPU resources
            self.heartbeat_actor = GPUHeartbeatActor.options(
                num_cpus=0.1,  
                num_gpus=0,    
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=gpu_node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}},
            ).remote()
        else:
            logger.error("[FATAL WARNING] GPU node never registered with Ray. Heartbeat failed.")
            self.heartbeat_actor = None
            
            
    def start_gpu_heartbeat(self):
        self.heartbeat_actor.start.remote()
    
    
    def stop_gpu_heartbeat(self):
        self.heartbeat_actor.stop.remote()
        logger.info("GPU heartbeat stopped.")
    
    
    def __call__(self, data: DataProto, global_step: int, k: int):
        # keep GPU utilization up while rewards are computed
        self.start_gpu_heartbeat()
        
        self.clean_reward_buffer_and_update_b_score(global_step)
        baseline_score = self.get_baseline_score()
        
        code_results = self.extract_code(data)
        assert len(code_results) == len(data), f"{len(code_results)} vs. {len(data)}"
        code_results = self.gemini_comment_code(code_results)
        
        results, crashed_error_count, unknown_error_count, timeout_error_count = \
            self.compute_rewards(
                len(data), code_results, global_step, baseline_score, k
            )

        # Prepare for final rewards
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        exec_time_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalty_time_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # (bsz, )
        raw_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        raw_exec_time_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        raw_grader_score_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        valid_submission_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        
        ml_str_lst = [None] * len(data)
        num_feature_lst = [None] * len(data)
        raw_code_lst = [None] * len(data)
        uid_lst = [None] * len(data)
        error_lst = [None] * len(data)
        valid_submissions = 0
        
        for i, result_dict in results:
            # Put the entire score at the last response token
            prompt_length = data[i].batch['prompts'].shape[-1]
            valid_response_length = data[i].batch['attention_mask'][prompt_length:].sum()
            is_invalid_zeroed = (
                getattr(self.reward_config, 'invalid_zero_reward_full_time', False)
                and not result_dict["valid_submission"]
            )
            if is_invalid_zeroed:
                result_dict = {
                    **result_dict,
                    "test_score": 0.0,
                    "scaled_test_score": 0.0,
                    "env_ins_score": getattr(
                        self.reward_config, 'invalid_reward_rate_penalty', 0.0),
                }

            # store raw results
            raw_reward_tensor[i] = result_dict["test_score"]
            raw_exec_time_tensor[i] = result_dict["exec_time"]

            raw_grader_score_tensor[i] = _finite_or_clamped(
                result_dict["raw_grader_score"], result_dict.get("uid", i))
            valid_submission_tensor[i] = result_dict["valid_submission"]

            scaled_test_score = get_test_score(result_dict)
            
            # store rewards and time for training
            reward_tensor[i, valid_response_length - 1] = scaled_test_score \
                + result_dict["env_ins_score"]
            exec_time_tensor[i, valid_response_length - 1] = self.clip_time(
                result_dict["exec_time"] + result_dict["min_time"])

            penalty_time_tensor[i, valid_response_length - 1] = \
                self.penalty_time(result_dict, is_invalid_zeroed)


            valid_submissions += result_dict["valid_submission"]
            ml_str_lst[i] = result_dict["ml_algo_str"]
            num_feature_lst[i] = result_dict["num_features"]
            raw_code_lst[i] = result_dict["raw_code"]
            uid_lst[i] = result_dict["uid"]
            error_lst[i] = result_dict["error"]
            
            self.update_reward_buffer(reward=scaled_test_score)

        self.stop_gpu_heartbeat()

        return {
            "raw_reward_tensor": raw_reward_tensor,
            "raw_exec_time_tensor": raw_exec_time_tensor,
            "raw_grader_score_tensor": raw_grader_score_tensor,
            "reward_tensor": reward_tensor,
            "exec_time_tensor": exec_time_tensor,
            "penalty_time_tensor": penalty_time_tensor,
            "valid_submission_tensor": valid_submission_tensor,
            "valid_submission": valid_submissions,
            "crashed_error_count": crashed_error_count,
            "unknown_error_count": unknown_error_count,
            "timeout_error_count": timeout_error_count,
            "ml_algo_str_lst": ml_str_lst,
            "num_feature_lst": num_feature_lst,
            "b_score": self.b_score,
            "raw_code_lst": raw_code_lst,
            "uid_lst": uid_lst,
            "error_lst": error_lst,
        }
        

def get_test_score(result_dict):
    if "scaled_test_score" in result_dict:
        return result_dict["scaled_test_score"]
    else:
        return result_dict["test_score"]


@ray.remote
class GPUHeartbeatActor:
    def __init__(self, dim=4096):
        logger.info(f"Available GPUs (by Ray): {os.environ['CUDA_VISIBLE_DEVICES']}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force using GPU 0 for this actor
        logger.info(f"Available GPUs (set by us): {os.environ['CUDA_VISIBLE_DEVICES']}")
        self.dim = dim
        self.running = False
        self.device = torch.device('cuda:0') 
        
    async def start(self):
        """Starts a background loop to keep GPU utilization high."""
        self.running = True
        
        A = torch.randn(self.dim, self.dim, device=self.device)
        B = torch.randn(self.dim, self.dim, device=self.device)
        
        while self.running:
            _ = torch.matmul(A, B)
            await asyncio.sleep(0)

    def stop(self):
        """Signals the background loop to terminate."""
        self.running = False


class ScalarizingFunction:
    def __init__(self, func_name: str, beta: float = 3.0):
        self.func_name = func_name
        self.beta = beta
    
    def __call__(self, val):
        if self.func_name == "identity":
            return val
        elif self.func_name == "neg_exp":
            return np.exp(-val / self.beta)
        elif self.func_name == "reciprocal":
            return 1 / (1 + val / self.beta)
        else:
            raise ValueError(f"Unknown scalarizing function: {self.func_name}")
        
    