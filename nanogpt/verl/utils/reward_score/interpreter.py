import logging
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import DataClassJsonMixin  # optional if you want to .to_json()
import contextlib
import io
import humanize
import subprocess
import resource
import psutil
import threading
import threadpoolctl

logger = logging.getLogger("aide")


class TimeoutException(Exception):
    """Custom exception for in-process timeouts."""
    pass


def timeout_handler(signum, frame):
    """This function is called when the OS alarm goes off."""
    raise TimeoutException("In-process execution timed out!")


def get_current_virtual_memory():
    return psutil.Process().memory_info().vms


def read_host_mem_state() -> tuple[int, int] | None:
    """
    (available_bytes, limit_bytes) for the memory budget this process runs under,
    or None if it cannot be determined. Prefers the memory cgroup limit and falls
    back to node-wide MemAvailable/MemTotal.
    """
    # cgroup v2: /proc/self/cgroup -> "0::<path>"
    try:
        with open("/proc/self/cgroup") as f:
            cg_path = None
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0":
                    cg_path = parts[2]
                    break
        if cg_path is not None:
            base = Path("/sys/fs/cgroup") / cg_path.lstrip("/")
            # walk up until a cgroup with a real limit is found (leaf cgroups
            # often report "max")
            d = base
            while str(d).startswith("/sys/fs/cgroup"):
                max_f, cur_f = d / "memory.max", d / "memory.current"
                if max_f.is_file() and cur_f.is_file():
                    raw = max_f.read_text().strip()
                    if raw != "max":
                        limit = int(raw)
                        current = int(cur_f.read_text().strip())
                        return max(0, limit - current), limit
                d = d.parent
    except (OSError, ValueError):
        pass
    # node-wide fallback
    try:
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except Exception:
        return None


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """
    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    # Line in the generated script where the final exception surfaced (deepest
    # script frame of the last traceback); None if not attributable.
    exc_line: int | None = None
    # Message text of the final exception (uncapped).
    exc_msg: str | None = None


class Interpreter:
    def __init__(
        self,
        working_dir: str | Path,
        timeout: int,
        code_mem_limit: int,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        steps_for_saving: int = 2,
        mode: str = "in-process",
        num_cpus: int = 1,
        mem_guard: bool = True,
        mem_guard_low_frac: float = 0.10,
        mem_guard_crit_frac: float = 0.04,
        mem_guard_floor_gb: float = 4.0,
    ):
        """
        Runs generated code either in the current process or in a subprocess.

        Memory guard (subprocess mode only): a watchdog thread polls free memory
        in the job's cgroup while the child runs and kills the child before the
        trainer is OOM-killed; the sample then grades as a MemoryError.
          - free < mem_guard_low_frac of the limit: kill the child if its process
            tree holds more than mem_guard_floor_gb RSS.
          - free < mem_guard_crit_frac: kill the child unconditionally.
        """
        assert mode in ["in-process", "subprocess"]
        self.mode = mode
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # The timeout is only enforced in subprocess mode.
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.steps_for_saving = steps_for_saving
        self.code_mem_limit = code_mem_limit
        self.mem_guard = mem_guard
        self.mem_guard_low_frac = mem_guard_low_frac
        self.mem_guard_crit_frac = mem_guard_crit_frac
        self.mem_guard_floor_bytes = int(mem_guard_floor_gb * 1024 ** 3)
        # Thread cap for BLAS/OpenMP pools in the executed code. Ray's num_cpus
        # can be fractional, so round up to at least 1.
        self.num_cpus = max(1, math.ceil(num_cpus))

        # Keep a global scope so multiple runs can share variables,
        # or re-init in run() if reset_session=True
        self._global_scope = {}


    def _proc_tree_rss(self, pid: int) -> int:
        """RSS of pid plus all descendants (joblib/loky workers count too)."""
        total = 0
        try:
            p = psutil.Process(pid)
            for proc in [p] + p.children(recursive=True):
                try:
                    total += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
        return total

    def _start_mem_guard(self, proc, poll_interval: float = 0.5):
        """
        Watchdog thread that SIGKILLs the child's process group when the job's
        memory runs low. Returns (thread, state); state['killed'] holds the reason
        if the guard fired. The thread exits when the child does.
        """
        state = {"killed": None}
        if not self.mem_guard:
            return None, state
        mem0 = read_host_mem_state()
        if mem0 is None:
            logger.warning("mem guard: cannot read memory state; guard disabled")
            return None, state
        limit = mem0[1]
        low_bytes = int(limit * self.mem_guard_low_frac)
        crit_bytes = int(limit * self.mem_guard_crit_frac)

        def _watch():
            while proc.poll() is None:
                st = read_host_mem_state()
                if st is None:
                    return
                avail = st[0]
                if avail >= low_bytes:
                    time.sleep(poll_interval)
                    continue
                rss = self._proc_tree_rss(proc.pid)
                critical = avail < crit_bytes
                # Below the low-water mark, kill only samples holding memory;
                # below the critical mark, kill unconditionally.
                if critical or rss > self.mem_guard_floor_bytes:
                    state["killed"] = (
                        f"host memory guard: {avail / 1024**3:.2f} GB free of "
                        f"{limit / 1024**3:.2f} GB, this sample held "
                        f"{rss / 1024**3:.2f} GB")
                    logger.warning("mem guard killing sample: %s", state["killed"])
                    try:
                        os.killpg(proc.pid, 9)
                    except (ProcessLookupError, PermissionError):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return
                time.sleep(poll_interval)

        t = threading.Thread(target=_watch, daemon=True)
        t.start()
        return t, state

    def cleanup_session(self):
        """
        No-op; there is no persistent session to terminate.
        """

        pass
    
    def run(self, code: str, step: int, reset_session: bool = True) -> ExecutionResult:
        if self.mode == "in-process":
            return self._run_in_process(code, step, reset_session)
        else:
            return self._run_subprocess(code, step)
    
    
    def _save_code_if_needed(self, step, runfile_path, code):
        if step % self.steps_for_saving == 0:
            with open(runfile_path, "w", encoding="utf-8") as f:
                f.write(code)
    
    
    def _constraint_gpus(self):
        pass    # no constraint for now
    
    
    # Matches the exception class on a traceback's final line, e.g.
    # "ValueError: bad input" or "sklearn.exceptions.NotFittedError: ...".
    _EXC_LINE = re.compile(
        r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'
        r'(?:Error|Exception|Interrupt|Exit|Warning))\b')

    @classmethod
    def _parse_exc_type_from_output(cls, lines) -> str | None:
        """
        Best-effort exception class name from a child's traceback output (last
        matching line wins).
        """
        for line in reversed(lines):
            m = cls._EXC_LINE.match(line.strip())
            if m:
                # keep only the class name, not the module path
                return m.group(1).split('.')[-1]
        return None

    @classmethod
    def _parse_exc_msg_from_output(cls, lines) -> str | None:
        """
        Message text of the final exception: what follows 'Type: ' on the last
        traceback line (None when the exception line has no message).
        """
        for line in reversed(lines):
            s = line.strip()
            if cls._EXC_LINE.match(s):
                _, sep, msg = s.partition(': ')
                return msg.strip() if sep and msg.strip() else None
        return None

    # Matches a traceback frame header, e.g.
    #   File "/workdir/runfile.py", line 42, in <module>
    # (also SyntaxError's frame, which has no ", in <name>" part).
    _FRAME_LINE = re.compile(r'^File "([^"]+)", line (\d+)')

    @classmethod
    def _parse_exc_line_from_output(cls, lines, script_name) -> int | None:
        """
        Line number of the deepest frame in the generated script, from the last
        traceback in the output. Errors raised inside library calls report the
        script line that made the call.
        """
        found = None
        for line in lines:
            m = cls._FRAME_LINE.match(line.strip())
            if m and os.path.basename(m.group(1)) == script_name:
                found = int(m.group(2))
        return found

    def _run_subprocess(self, code: str, step: int) -> ExecutionResult:
        """
        Execute `code` in a fresh child interpreter with a per-execution timeout.
        `exec_time` is the child's wall-clock runtime.

          - CPU cap: thread-pool env vars plus LOKY_MAX_CPU_COUNT, so joblib
            `n_jobs=-1` spawns `num_cpus` workers.
          - Memory cap: RLIMIT_AS set in the child via preexec_fn.
          - The child gets its own process group, which is SIGKILLed on timeout
            so joblib/loky grandchildren do not outlive the sample.
        """
        self._constraint_gpus()

        runfile_path = self.working_dir / self.agent_file_name
        # The child runs the script by path, so tracebacks have real line numbers.
        with open(runfile_path, "w", encoding="utf-8") as f:
            f.write(code)

        child_env = os.environ.copy()
        n = str(self.num_cpus)
        child_env.update({
            "OMP_NUM_THREADS": n,
            "OPENBLAS_NUM_THREADS": n,
            "MKL_NUM_THREADS": n,
            "NUMEXPR_NUM_THREADS": n,
            "VECLIB_MAXIMUM_THREADS": n,
            "LOKY_MAX_CPU_COUNT": n,
        })

        mem_limit = self.code_mem_limit

        def _child_limits():
            # Runs in the child between fork and exec; bounds its address space.
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

        start_time = time.time()
        lines_out = []
        exc_type = None
        exc_info = None
        exc_stack = None
        exc_line = None
        exc_msg = None

        try:
            proc = subprocess.Popen(
                [sys.executable, str(runfile_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.working_dir),
                close_fds=True,
                start_new_session=True,
                env=child_env,
                preexec_fn=_child_limits,
            )
            # RLIMIT_AS caps each sample, but concurrent samples can still exhaust
            # the job's memory; the watchdog kills the child first.
            guard_thread, guard_state = self._start_mem_guard(proc)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                if stdout:
                    lines_out.extend(stdout.splitlines())
                if stderr:
                    lines_out.extend(stderr.splitlines())

                if proc.returncode != 0:
                    if guard_state["killed"]:
                        # Killed by the memory guard: report as MemoryError.
                        exc_type = "MemoryError"
                        exc_msg = guard_state["killed"]
                        exc_info = {"returncode": proc.returncode,
                                    "mem_guard": True}
                        lines_out.append(f"MemoryError: {exc_msg}")
                    else:
                        exc_type = (self._parse_exc_type_from_output(lines_out)
                                    or ("Killed" if proc.returncode == -9
                                        else "SubprocessCrash"))
                        exc_info = {"returncode": proc.returncode}
                        exc_line = self._parse_exc_line_from_output(
                            lines_out, self.agent_file_name)
                        exc_msg = self._parse_exc_msg_from_output(lines_out)

            except subprocess.TimeoutExpired:
                exc_type = "TimeoutError"
                # Kill the child's entire process group, including loky/joblib workers.
                try:
                    os.killpg(proc.pid, 9)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                stdout, stderr = proc.communicate()
                lines_out.append(
                    f"Error: Execution timed out after {self.timeout} seconds.")
                if stdout:
                    lines_out.extend(stdout.splitlines())
                if stderr:
                    lines_out.extend(stderr.splitlines())

        except Exception as e:
            # Launcher-side failure (couldn't even start the child)
            exc_type = e.__class__.__name__
            lines_out.append(f"Error: {str(e)}")

        # Wall-clock runtime of the child (about self.timeout for timed-out samples).
        exec_time = time.time() - start_time
        lines_out.append(f"Execution time: {humanize.naturaldelta(exec_time)}.")

        return ExecutionResult(
            term_out=lines_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
            exc_line=exc_line,
            exc_msg=exc_msg,
        )


    def _run_in_process(self, code: str, step: int, reset_session: bool = True) -> ExecutionResult:
        """
        Execute the provided Python code in this process, capturing stdout & stderr.

        Args:
            code (str): Python code to execute.
            reset_session (bool): If True, re-initialize the global scope each time.

        Returns:
            ExecutionResult: Object containing stdout/stderr, exec time, exception info, etc.
        """
        self._constraint_gpus()
        soft_mem_limit, hard_mem_limit = resource.getrlimit(resource.RLIMIT_AS)
        curr_vmem = get_current_virtual_memory()
        
        logger.info(f"Running code in-process (reset_session={reset_session}).")

        if reset_session:
            self._global_scope = {}

        # Optionally write code to a file in working_dir
        runfile_path = self.working_dir / self.agent_file_name
        
        self._save_code_if_needed(step, runfile_path, code)

        start_time = time.time()
        out_buf = io.StringIO()

        exc_type = None
        exc_info = None
        exc_stack = None
        exc_line = None
        exc_msg = None
        lines_out = []

        # set memory limit to be lower to avoid OOM crash due to generated LLM code
        resource.setrlimit(resource.RLIMIT_AS, (self.code_mem_limit + curr_vmem, hard_mem_limit))

        try:
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(out_buf):
                # Temporarily switch current working directory so user code writes 
                # local files to self.working_dir
                old_cwd = os.getcwd()
                try:
                    os.chdir(str(self.working_dir))
                    compiled_code = compile(code, filename=str(runfile_path), mode="exec")
                    with threadpoolctl.threadpool_limits(limits=self.num_cpus):
                        exec(compiled_code, self._global_scope)
                finally:
                    os.chdir(old_cwd)

        except MemoryError:
            exc_type = "MemoryError"
            error_msg = f"CRITICAL: Memory limit of {self.code_mem_limit / (1024**3):.2f} GB exceeded."
            print(error_msg)
            lines_out.append(error_msg)
            
        except BaseException as e:
            # Gather traceback info
            tb_str, e_cls_name, e_info, e_stack = self._format_exception(e, runfile_path)
            lines_out.append(tb_str)
            exc_type = e_cls_name
            exc_info = e_info
            exc_stack = e_stack
            exc_msg = str(e) or None
            # Deepest frame inside the generated script (code was compiled
            # with filename=runfile_path, so its frames carry that path).
            for fname, lineno, _, _ in e_stack:
                if os.path.basename(str(fname)) == self.agent_file_name:
                    exc_line = lineno

        finally:
            # Restore original memory limits
            resource.setrlimit(resource.RLIMIT_AS, (soft_mem_limit, hard_mem_limit))

        out_val = out_buf.getvalue()
        if out_val.strip():
            lines_out.extend(out_val.splitlines())

        exec_time = time.time() - start_time
        lines_out.append(f"Execution time: {humanize.naturaldelta(exec_time)}.")

        return ExecutionResult(
            term_out=lines_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
            exc_line=exc_line,
            exc_msg=exc_msg,
        )


    def _format_exception(self, e: BaseException, runfile_path: Path):
        """
        Format an exception stack trace (IPython style if self.format_tb_ipython).
        """
        if self.format_tb_ipython:
            import IPython.core.ultratb
            tb = IPython.core.ultratb.VerboseTB(color_scheme="NoColor")
            tb_str = tb.text(*sys.exc_info())
        else:
            tb_lines = traceback.format_exception(e.__class__, e, e.__traceback__)
            tb_str = "".join(tb_lines)

        # Replace the full path with just the filename
        tb_str = tb_str.replace(str(runfile_path), self.agent_file_name)

        e_cls_name = e.__class__.__name__
        exc_info = {}
        if hasattr(e, "args"):
            exc_info["args"] = [str(arg) for arg in e.args]

        tb = traceback.extract_tb(e.__traceback__)
        exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

        return tb_str, e_cls_name, exc_info, exc_stack
