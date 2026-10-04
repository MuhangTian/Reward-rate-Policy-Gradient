"""
Injected into every attempt via PYTHONPATH: caps per-GPU memory to the
official 80 GB budget when running on larger-memory GPUs.

GOLF_GPU_MEM_BYTES (int) sets the cap; when unset this module is a no-op.
Implemented as a post-import hook on torch, using
set_per_process_memory_fraction on each device.
"""
import os
import sys

_BYTES = os.environ.get("GOLF_GPU_MEM_BYTES")
if _BYTES:
    _BYTES = int(_BYTES)

    class _TorchMemCap:
        def find_module(self, name, path=None):  # legacy hook: simplest
            return None

        @staticmethod
        def apply(torch):
            # Do not trigger CUDA init here: the torchrun launcher forks rank
            # workers after importing torch, and initializing CUDA before the
            # fork deadlocks the children. Only cap processes whose CUDA
            # context already exists; is_initialized() does not initialize.
            if not torch.cuda.is_initialized():
                return
            for d in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(d).total_memory
                frac = min(_BYTES / total, 1.0)
                torch.cuda.set_per_process_memory_fraction(frac, d)
            torch._golf_memcap_done = True

    import builtins
    _orig_import = builtins.__import__

    def _hook(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if name == "torch" or name.startswith("torch."):
            torch = sys.modules.get("torch")
            if torch is not None and getattr(torch, "cuda", None) is not None \
                    and not getattr(torch, "_golf_memcap_done", False):
                try:
                    _TorchMemCap.apply(torch)   # no-op until CUDA is live
                except Exception:
                    pass  # cap re-attempted on next import
        return mod

    builtins.__import__ = _hook
