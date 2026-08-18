import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_source(path: Path, name: str, *, needs_utils=False, needs_splitk=False, needs_gelu=False):
    if needs_utils:
        _load(ROOT / "support" / "utils.py", "tilegym.ops.cutile.utils")
    if needs_splitk:
        _load(ROOT / "attention" / "flash_decode" / "splitk_reduce.py", "tilegym.ops.cutile.splitk_reduce")
    if needs_gelu:
        _load(ROOT / "activation" / "fused" / "gelu.py", "tilegym.ops.cutile.activation.gelu")
    return _load(path, name)


def elapsed_ms(call):
    call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def describe(value):
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)} dtype={value.dtype} mean={value.float().mean().item():.6f}"
    if isinstance(value, tuple):
        return " | ".join(describe(item) if item is not None else "None" for item in value)
    return repr(value)
