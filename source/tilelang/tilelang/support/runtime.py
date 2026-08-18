import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_fla_linear_boundary():
    for name in ("fla", "fla.ops", "fla.modules"):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__path__ = []
    linear = sys.modules.setdefault("fla.ops.linear_attn", types.ModuleType("fla.ops.linear_attn"))
    linear.fused_chunk_linear_attn = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("reference-only FLA call"))
    l2norm = sys.modules.setdefault("fla.modules.l2norm", types.ModuleType("fla.modules.l2norm"))
    l2norm.l2norm_fwd = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("reference-only FLA call"))


def load_source(path: Path, name: str, *, aliases=(), needs_fla_linear=False):
    sys.path.insert(0, str(path.parent))
    if needs_fla_linear:
        _install_fla_linear_boundary()
    for alias, alias_path in aliases:
        _load(alias_path, alias)
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
        if "float4" in str(value.dtype):
            return f"shape={tuple(value.shape)} dtype={value.dtype} packed=True"
        return f"shape={tuple(value.shape)} dtype={value.dtype} mean={value.float().mean().item():.6f}"
    if isinstance(value, tuple):
        return " | ".join(describe(item) for item in value)
    return repr(value)
