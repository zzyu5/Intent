import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton


def _identity_decorator(*args, **kwargs):
    return lambda function: function


def _prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=None):
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist()
    pairs = [(sequence, chunk) for sequence, length in enumerate(lengths) for chunk in range((length + chunk_size - 1) // chunk_size)]
    return torch.tensor(pairs, device=cu_seqlens.device, dtype=torch.int32)


def _install_package_boundary():
    for name in ("fla", "fla.modules", "fla.modules.conv", "fla.modules.conv.triton", "fla.ops", "fla.ops.utils"):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__path__ = []
    backends = sys.modules.setdefault("fla.modules.backends", types.ModuleType("fla.modules.backends"))
    backends.dispatch = _identity_decorator
    utils = sys.modules.setdefault("fla.utils", types.ModuleType("fla.utils"))
    utils.IS_AMD = False
    utils.autotune_cache_kwargs = {}
    utils.input_guard = _identity_decorator
    sys.modules["fla.ops.utils"].prepare_chunk_indices = _prepare_chunk_indices
    cache = sys.modules.setdefault("fla.ops.utils.cache", types.ModuleType("fla.ops.utils.cache"))
    cache.fla_cache_autotune = lambda configs, key, **kwargs: triton.autotune(configs=configs, key=key)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_causal_conv(path: Path):
    _install_package_boundary()
    _load(path.with_name("kernels.py"), "fla.modules.conv.triton.kernels")
    return _load(path, "fla.modules.conv.triton.ops")


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
