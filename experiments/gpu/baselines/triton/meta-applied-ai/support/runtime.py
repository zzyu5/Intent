import importlib.util
import sys
import types
from pathlib import Path

import torch


def load_source(path: Path, module_name: str, *, stub_vllm: bool = False):
    if stub_vllm:
        vllm = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
        extension = sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))
        ops = sys.modules.setdefault("vllm._C.ops", types.ModuleType("vllm._C.ops"))
        extension.ops = ops
        vllm._C = extension
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def find_upstream_root(path: Path, name: str) -> Path:
    for parent in path.parents:
        if parent.name == name:
            return parent
    raise RuntimeError(f"cannot find {name} above {path}")
