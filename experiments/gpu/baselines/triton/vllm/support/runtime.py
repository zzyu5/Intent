import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl


class _CurrentPlatform:
    def is_rocm(self):
        return False

    def is_arch_support_pdl(self):
        return torch.cuda.get_device_capability()[0] >= 9


def load_source(path: Path, module_name: str):
    vllm = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    vllm.__path__ = []
    platforms = sys.modules.setdefault("vllm.platforms", types.ModuleType("vllm.platforms"))
    platforms.current_platform = _CurrentPlatform()
    triton_utils = sys.modules.setdefault("vllm.triton_utils", types.ModuleType("vllm.triton_utils"))
    triton_utils.triton = triton
    triton_utils.tl = tl
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
