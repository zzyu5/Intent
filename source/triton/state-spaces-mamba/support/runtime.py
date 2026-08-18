import sys
from pathlib import Path

import torch


def activate_upstream(path: Path) -> Path:
    root = next(parent for parent in path.parents if parent.name == "state-spaces-mamba")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


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
