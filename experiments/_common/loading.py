from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    if name in sys.modules:
        loaded = sys.modules[name]
        loaded_path = Path(getattr(loaded, "__file__", ""))
        if loaded_path != path:
            raise ImportError(
                f"module name {name!r} already belongs to {loaded_path}, not {path}"
            )
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
