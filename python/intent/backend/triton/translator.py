from __future__ import annotations

import subprocess
from pathlib import Path


def translate_mlir(module_text: str, translator: str | Path) -> str:
    executable = Path(translator)
    if not executable.is_file():
        raise FileNotFoundError(f"Intent Triton translator does not exist: {executable}")
    completed = subprocess.run(
        [str(executable), "-"],
        input=module_text,
        text=True,
        capture_output=True,
        check=True,
    )
    if not completed.stdout:
        raise RuntimeError("Intent Triton translator emitted no source")
    return completed.stdout
