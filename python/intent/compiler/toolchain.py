from __future__ import annotations

import subprocess
from pathlib import Path


def run_tool(
    executable_path: str | Path,
    module_text: str,
    options: tuple[str, ...],
    role: str,
) -> str:
    executable = Path(executable_path)
    if not executable.is_file():
        raise FileNotFoundError(f"{role} does not exist: {executable}")
    completed = subprocess.run(
        [str(executable), *options, "-"],
        input=module_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{role} failed:\n{completed.stderr}")
    if not completed.stdout:
        raise RuntimeError(f"{role} emitted no output")
    return completed.stdout
