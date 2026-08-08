from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def run_compiler(
    executable_path: str | Path,
    module_text: str,
    options: tuple[str, ...],
    role: str,
) -> tuple[str, str]:
    executable = Path(executable_path)
    if not executable.is_file():
        raise FileNotFoundError(f"{role} does not exist: {executable}")
    with tempfile.TemporaryDirectory(prefix="intentdsl-compile-") as directory:
        output_directory = Path(directory)
        source_path = output_directory / "kernel.py"
        mlir_path = output_directory / "kernel.mlir"
        completed = subprocess.run(
            [
                str(executable),
                *options,
                f"--source-output={source_path}",
                f"--ir-output={mlir_path}",
                "-",
            ],
            input=module_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{role} failed:\n{completed.stderr}")
        if not source_path.is_file() or not mlir_path.is_file():
            raise RuntimeError(f"{role} did not produce both compiler outputs")
        source = source_path.read_text(encoding="utf-8")
        realized_mlir = mlir_path.read_text(encoding="utf-8")
        if not source or not realized_mlir:
            raise RuntimeError(f"{role} produced an empty compiler output")
        return source, realized_mlir
