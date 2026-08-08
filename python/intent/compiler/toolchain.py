from __future__ import annotations

import subprocess
from pathlib import Path

from intent.targets import ResolvedTritonTarget


def realize_mlir(
    kernel_mlir: str,
    realizer: str | Path,
    target: ResolvedTritonTarget,
) -> str:
    return _run(
        realizer,
        kernel_mlir,
        (
            f"--architecture={target.architecture}",
            f"--device={target.device}",
            f"--warp-size={target.warp_size}",
        ),
        "Intent realizer",
    )


def translate_mlir(module_text: str, translator: str | Path) -> str:
    return _run(translator, module_text, (), "Intent Triton translator")


def _run(
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
