from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedMojoTarget:
    executable: str
    triple: str
    cpu: str
    features: str
    vector_bits: int
    workers: int
    build_threads: int

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return ("--target=mojo", f"--cpu-vector-bits={self.vector_bits}", f"--cpu-workers={self.workers}")

    @property
    def compiler_role(self) -> str:
        return "Intent CPU compiler"

    @property
    def native_options(self) -> tuple[str, ...]:
        return (
            f"--target-triple={self.triple}", f"--target-cpu={self.cpu}",
            f"--target-features={self.features}", "--fp-mode=contract=off",
            f"--num-threads={self.build_threads}", "-O3",
        )

    def materialize(self, source: str, module_text: str, entry_name: str,
                    metadata: dict[str, object]) -> CompiledArtifact:
        from intent.runtime.mojo import materialize_mojo_artifact
        return materialize_mojo_artifact(source, module_text, metadata, self)


@dataclass(frozen=True, slots=True)
class MojoTarget:
    workers: int = 8
    compiler: str | Path | None = None
    build_threads: int = 4

    def resolve(self) -> ResolvedMojoTarget:
        if self.workers <= 0 or self.build_threads <= 0:
            raise ValueError("Mojo worker and compiler thread budgets must be positive")
        selected = str(self.compiler) if self.compiler is not None else os.environ.get("INTENT_MOJO", "mojo")
        executable = shutil.which(selected)
        if executable is None:
            raise FileNotFoundError("Mojo compiler was not found; set INTENT_MOJO or MojoTarget(compiler=...)")
        result = subprocess.run([executable, "build", "--print-effective-target"],
                                text=True, capture_output=True, check=True)
        configuration = {}
        for line in result.stdout.splitlines():
            if line.strip().startswith("--target-"):
                key, value = line.strip().split(maxsplit=1)
                configuration[key] = value
        triple = configuration["--target-triple"]
        if not triple.startswith("x86_64-") or "linux" not in triple:
            raise NotImplementedError("Mojo CPU currently requires Linux x86-64")
        features = configuration["--target-features"]
        feature_set = set(features.split(","))
        if "+avx512f" in feature_set:
            vector_bits = 512
        elif "+avx2" in feature_set:
            vector_bits = 256
        else:
            raise NotImplementedError("Mojo CPU currently requires AVX2 or AVX512")
        return ResolvedMojoTarget(executable, triple, configuration["--target-cpu"], features,
                                  vector_bits, self.workers, self.build_threads)
