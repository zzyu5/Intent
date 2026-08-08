from __future__ import annotations

import builtins as python_builtins
import linecache

from .artifact import CompiledArtifact


def materialize_triton_artifact(
    source: str,
    module_text: str,
    entry_name: str,
) -> CompiledArtifact:
    filename = f"<intent-triton:{entry_name}>"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    namespace: dict[str, object] = {
        "__name__": f"intent.generated.{entry_name}",
    }
    code = python_builtins.compile(source, filename, "exec")
    exec(code, namespace)
    launcher = namespace.get("launch")
    runner = namespace.get("run")
    if not callable(launcher) or not callable(runner):
        raise RuntimeError("generated Triton source must define launch() and run()")
    return CompiledArtifact(
        source=source,
        mlir=module_text,
        _launcher=launcher,
        _runner=runner,
    )
