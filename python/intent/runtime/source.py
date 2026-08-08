from __future__ import annotations

import builtins as python_builtins
import linecache

from .artifact import BackendIRCollector
from .artifact import CompiledArtifact


def materialize_python_source(
    *,
    target_name: str,
    source: str,
    module_text: str,
    entry_name: str,
    backend_ir_collector: BackendIRCollector | None,
) -> CompiledArtifact:
    filename = f"<intent-{target_name}:{entry_name}>"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    namespace: dict[str, object] = {
        "__name__": f"intent.generated.{target_name}.{entry_name}",
    }
    code = python_builtins.compile(source, filename, "exec")
    exec(code, namespace)
    launcher = namespace.get("launch")
    runner = namespace.get("run")
    if not callable(launcher) or not callable(runner):
        raise RuntimeError(
            f"generated {target_name} source must define launch() and run()"
        )
    return CompiledArtifact(
        source=source,
        mlir=module_text,
        _launcher=launcher,
        _runner=runner,
        _backend_ir_collector=backend_ir_collector,
    )
