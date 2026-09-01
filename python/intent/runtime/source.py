from __future__ import annotations

import builtins as python_builtins
from itertools import count
import linecache

from .artifact import BackendIRCollector
from .artifact import CompiledArtifact


_MATERIALIZATION_IDS = count()


def materialize_python_source(
    *,
    target_name: str,
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
    backend_ir_collector: BackendIRCollector | None,
) -> CompiledArtifact:
    materialization_id = next(_MATERIALIZATION_IDS)
    module_name = (
        f"intent.generated.{target_name}.{entry_name}.{materialization_id}"
    )
    filename = f"<{module_name}>"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    namespace: dict[str, object] = {
        "__name__": module_name,
    }
    code = python_builtins.compile(source, filename, "exec", dont_inherit=True)
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
        device=device,
        _launcher=launcher,
        _runner=runner,
        _backend_ir_collector=backend_ir_collector,
        _namespace=namespace,
    )
