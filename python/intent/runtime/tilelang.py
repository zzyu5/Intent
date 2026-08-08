from .artifact import CompiledArtifact
from .source import materialize_python_source


def materialize_tilelang_artifact(
    source: str,
    module_text: str,
    entry_name: str,
) -> CompiledArtifact:
    return materialize_python_source(
        target_name="tilelang",
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        backend_ir_collector=None,
    )
