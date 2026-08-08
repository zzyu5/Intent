from .artifact import CompiledArtifact
from .source import materialize_python_source


def _collect_triton_ir(compiled_kernel: object) -> dict[str, str]:
    asm = getattr(compiled_kernel, "asm", None)
    if not isinstance(asm, dict):
        raise RuntimeError("Triton launch did not return a compiled kernel artifact")
    result = {name: value for name, value in asm.items() if isinstance(value, str)}
    if not result:
        raise RuntimeError("Triton compiled artifact exposes no textual backend IR")
    return result


def materialize_triton_artifact(
    source: str,
    module_text: str,
    entry_name: str,
) -> CompiledArtifact:
    return materialize_python_source(
        target_name="triton",
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        backend_ir_collector=_collect_triton_ir,
    )
