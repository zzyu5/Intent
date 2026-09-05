from .artifact import CompiledArtifact
from .source import materialize_python_source
from .tuning import TuningState


class TuningHooks:
    def __init__(self, names: tuple[str, ...], writable: tuple[bool, ...]):
        self.names = names
        self.writable = writable

    def before(self, arguments: dict, reset_only: bool = False) -> None:
        if not reset_only:
            self.state = TuningState(tuple(arguments[name] for name in self.names),
                                     self.writable)

    def after(self, arguments: dict, exception: Exception | None) -> None:
        self.state.restore()
        del self.state


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
    device: int,
) -> CompiledArtifact:
    return materialize_python_source(
        target_name="triton",
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=_collect_triton_ir,
    )
