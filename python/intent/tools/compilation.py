"""Shared implementation for the public command line and compile MCP."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, is_dataclass
import importlib
import importlib.util
from itertools import count
import io
import os
from pathlib import Path
import shutil
import sys

from .backends import backend, make_target

_MODULE_IDS = count()


@contextmanager
def _definition(program: str, symbol: str):
    from intent.api import KernelDefinition

    if not symbol.isidentifier():
        raise ValueError("kernel must be a module-level Python identifier")
    path = Path(program).expanduser()
    module_name = None
    if path.suffix == ".py" or "/" in program:
        path = path.resolve(strict=True)
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("program must name an existing Python file")
        module_name = f"_intent_user_program_{next(_MODULE_IDS)}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load Python program {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        previous = list(sys.path)
        try:
            sys.path.insert(0, str(path.parent))
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
        finally:
            sys.path[:] = previous
    else:
        module = importlib.import_module(program)
    try:
        definition = getattr(module, symbol)
        if not isinstance(definition, KernelDefinition):
            raise TypeError(f"{program}:{symbol} is not an @intent.kernel definition")
        yield definition
    finally:
        if module_name is not None:
            del sys.modules[module_name]


def _files(directory: Path | None) -> dict[str, str]:
    if directory is None:
        return {}
    return {name: str(directory / name) for name in
            ("input.mlir", "kernel.mlir", "kernel.source", "artifact.json", "compiler.log")
            if (directory / name).is_file()}


def compile_request(program: str, kernel: str, target: str | None = None, *,
                    target_options: dict | None = None, constexprs: dict | None = None,
                    compiler: str | None = None, tuning_config: str | None = None,
                    materialize: bool = False, stage: str = "provider") -> dict:
    """Compile an existing definition without invoking its kernel.

    Loading the Python module executes its ordinary top-level host code. Native
    JIT/tuning may remain deferred after materialization, depending on provider.
    """
    import intent
    from intent.compiler.toolchain import CompilerStage

    current_stage = "request"
    transcript = io.StringIO()
    result = {"program": program, "kernel": kernel, "target": target, "requested_stage": stage}
    try:
        selected_stage = CompilerStage(stage)
        if selected_stage is CompilerStage.KIR:
            if target is not None or target_options or tuning_config is not None or materialize:
                raise ValueError("KIR compilation does not accept target, target options, tuning or materialization")
        elif target is None:
            raise ValueError("shared/provider compilation requires a target")
        if materialize and selected_stage is not CompilerStage.PROVIDER:
            raise ValueError("materialization requires provider compilation")
        current_stage = "program_loading"
        # User module prints must not corrupt JSON output or the MCP transport.
        with redirect_stdout(transcript), _definition(program, kernel) as definition:
            result["definition"] = asdict(definition.source)
            selected = None
            if target is not None:
                current_stage = "target_options"
                selected = make_target(target, dict(target_options or {}))
            current_stage = "compilation"
            if selected_stage is CompilerStage.PROVIDER:
                generated = intent.generate(definition, target=selected, compiler=compiler,
                                            constexprs=constexprs, tuning_config=tuning_config)
                result.update(cache_directory=str(generated.cache_directory),
                              files=_files(generated.cache_directory), metadata=generated.metadata)
                if materialize:
                    current_stage = "generated_source_materialization"
                    generated.materialize()
                result.update(status="materialized" if materialize else "generated")
            else:
                lowered = intent.compile_ir(definition, stage=selected_stage.value,
                                            target=selected, compiler=compiler,
                                            constexprs=constexprs, tuning_config=tuning_config)
                result.update(status="lowered", cache_directory=str(lowered.cache_directory),
                              files=_files(lowered.cache_directory))
            result["tool_invoked_kernel"] = False
    except (Exception, SystemExit) as error:
        directory = getattr(error, "cache_directory", None)
        if directory is not None:
            result.update(cache_directory=str(directory), files=_files(directory))
        diagnostic = {"type": type(error).__name__, "message": str(error)}
        cause = error
        while cause is not None:
            for name in ("stdout", "stderr"):
                value = getattr(cause, name, None)
                if isinstance(value, str) and value:
                    diagnostic.setdefault(name, value)
            detail = getattr(cause, "diagnostic", None)
            if detail is not None and is_dataclass(detail):
                diagnostic["frontend"] = asdict(detail)
                break
            cause = cause.__cause__
        result.update(status="error", stage=getattr(error, "stage", current_stage),
                      diagnostic=diagnostic, tool_invoked_kernel=False)
    if transcript.getvalue():
        result["program_stdout"] = transcript.getvalue()
    return result


def doctor(target: str, *, target_options: dict | None = None, compiler: str | None = None) -> dict:
    from intent.compiler.toolchain import compiler_info
    from intent.targets.base import ResolvedTarget

    description = backend(target)
    checks = []
    result = {"target": target, "scope": "Dependency and target resolution only; no kernel compilation or execution",
              "toolchain": description["toolchain"], "checks": checks}

    def check(name, action):
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                value = action()
        except Exception as error:
            checks.append({"name": name, "status": "error", "type": type(error).__name__, "message": str(error)})
            return None
        checks.append({"name": name, "status": "available", "detail": value})
        if output.getvalue():
            checks[-1]["stdout"] = output.getvalue()
        return value

    def python_module(name):
        module = importlib.import_module(name)
        return {"path": getattr(module, "__file__", None), "version": str(getattr(module, "__version__", "not declared"))}

    def inspect_compiler():
        facts = compiler_info(compiler)
        if target not in facts["providers"]:
            raise NotImplementedError(
                f"The selected Intent compiler does not contain {target!r}; "
                f"built providers: {', '.join(facts['providers'])}"
            )
        return facts

    information = check("intent compiler", inspect_compiler)
    executable = information["executable"] if information is not None else None
    if executable is not None and target != "bangc":
        def profiles():
            names = ("shared", target) if target in {"triton", "cutile"} else (target,)
            paths = [Path(executable).parent / "profiles" / f"{name}.json" for name in names]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Compiler profile is missing: {path}")
            return [str(path) for path in paths]
        check("compiler profiles", profiles)
    for name in description["modules"]:
        check(name, lambda name=name: python_module(name))
    if target == "cutile":
        def tile_compiler():
            from cuda.tile._compile import _find_compiler_bin
            return {"path": _find_compiler_bin().path, "resolver": "cuda.tile"}
        check("cuTile native compiler", tile_compiler)

    def resolve():
        value = make_target(target, dict(target_options or {})).resolve()
        result["callable_materialization"] = isinstance(value, ResolvedTarget)
        return {"compiler_options": list(value.compiler_options),
                "facts": asdict(value) if is_dataclass(value) else {}}

    check("target", resolve)
    if target == "bangc":
        def neuware():
            value = make_target(target, dict(target_options or {})).resolve()
            path = shutil.which(value.compiler or os.environ.get("INTENT_BANGC_CNCC") or
                                str(Path(value.neuware) / "bin/cncc"))
            if path is None:
                raise FileNotFoundError("BANG C compiler is unavailable; select compiler and neuware paths")
            library = Path(value.neuware) / "lib64/libcnrt.so"
            if not library.is_file():
                raise FileNotFoundError(f"CNRT runtime library is missing: {library}")
            return {"compiler": path, "runtime": str(library), "device_execution": "not checked"}
        check("NeuWare tools", neuware)
    result["status"] = "available" if all(item["status"] == "available" for item in checks) else "unavailable"
    return result
