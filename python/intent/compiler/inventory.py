from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from intent.api import KernelDefinition
from intent.compiler.toolchain import CompilationStageError, run_shared_compiler
from intent.frontend import lower_to_mlir
from intent.language.annotations import ConstexprSpec
from intent.targets import TritonTarget


def _module_name(path: Path, import_root: Path) -> str:
    return ".".join(path.with_suffix("").relative_to(import_root).parts)


def _discover(kernel_root: Path) -> tuple[list[Path], list[KernelDefinition]]:
    import_root = kernel_root.parent
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    files = sorted(kernel_root.rglob("*.py"))
    definitions: list[KernelDefinition] = []
    for path in files:
        module = importlib.import_module(_module_name(path, import_root))
        definitions.extend(
            value
            for value in vars(module).values()
            if isinstance(value, KernelDefinition)
            and value.__module__ == module.__name__
        )
    definitions.sort(key=lambda definition: (definition.__module__, definition.__qualname__))
    return files, definitions


def _required_constexprs(definition: KernelDefinition) -> set[str]:
    return {
        name
        for name, parameter in definition.signature.parameters.items()
        if isinstance(parameter.annotation, ConstexprSpec)
        and parameter.default is inspect.Parameter.empty
    }


def _key(definition: KernelDefinition) -> str:
    return f"{definition.__module__}:{definition.__qualname__}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile the complete Intent kernel corpus through the shared GPU verifier"
    )
    parser.add_argument("--kernel-root", type=Path, required=True)
    parser.add_argument("--bindings-module", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    files, definitions = _discover(arguments.kernel_root.resolve())
    bindings_module = importlib.import_module(arguments.bindings_module)
    bindings = getattr(bindings_module, "SHARED_COMPILE_BINDINGS")
    if not isinstance(bindings, dict):
        raise TypeError("SHARED_COMPILE_BINDINGS must be a dictionary")

    required = {
        _key(definition): _required_constexprs(definition)
        for definition in definitions
        if _required_constexprs(definition)
    }
    missing = sorted(required.keys() - bindings.keys())
    extra = sorted(bindings.keys() - required.keys())
    mismatched = sorted(
        key
        for key, names in required.items()
        if key in bindings and set(bindings[key]) != names
    )
    if missing or extra or mismatched:
        for label, values in (
            ("missing", missing),
            ("extra", extra),
            ("mismatched", mismatched),
        ):
            for value in values:
                print(f"binding_{label}: {value}", file=sys.stderr)
        return 2

    resolved = TritonTarget(arguments.device).resolve()

    def compile_one(definition: KernelDefinition) -> tuple[str, str | None]:
        key = _key(definition)
        try:
            module_text = lower_to_mlir(definition, constexprs=bindings.get(key))
            run_shared_compiler(
                arguments.compiler,
                module_text,
                resolved.compiler_options,
                resolved.compiler_role,
            )
        except CompilationStageError as error:
            return key, f"{error.stage}: {error}"
        except Exception as error:
            return key, f"frontend_kir: {error}"
        return key, None

    failures: list[tuple[str, str]] = []
    workers = max(1, arguments.jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(compile_one, definition) for definition in definitions}
        for future in as_completed(futures):
            key, error = future.result()
            if error is not None:
                failures.append((key, error))
    failures.sort()
    for key, error in failures:
        detail = error if arguments.verbose else error.splitlines()[0]
        print(f"failed: {key}: {detail}", file=sys.stderr)
    print(
        f"files={len(files)} kernels={len(definitions)} "
        f"passed={len(definitions) - len(failures)} failed={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
