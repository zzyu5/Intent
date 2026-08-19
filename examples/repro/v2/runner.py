from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import intent

from .measurement import evaluate
from .model import Context
from .model import ResultRow
from .providers import load_cases
from .registry import BY_PROVIDER


FIELDS = (
    "kernel",
    "case",
    "generated_p50_ms",
    "source_p50_ms",
    "ratio",
    "status",
)


def _target(provider: str):
    if provider == "triton":
        return intent.TritonTarget(device=0)
    if provider == "cutile":
        return intent.CuTileTarget(device=0)
    if provider == "tilelang":
        return intent.TileLangTarget(device=0)
    raise ValueError(f"unknown provider: {provider}")


def _write(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "kernel": row.kernel,
                    "case": row.case,
                    "generated_p50_ms": (
                        "" if row.generated_p50_ms is None else f"{row.generated_p50_ms:.6f}"
                    ),
                    "source_p50_ms": (
                        "" if row.source_p50_ms is None else f"{row.source_p50_ms:.6f}"
                    ),
                    "ratio": "" if row.ratio is None else f"{row.ratio:.6f}",
                    "status": row.status,
                }
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(BY_PROVIDER))
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", action="append")
    arguments = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    provider = arguments.provider
    selected = set(arguments.kernel or ())
    entries = tuple(
        entry
        for entry in BY_PROVIDER[provider]
        if not selected or entry.kernel in selected
    )
    if selected - {entry.kernel for entry in entries}:
        unknown = ", ".join(sorted(selected - {entry.kernel for entry in entries}))
        parser.error(f"unknown {provider} V2 kernel(s): {unknown}")

    project_root = Path(__file__).resolve().parents[3]
    context = Context(
        compiler=arguments.compiler,
        project_root=project_root,
        target=_target(provider),
        provider=provider,
    )
    cases = load_cases(provider)
    rows: list[ResultRow] = []
    for entry in entries:
        factory = cases.get(entry.kernel)
        if factory is None:
            raise NotImplementedError(
                f"{provider} V2 case adapter is missing for {entry.kernel}"
            )
        try:
            comparison = factory(context)
        except NotImplementedError as error:
            print(f"{provider}:{entry.kernel}: unsupported: {error}")
            rows.append(ResultRow(entry.kernel, entry.case, None, None, None, "unsupported"))
            _write(arguments.output, rows)
            continue
        except Exception as error:
            print(f"{provider}:{entry.kernel}: compile_failed: {error}")
            rows.append(ResultRow(entry.kernel, entry.case, None, None, None, "compile_failed"))
            _write(arguments.output, rows)
            continue
        try:
            generated_p50, source_p50 = evaluate(comparison)
        except RuntimeError as error:
            print(f"{provider}:{entry.kernel}: numerical_failed: {error}")
            rows.append(ResultRow(entry.kernel, entry.case, None, None, None, "numerical_failed"))
            _write(arguments.output, rows)
            continue
        ratio = generated_p50 / source_p50
        print(
            f"{provider}:{entry.kernel}: pass "
            f"generated={generated_p50:.6f} ms source={source_p50:.6f} ms ratio={ratio:.6f}"
        )
        rows.append(
            ResultRow(
                entry.kernel,
                entry.case,
                generated_p50,
                source_p50,
                ratio,
                "pass",
            )
        )
        _write(arguments.output, rows)


if __name__ == "__main__":
    main()

