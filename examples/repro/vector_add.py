from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import intent
import intent.language as I


ELEMENTS = 1 << 20


@intent.kernel
def vector_add(
    lhs: I.In[I.f32, (ELEMENTS,)],
    rhs: I.In[I.f32, (ELEMENTS,)],
    output: I.Out[I.f32, (ELEMENTS,)],
):
    for index in range(ELEMENTS):
        output[index] = lhs[index] + rhs[index]


def _launcher() -> str:
    return f"""
  func.func @main() -> i32 {{
    %lhs = memref.alloc() : memref<{ELEMENTS}xf32>
    %rhs = memref.alloc() : memref<{ELEMENTS}xf32>
    %output = memref.alloc() : memref<{ELEMENTS}xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cN = arith.constant {ELEMENTS} : index
    %two = arith.constant 2.000000e+00 : f32
    scf.for %index = %c0 to %cN step %c1 {{
      %integer = arith.index_cast %index : index to i64
      %left = arith.sitofp %integer : i64 to f32
      %right = arith.mulf %left, %two : f32
      memref.store %left, %lhs[%index] : memref<{ELEMENTS}xf32>
      memref.store %right, %rhs[%index] : memref<{ELEMENTS}xf32>
      scf.yield
    }}
    func.call @vector_add(%lhs, %rhs, %output) :
      (memref<{ELEMENTS}xf32>, memref<{ELEMENTS}xf32>, memref<{ELEMENTS}xf32>) -> ()
    %true = arith.constant true
    %all = scf.for %index = %c0 to %cN step %c1 iter_args(%valid = %true) -> (i1) {{
      %integer = arith.index_cast %index : index to i64
      %left = arith.sitofp %integer : i64 to f32
      %right = arith.mulf %left, %two : f32
      %expected = arith.addf %left, %right : f32
      %actual = memref.load %output[%index] : memref<{ELEMENTS}xf32>
      %equal = arith.cmpf oeq, %actual, %expected : f32
      %next = arith.andi %valid, %equal : i1
      scf.yield %next : i1
    }}
    memref.dealloc %lhs : memref<{ELEMENTS}xf32>
    memref.dealloc %rhs : memref<{ELEMENTS}xf32>
    memref.dealloc %output : memref<{ELEMENTS}xf32>
    %success = arith.constant 0 : i32
    %failure = arith.constant 1 : i32
    %status = arith.select %all, %success, %failure : i32
    return %status : i32
  }}
"""


def _insert_launcher(module: str) -> str:
    closing_brace = module.rfind("}")
    if closing_brace < 0:
        raise ValueError("lowered MLIR module has no closing brace")
    return module[:closing_brace] + _launcher() + module[closing_brace:]


def _run(command: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent-opt", default="/tmp/intentdsl-build/tools/intent-opt/intent-opt"
    )
    parser.add_argument("--mlir-opt", default="/usr/lib/llvm-20/bin/mlir-opt")
    parser.add_argument(
        "--mlir-translate", default="/usr/lib/llvm-20/bin/mlir-translate"
    )
    parser.add_argument("--clang", default="/usr/lib/llvm-20/bin/clang")
    arguments = parser.parse_args()

    intent_mlir = intent.lower_to_mlir(vector_add)
    print("=== Intent MLIR ===")
    print(intent_mlir, end="")

    scf_mlir = _run(
        [arguments.intent_opt, "--convert-intent-to-scf"], input_text=intent_mlir
    )
    executable_mlir = _insert_launcher(scf_mlir)
    llvm_mlir = _run(
        [
            arguments.mlir_opt,
            "--convert-scf-to-cf",
            "--convert-arith-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-func-to-llvm",
            "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts",
        ],
        input_text=executable_mlir,
    )
    llvm_ir = _run(
        [arguments.mlir_translate, "--mlir-to-llvmir"], input_text=llvm_mlir
    )

    with tempfile.TemporaryDirectory(prefix="intentdsl-repro-") as directory:
        root = Path(directory)
        llvm_path = root / "vector_add.ll"
        executable = root / "vector_add"
        llvm_path.write_text(llvm_ir)
        _run([arguments.clang, "-O2", str(llvm_path), "-o", str(executable)])
        _run([str(executable)])

    print(f"backend numerical comparison: PASS ({ELEMENTS} f32 elements)")


if __name__ == "__main__":
    main()
