# Native ABI adapter for the installed Modular/MAX linalg.matmul.cpu library.
from layout import Coord, TileTensor, row_major
from linalg.matmul.cpu import matmul
from std.os import abort
from std.runtime import initialize_runtime


@export("max_gemm")
@no_inline
def max_gemm(
    a: Pointer[Float32, MutUntrackedOrigin], a_d0: Int64, a_d1: Int64, a_s0: Int64, a_s1: Int64,
    b: Pointer[Float32, MutUntrackedOrigin], b_d0: Int64, b_d1: Int64, b_s0: Int64, b_s1: Int64,
    c: Pointer[Float32, MutUntrackedOrigin], c_d0: Int64, c_d1: Int64, c_s0: Int64, c_s1: Int64,
) abi("C"):
    initialize_runtime()
    var left = TileTensor(a, row_major(Coord(Int(a_d0), Int(a_d1)))).as_immut()
    var right = TileTensor(b, row_major(Coord(Int(b_d0), Int(b_d1)))).as_immut()
    var output = TileTensor(c, row_major(Coord(Int(c_d0), Int(c_d1))))
    try:
        matmul[transpose_b=False, b_packed=False](
            output, left, right, kernel_type_m=Int(a_d0), num_threads=8
        )
    except error:
        abort(String("MAX GEMM failed: ", error))
