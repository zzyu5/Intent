# Native ABI adapter for the installed Modular/MAX CPU Softmax library.
from layout import Coord, TileTensor, row_major
from nn.softmax import softmax
from std.os import abort
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of


@export("max_softmax")
@no_inline
def max_softmax(
    x: Pointer[Float32, MutUntrackedOrigin], x_d0: Int64, x_d1: Int64, x_s0: Int64, x_s1: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], y_d0: Int64, y_d1: Int64, y_s0: Int64, y_s1: Int64,
) abi("C"):
    initialize_runtime()
    var shape = Coord(Int(x_d0), Int(x_d1))
    var input_tile = TileTensor(x, row_major(shape)).as_immut()
    var result = TileTensor(output, row_major(Coord(Int(y_d0), Int(y_d1))))
    comptime width = simd_width_of[DType.float32]()

    try:
        softmax[DType.float32, width, 2](input_tile, result, 1)
    except error:
        abort(String("MAX Softmax failed: ", error))
