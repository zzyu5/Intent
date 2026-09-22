# Native ABI adapter for the installed Modular/MAX normalization library.
from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm
from max.gpu.host import DeviceContext
from std.os import abort
from std.runtime import initialize_runtime
from std.utils.index import IndexList


@export("max_rms_norm")
@no_inline
def max_rms_norm(
    x: Pointer[Float32, MutUntrackedOrigin], x_d0: Int64, x_d1: Int64, x_s0: Int64, x_s1: Int64,
    weight: Pointer[Float32, MutUntrackedOrigin], w_d0: Int64, w_s0: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], y_d0: Int64, y_d1: Int64, y_s0: Int64, y_s1: Int64,
    inverse_features: Float32, epsilon: Float32,
) abi("C"):
    initialize_runtime()
    var gamma = TileTensor(weight, row_major(Coord(Int(w_d0)))).as_immut()
    var shape = Coord(Int(x_d0), Int(x_d1))
    @always_inline
    def read[width: Int, alignment: Int](coordinate: Coord) capturing -> SIMD[DType.float32, width]:
        return x.unsafe_offset(Int(coordinate[0].value()) * Int(x_s0) + Int(coordinate[1].value())).unsafe_load[width=width]()
    @always_inline
    def write[width: SIMDLength, rank: Int, alignment: Int](coordinate: IndexList[rank], value: SIMD[DType.float32, width]) capturing:
        output.unsafe_offset(Int(coordinate[0]) * Int(y_s0) + Int(coordinate[1])).unsafe_store(value)
    try:
        var context = DeviceContext(api="cpu")
        rms_norm[DType.float32, 2, read, write, target="cpu"](
            shape, gamma, epsilon, Float32(0), context
        )
    except error:
        abort(String("MAX RMSNorm failed: ", error))
