# The installed MAX CPU elementwise implementation owns scheduling and vectorization.
from max.algorithm.backend.cpu.elementwise import _elementwise_impl_cpu
from std.runtime import initialize_runtime
from std.sys.info import simd_width_of
from std.utils.coord import Coord


@export("max_affine")
@no_inline
def max_affine(
    x: Pointer[Float32, MutUntrackedOrigin], x_d0: Int64, x_d1: Int64, x_d2: Int64,
    x_s0: Int64, x_s1: Int64, x_s2: Int64,
    scale: Pointer[Float32, MutUntrackedOrigin], s_d0: Int64, s_d1: Int64, s_s0: Int64, s_s1: Int64,
    bias: Pointer[Float32, MutUntrackedOrigin], b_d0: Int64, b_d1: Int64, b_s0: Int64, b_s1: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], y_d0: Int64, y_d1: Int64, y_d2: Int64,
    y_s0: Int64, y_s1: Int64, y_s2: Int64,
) abi("C"):
    initialize_runtime()
    @always_inline
    def body[width: Int, alignment: Int = 1](coordinate: Coord) {
        imm x, imm scale, imm bias, imm output, imm x_s0, imm x_s1,
        imm s_s0, imm s_s1, imm b_s0, imm b_s1, imm y_s0, imm y_s1
    }:
        var batch = Int(coordinate[0].value())
        var row = Int(coordinate[1].value())
        var column = Int(coordinate[2].value())
        var values = x.unsafe_offset(batch * Int(x_s0) + row * Int(x_s1) + column).unsafe_load[width=width]()
        var factor = scale.unsafe_offset(batch * Int(s_s0) + row * Int(s_s1)).unsafe_load()
        var shift = bias.unsafe_offset(batch * Int(b_s0) + row * Int(b_s1)).unsafe_load()
        var result = values * SIMD[DType.float32, width](factor) + SIMD[DType.float32, width](shift)
        output.unsafe_offset(batch * Int(y_s0) + row * Int(y_s1) + column).unsafe_store(result)
    _elementwise_impl_cpu[
        simd_width_of[DType.float32](), trace_description="batched-affine"
    ](body, shape=Coord(Int(x_d0), Int(x_d1), Int(x_d2)))
