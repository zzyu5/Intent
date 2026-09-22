# Same-moments LayerNorm through MAX rowwise library operations, not its Welford entry.
from algorithm import rowwise
from algorithm.reduce_op import ReduceSum
from layout import Coord
from max.gpu.host import DeviceContext
from std.math import rsqrt
from std.os import abort
from std.runtime import initialize_runtime
from std.utils.index import IndexList


@export("max_layer_norm_moments")
@no_inline
def max_layer_norm_moments(
    x: Pointer[Float32, MutUntrackedOrigin], x_d0: Int64, x_d1: Int64, x_s0: Int64, x_s1: Int64,
    weight: Pointer[Float32, MutUntrackedOrigin], w_d0: Int64, w_s0: Int64,
    bias: Pointer[Float32, MutUntrackedOrigin], b_d0: Int64, b_s0: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], y_d0: Int64, y_d1: Int64, y_s0: Int64, y_s1: Int64,
    inverse_features: Float32, epsilon: Float32,
) abi("C"):
    initialize_runtime()
    var shape = Coord(Int(x_d0), Int(x_d1))
    var axis_size = Int(x_d1)
    comptime width = rowwise.pick_simd_width[ReduceSum[DType.float32, 1], "cpu", 64, DType.float32]()

    @always_inline
    def body[params: rowwise.ContextParams](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var x, var x_s0, var weight, var bias, var output, var y_s0,
        var axis_size, var inverse_features, var epsilon
    }:
        comptime rank = row_coords.rank

        @always_inline
        def load[w: Int, alignment: Int, coord_rank: Int](index: IndexList[coord_rank]) {
            var x, var x_s0
        } -> SIMD[DType.float32, w]:
            var offset = Int(index[0]) * Int(x_s0) + Int(index[1])
            return x.unsafe_offset(offset).unsafe_load[width=w]()

        var row = rowwise.Row[params, DType.float32, DType.float32, 1, rank, is_cached=True](
            row_coords, axis_size, ctx, load
        )

        @always_inline
        def identity[w: Int](value: SIMD[DType.float32, w], index: IndexList[rank]) {} -> SIMD[DType.float32, w]:
            return value

        @always_inline
        def square[w: Int](value: SIMD[DType.float32, w], index: IndexList[rank]) {} -> SIMD[DType.float32, w]:
            return value * value

        var sum_x = row.reduce[ReduceSum[DType.float32, params.simd_width]](identity, load).acc[0]
        var sum_x2 = row.reduce[ReduceSum[DType.float32, params.simd_width]](square, load).acc[0]
        var mean = sum_x * inverse_features
        var variance = sum_x2 * inverse_features - mean * mean
        var inverse_std = rsqrt(variance + epsilon)

        @always_inline
        def write[w: Int](value: SIMD[DType.float32, w], index: IndexList[rank]) {
            var mean, var inverse_std, var weight, var bias, var output, var y_s0
        }:
            var column = Int(index[1])
            var gamma = weight.unsafe_offset(column).unsafe_load[width=w]()
            var beta = bias.unsafe_offset(column).unsafe_load[width=w]()
            var normalized = (value - SIMD[DType.float32, w](mean)) * SIMD[DType.float32, w](inverse_std)
            var offset = Int(index[0]) * Int(y_s0) + column
            output.unsafe_offset(offset).unsafe_store(normalized * gamma + beta)

        row.elementwise(write, load)

    try:
        var context = DeviceContext(api="cpu")
        rowwise.launch[axis=1, simd_width=width, target="cpu", num_phases=3](body, shape, context)
    except error:
        abort(String("MAX rowwise LayerNorm failed: ", error))
