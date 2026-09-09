# IntentDSL-authored CPU SIMD baseline; row partitioning follows the composition
# of parallel work and vector traversal in Modular's CPU elementwise reference.
from max.algorithm import parallelize
from std.ffi import external_call
from std.runtime import initialize_runtime


@export("source_affine")
@no_inline
def source_affine(
    a0: Pointer[Float32, MutUntrackedOrigin], a0_d0: Int64, a0_d1: Int64, a0_d2: Int64, a0_s0: Int64, a0_s1: Int64, a0_s2: Int64,
    a1: Pointer[Float32, MutUntrackedOrigin], a1_d0: Int64, a1_d1: Int64, a1_s0: Int64, a1_s1: Int64,
    a2: Pointer[Float32, MutUntrackedOrigin], a2_d0: Int64, a2_d1: Int64, a2_s0: Int64, a2_s1: Int64,
    a3: Pointer[Float32, MutUntrackedOrigin], a3_d0: Int64, a3_d1: Int64, a3_d2: Int64, a3_s0: Int64, a3_s1: Int64, a3_s2: Int64,
) abi("C"):
    initialize_runtime()
    var environment = external_call["intent_cpu_enter_ieee", UInt32]()
    var columns = Int(a0_d2)

    def row_body(row: Int) {imm a0, imm a1, imm a2, imm a3, imm columns}:
        var saved = external_call["intent_cpu_enter_ieee", UInt32]()
        var scale = a1.unsafe_offset(row).unsafe_load()
        var bias = a2.unsafe_offset(row).unsafe_load()
        var scale_v = SIMD[DType.float32, 16](scale)
        var bias_v = SIMD[DType.float32, 16](bias)
        var full = columns // 16 * 16
        for column in range(0, full, 16):
            var position = row * columns + column
            var x = a0.unsafe_offset(position).unsafe_load[width=16]()
            a3.unsafe_offset(position).unsafe_store(x * scale_v + bias_v)
        for column in range(full, columns):
            var position = row * columns + column
            var x = a0.unsafe_offset(position).unsafe_load()
            a3.unsafe_offset(position).unsafe_store(x * scale + bias)
        external_call["intent_cpu_leave_ieee", NoneType](saved)

    parallelize(row_body, Int(a0_d0 * a0_d1), 8)
    external_call["intent_cpu_leave_ieee", NoneType](environment)
