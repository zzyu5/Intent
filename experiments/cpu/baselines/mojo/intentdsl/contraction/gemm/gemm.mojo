# IntentDSL-authored f32 SIMD baseline. Cache tiles, B packing and a 4x16
# register microkernel follow Modular's CPU GEMM organization, not its MAX API.
from max.algorithm import parallelize
from std.collections import Array
from std.ffi import external_call
from std.math import fma
from std.runtime import initialize_runtime


@export("source_gemm")
@no_inline
def source_gemm(
    a0: Pointer[Float32, MutUntrackedOrigin], a0_d0: Int64, a0_d1: Int64, a0_s0: Int64, a0_s1: Int64,
    a1: Pointer[Float32, MutUntrackedOrigin], a1_d0: Int64, a1_d1: Int64, a1_s0: Int64, a1_s1: Int64,
    a2: Pointer[Float32, MutUntrackedOrigin], a2_d0: Int64, a2_d1: Int64, a2_s0: Int64, a2_s1: Int64,
) abi("C"):
    initialize_runtime()
    var environment = external_call["intent_cpu_enter_ieee", UInt32]()
    var rows = Int(a0_d0)
    var depth = Int(a0_d1)
    var columns = Int(a1_d1)
    var n_tiles = columns // 64

    def tile_body(task: Int) {imm a0, imm a1, imm a2, imm rows, imm depth, imm columns, imm n_tiles}:
        var saved = external_call["intent_cpu_enter_ieee", UInt32]()
        var m_begin = task // n_tiles * 64
        var n_begin = task % n_tiles * 64
        var storage = Array[Float32, 2048](uninitialized=True)
        var packed = storage.unsafe_ptr()
        for row in range(m_begin, m_begin + 64):
            for column in range(n_begin, n_begin + 64, 16):
                a2.unsafe_offset(row * columns + column).unsafe_store(SIMD[DType.float32, 16](0))
        for k_begin in range(0, depth, 32):
            for k in range(32):
                for n in range(0, 64, 16):
                    var value = a1.unsafe_offset((k_begin + k) * columns + n_begin + n).unsafe_load[width=16]()
                    packed.unsafe_offset(k * 64 + n).unsafe_store(value)
            for m in range(m_begin, m_begin + 64, 4):
                for n in range(0, 64, 16):
                    var c0 = a2.unsafe_offset(m * columns + n_begin + n).unsafe_load[width=16]()
                    var c1 = a2.unsafe_offset((m + 1) * columns + n_begin + n).unsafe_load[width=16]()
                    var c2 = a2.unsafe_offset((m + 2) * columns + n_begin + n).unsafe_load[width=16]()
                    var c3 = a2.unsafe_offset((m + 3) * columns + n_begin + n).unsafe_load[width=16]()
                    for k in range(32):
                        var right = packed.unsafe_offset(k * 64 + n).unsafe_load[width=16]()
                        var left0 = SIMD[DType.float32, 16](a0.unsafe_offset(m * depth + k_begin + k).unsafe_load())
                        var left1 = SIMD[DType.float32, 16](a0.unsafe_offset((m + 1) * depth + k_begin + k).unsafe_load())
                        var left2 = SIMD[DType.float32, 16](a0.unsafe_offset((m + 2) * depth + k_begin + k).unsafe_load())
                        var left3 = SIMD[DType.float32, 16](a0.unsafe_offset((m + 3) * depth + k_begin + k).unsafe_load())
                        c0 = fma(left0, right, c0)
                        c1 = fma(left1, right, c1)
                        c2 = fma(left2, right, c2)
                        c3 = fma(left3, right, c3)
                    a2.unsafe_offset(m * columns + n_begin + n).unsafe_store(c0)
                    a2.unsafe_offset((m + 1) * columns + n_begin + n).unsafe_store(c1)
                    a2.unsafe_offset((m + 2) * columns + n_begin + n).unsafe_store(c2)
                    a2.unsafe_offset((m + 3) * columns + n_begin + n).unsafe_store(c3)
        external_call["intent_cpu_leave_ieee", NoneType](saved)

    parallelize(tile_body, rows // 64 * n_tiles, 8)
    external_call["intent_cpu_leave_ieee", NoneType](environment)
