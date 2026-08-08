# 后端 lowering

Backend translator 接收同一 MLIR module 中的 Intent Kernel IR 与 Physical Plan，构造一个具体目标 kernel program。

Backend semantic path 全部位于 C++：`intent-compile` 先生成 Plan，由 Plan dialect verifier 验证 Plan 与 Kernel IR 的绑定，再调用选定 target emitter 发射源码。Python 只调用这一个 compiler 并运行已经生成的目标源码；它不构造 Plan，也不参与目标代码选择。

## 对应关系

| Intent | Triton | TileLang | cuTile | CPU/RVV |
|---|---|---|---|---|
| `@intent.kernel` | launch entry | `T.Kernel` entry | `@ct.kernel` | callable native kernel |
| logical region | block tensor shape | tiled buffer region | tile shape/index | loop/cache/vector region |
| ownership | `program_id`/grid | block bindings | `ct.bid`/grid | thread/task loops |
| logical view/index | pointer/load/store | buffer region/`T.copy` | array/tile load/store | address/stride loops |
| `contract` | `tl.dot` | `T.gemm` | MMA/tile matmul | FMA microkernel |
| `reduce` | `tl.max`/`tl.sum` | reduce instruction | tile reduction | vector/scalar reduction |
| logical validity | mask/tail | predicate | boundary handling | tail loop/`vsetvl` |
| Plan storage | compiler local/layout | shared/fragment | tile storage | register/cache/stack |
| Plan pipeline | stages/software pipeline | `T.Pipelined` | compiler/hints | prefetch/unroll |

## Emitter 职责

- materialize physical worker identity 与 ownership traversal；
- lower logical views/indexing 为目标地址与 boundary handling；
- lower structured nodes 为 target primitive 或复合 microkernel；
- realize storage、layout、packing 与 synchronization；
- realize pipeline、prefetch 与 launch attributes；
- 保持 source variable 与 region 的可识别命名。

Emitter 不选择新的 source algorithm。它实现 Physical Plan 已经选择的 realization。

Translator 开始前必须由 MLIR parser 读取 module，并验证 Kernel IR 与 Physical Plan。生成过程只按稳定 node ID 读取 Plan binding，并按 IR 的 def-use 与 region 结构发射代码；禁止接收旁路 Python Plan、根据 kernel 名称套用模板，或重新猜测 Python AST 中已经 lowering 的语义。Plan 缺失 binding、binding 指向错误 region，或目标尚未支持某个结构时，编译立即失败。

## 首个 Triton lowering

首个 stable-softmax realization 将 Plan 映射为一个 Triton entry：

- persistent ownership 生成 `program_id(0)`、`num_programs(0)` 与 grid-stride row traversal；
- column extent 生成 `next_power_of_2(n_cols)` 和 `tl.arange`；
- boundary binding 生成 `column < n_cols`、negative-infinity masked load 与 masked store；
- reduction/pointwise bindings按 Kernel IR def-use 生成 `tl.max`、subtract、`tl.exp`、`tl.sum` 与 divide；
- pipeline/launch binding 生成 stage policy、warps 与 occupancy-based program count；
- storage/layout binding 决定 row-major pointer relation 与 ABI read/write direction。

这是一个经过严格 verifier 约束的专用 emitter，不是“任意 Plan 字符串解释器”。当前代码直接读取 ownership worker axis、reduction axis、pipeline stages、launch warps、target/device/warp 与 ABI storage direction；其余字段只能取 stable-softmax verifier 接受的唯一值，例如 `next_power_of_two`、masked negative-infinity boundary、row-major layout 和 persistent occupancy。Emitter 因而发射与这些已验证值对应的固定结构。要支持第二种 tile、boundary、layout 或 traversal，必须先扩展 Plan legality 与 C++ emission 分支，不能让未识别值落入默认实现。

Translator 输出独立、可读的 Triton Python source，并将其编译成 callable entry。首个 translator 只接受上述 stable-softmax structure；其他 reduce、contract、ragged、atomic、state stream 或 control-flow realization 不会静默退回 Python 实现。

## 与下层 kernel 系统的关系

Intent 不重做 Triton、TileLang、cuTile 或 native compiler 已经成熟的 instruction、layout、pipeline 与 machine code generation。它决定应构造怎样的 target kernel program，再由下层映射到具体机器。

下层增加新指令、新 layout 或新 pipeline 能力时，同一个 Intent source 可以重新 realize 并继承这些能力。

## 参考生态

- [Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)
- [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton `tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
- [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [TileLang Language Basics](https://tilelang.com/programming_guides/language_basics.html)
- [cuTile Execution Model](https://docs.nvidia.com/cuda/cutile-python/execution.html)
