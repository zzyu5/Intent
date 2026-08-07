# 后端 lowering

Backend emitter 接收 Kernel IR 与 Physical Plan，构造一个具体目标 kernel program。

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

Emitter 开始前必须验证 Kernel IR 与 Physical Plan。生成过程只按稳定 node ID 读取 Plan binding，并按 IR 的 def-use 与 region 结构发射代码；禁止根据 kernel 名称套用模板，也禁止重新猜测 Python AST 中已经 lowering 的语义。Plan 缺失 binding、binding 指向错误 region，或目标尚未支持某个结构时，编译立即失败。

## 首个 Triton lowering

一维 pointwise realization 将 Plan 映射为一个 Triton entry：

- Plan program ownership 生成 `tl.program_id(0)`；
- extent 与 launch block 生成 `program * BLOCK_SIZE + tl.arange(...)`；
- masked boundary 生成 `offsets < N_ELEMENTS`，并传给每个 load/store；
- global contiguous storage 与 source index relation生成 pointer offset；
- pointwise primitive binding 生成对应的 `+`、`-`、`*` 或 `/`；
- Plan launch 保存 grid、block size、warp 数和 stage 数，用户调用 artifact 时不再传 grid。

Emitter 输出独立、可读的 Triton Python source，并将其编译成 callable entry。当前没有 Plan binding 的 reduce、contract、ragged、atomic、state stream 或多维 address lowering 不会静默退回 Python 实现。

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
