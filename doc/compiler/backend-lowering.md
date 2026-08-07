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
