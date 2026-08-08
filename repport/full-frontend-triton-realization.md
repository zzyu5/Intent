# Frontend、Kernel IR 与首个 Triton Realization

## 本轮结论

当前正式编译路径已经变为：

```text
Python DSL
  -> typed Python Kernel IR
  -> Intent Kernel MLIR
  -> stable-softmax Realizer
  -> Physical Plan MLIR
  -> MLIR parse + Kernel/Plan verify
  -> C++ Triton source translator
  -> Triton JIT
  -> callable artifact
```

最终目标代码是 Triton Python source，不是 Triton MLIR。Backend translator 只接收同一 MLIR module 中的 Kernel IR 与 Physical Plan；它不接收 Python Kernel IR、Python Plan 或 AST 旁路输入。

“完整 frontend”在这里严格指文档定义的 Core 构造具有闭合表示链，不表示所有算法、所有组合或所有后端都已经实现。Physical Plan、realizer 与 backend 本轮只完成一个真实 stable-softmax realization。

## Intent MLIR 的身份

当前 Intent dialect 是本项目重新定义的干净 Kernel IR 边界，不是 TianchenIR 的复制改名。二者共享 MLIR/TableGen 技术栈与少量常见 operation 名称，但 schema 和编译目标不同：

- 当前设计以 logical domain/region、parallel/ordered/state stream、logical view/index relation、structured tensor primitive 与 effects 为核心；
- TianchenIR 的既有 IntentIR JSON lowering 和具体 tensor-op 体系不是当前 backend 的依赖；
- 当前 backend 只消费本项目产生并验证的 Intent Kernel MLIR 与 `intent_plan` MLIR。

## Frontend 与 Kernel IR 的完成边界

文档 Core 已建立同一条实现链：

```text
Python DSL surface
  -> AST lowering
  -> typed Kernel IR node/type
  -> Python semantic verifier
  -> Intent MLIR serialization
  -> C++ Kernel IR boundary verifier
```

覆盖的 Core 类别包括：

- kernel/helper ABI、view kind、dtype、symbolic shape、runtime scalar 与 constexpr；
- domain/region/product/partition、parallel/ordered/state stream；
- structured if/for/while 与 SSA carry；
- tensor expression、broadcast、reshape、transpose、mask 与 cast；
- reduce、scan、contract 与 record state；
- index relation、gather/scatter 与 ragged membership；
- logical buffer、atomic、fence、effects 与 logical RNG identity。

MLIR 边界显式保存 parameter、operation result 和 nested-region block argument 的稳定 value ID 与 source name；operation 使用独立稳定 node ID。Verifier 检查 ID 唯一性、result/type/shape metadata、region/terminator schema、index payload、effects、ABI constraints 以及 operation 必需属性。

本轮 repro 中的 11 个代表 kernel 共同产生全部 50 个当前 Core opcode，并全部经过 MLIR parser 和 Kernel IR verifier。这证明 Core 构造存在闭合路径，但不证明每个构造的全部 dtype、rank、shape、axis、嵌套方式与非法输入都已穷举。

ODS 中仍有一批 operation 使用 `AnyType` 与 metadata 表达跨构造语义；当前严格语义由 typed Python IR verifier 和 C++ Kernel IR boundary verifier 共同守住。因此这里不把它表述成“所有语义都已编码为各 op 的原生 ODS verifier”。

## Physical Plan MLIR

`intent_plan` 是独立注册的 MLIR dialect。首个 Plan 保存：

- logical `M/N` extent 与 `one/next_power_of_two` tile；
- persistent program ownership 与 grid-stride traversal；
- ABI value 的 global storage、read/write access 与 row-major layout；
- max、broadcast、subtract、exp、sum、broadcast、divide 的 primitive binding；
- reduction axis 与 identity；
- masked column tail、`column < N` predicate 与 negative-infinity load fill；
- 2/4-stage policy、shared-memory threshold、warps 与 persistent-occupancy grid policy；
- target architecture/device/warp size；
- 对 Kernel IR operation/value 的稳定 ID 引用。

Plan verifier 会把 `M/N` 重新绑定到 input view 的 symbolic shape 和两个 source domain，检查 row/column index relation、ABI/storage/layout、primitive operator/axis/identity、完整 def-use、boundary、ownership、pipeline 与 launch 的一致性。

Plan 中保存的是 `persistent_occupancy` grid policy。具体 `num_programs` 依赖 Triton JIT 得到的 register/shared-memory usage，因此由生成的 runtime wrapper 在 warmup 后计算，并限制为不超过 `n_rows`；本轮没有把这个动态数值伪装成静态 Plan 字段。中间 tensor SSA 的实际 register/spill 分配同样留给 Triton 编译器，Plan 不虚构固定 register placement。

Realizer 当前只识别一个严格的 stable-softmax def-use pattern，并采用与上游实现相同的确定性 policy。搜索、候选生成、cost model、其他 tile/traversal 和其他 kernel realization 尚未完成。

## Triton source translation

`intent-translate` 使用 MLIR parser 读取 module，执行 Kernel IR verifier 与 MLIR/Plan verifier，再发射 Triton source。它不按函数名识别 softmax；realizer 和 translator 都验证实际 operation 类型、node binding、region、def-use、index relation 与 ABI flow。

首个 translator 是 stable-softmax 专用 lowering，不是通用 Triton backend。其具体生成关系为：

- Plan ownership 生成 `program_id(0)`、`num_programs(0)` 和 persistent grid-stride row loop；
- Plan column extent 与 boundary 生成 `tl.arange`、`column < n_cols`、masked load/store 和 negative-infinity fill；
- Kernel IR body 按 operation 顺序与 SSA def-use 生成 `tl.max`、subtract、`tl.exp`、`tl.sum` 和 divide；
- Plan primitive axis/identity、pipeline stage policy、warps、target 与 grid policy 参与生成；
- ABI storage 与 index relation 决定 input/output pointer direction、row stride 与 address relation；layout binding 在发射该 row-major relation 前必须通过 Plan verifier；
- parameter、region argument 和 SSA result 的 source name 由 MLIR metadata 进入生成源码。

不符合这一结构或含有尚未支持的 operation 时，translator 直接失败，不使用 Python fallback，也不换成另一种 softmax 算法。

## 真实 baseline 与结果

Baseline 来自未修改的上游文件：

```text
source/triton/triton/normalization/softmax/02-fused-softmax.py
```

Repro 直接加载其中的原始 `softmax_kernel` 与 `softmax` wrapper，排除文件后部自运行测试和绘图 benchmark。比较条件为：

- CUDA device 固定为 `cuda:0`，并在加载 baseline 前设为 active device；
- 两边在同一显式 CUDA stream 上运行；
- shape 均为 `(8192, 8192)`，dtype 均为 `f32`；
- source algorithm 均为 full-row max → subtract → exp → full-row sum → divide；
- 两边均测量会分配 output 的 wrapper，而不是纯 kernel latency；
- 数值比较在计时外；
- `triton.testing.do_bench` 使用 `warmup=100`、`rep=500`，报告 p50/p95。

最终一次手动 repro 的结果：

| 项目 | 原版 | 自动生成版 |
|---|---:|---:|
| max abs error vs `torch.softmax` | `1.862645149230957e-09` | `1.862645149230957e-09` |
| wrapper p50 | `0.3695 ms` | `0.3695 ms` |
| wrapper p95 | `0.3716 ms` | `0.3725 ms` |
| generated/original | — | `1.0001x` (p50), `1.0023x` (p95) |

这次测量中生成版与原版性能实质相同；约千分之几的差异不能据此解释为显著快慢。结果只代表当前 GPU、shape、dtype 与一次 benchmark run。

生成 artifact 同时取得 Triton source、TTIR、TTGIR、LLVM IR 和 PTX；这些低层 IR 来自生成 source 的真实 Triton JIT，不是另一路手写产物。

## 唯一手动 repro

```bash
bash examples/repro/run_frontend_softmax.sh
```

该命令依次构建 `intent-opt` 与 `intent-translate`，验证 11 个 frontend module，打印 stable-softmax Kernel IR + Plan MLIR 和自动生成的 Triton source，执行数值比较并测量原版/生成版 wrapper。

## 尚未完成

- 50-opcode 存在性覆盖不是组合语义穷举，也不是“覆盖所有算法逻辑”的证明；
- ODS 的 per-op 强类型约束仍需逐步替代部分 `AnyType + metadata`；
- realizer 搜索、cost model、候选剪枝与自动调优尚未建立；
- Physical Plan 目前只有 stable-softmax 可执行实例；
- Triton translator 尚未覆盖 GEMM、attention、MoE、通用 reduction、ragged、atomic、state stream 或一般控制流；
- TileLang、cuTile、CPU/RVV backend 尚未实现；
- 当前性能结果不是跨 shape、dtype、GPU 的完整评测。
