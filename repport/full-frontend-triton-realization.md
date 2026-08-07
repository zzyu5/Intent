# 完整前端与首个 Triton realization 里程碑

## 结论

当前项目已经形成一条真实的算子编译链：

```text
Python eDSL
  -> Frontend
  -> verified Kernel IR
  -> Intent MLIR
  -> Realizer
  -> verified Physical Plan
  -> Triton source / callable artifact
  -> CUDA numerical execution
```

“完整前端”在本项目中的准确含义是：当前语言设计中的全部 50 个 `OpCode` 都有真实 Python DSL producer，生成的 Kernel IR 通过统一 verifier，并可发射为由 Intent dialect parser/verifier 接受的 MLIR。它不表示每一种参数组合都已经枚举，也不表示这 50 个节点都已有 Triton lowering。

## 前端覆盖

唯一 repro 对以下 11 个 DSL kernel 分别执行 frontend lowering、Kernel IR verification、Intent MLIR emission 与 `intent-opt` 解析：

| Kernel | 作用 |
|---|---|
| `vector_add` | 一维 logical loop 与 pointwise memory flow |
| `tensor_showcase` | tensor transformation、显式 broadcast、record、helper、reduce/scan |
| `control_showcase` | domain product、parallel/ordered/for/while、if、break/continue 与 loop carry |
| `stream_showcase` | state stream 与结构化多 SSA state |
| `memory_showcase` | partition、indices、gather/scatter、logical buffer、atomic、fence、random |
| `gemm` | canonical contract 与 constexpr activation |
| `stable_softmax` | canonical max/subtract/exp/sum/divide tensor-flow |
| `flash_attention_fwd` | canonical Q/K regions、online state、mask 与两次 contract |
| `reduction_pass1` | canonical partitioned first-pass reduction |
| `reduction_pass2` | canonical second-pass reduction |
| `moe_expert_ffn` | canonical ragged membership、routing gather、contract 与 scatter-reduce |

这组程序覆盖当前 `OpCode` 枚举的 50/50 个节点。覆盖是从 lowering 后的实际 function/region operation 收集的，不是手造 IR 清单。

前端在本里程碑中同时收紧了以下语义：

- operation 与 SSA value 都具有模块内稳定 node ID，Physical Plan 通过 ID 引用 Kernel IR；
- tensor 扩张被正规化为显式 `broadcast`，binary、compare、select、mask 与 indexed store 共用同一 shape 规则；
- tuple 只表示有序多 SSA schema，不再保留无 producer 的 opaque tuple IR；record 继续承担命名字段；
- `_` 是解构丢弃目标，不进入环境或 structured-region carry state；
- index relation 同时验证静态边界、动态 index dtype、result shape 和 store-value broadcasting；
- random seed/index、ragged selector、gather/scatter 与 atomic producer 和 verifier 使用相同约束；
- Intent MLIR 保留稳定 node ID、函数结果类型信息、enum members、symbol/dynamic shape metadata 与 optional unit attributes。

## Physical Plan 与 Triton 路径

Realizer 不按 kernel 名称选择模板。首个 Triton realization 结构化匹配一个静态一维 f32 logical loop，并生成独立 Physical Plan：

- extent：`tile_size = 256`；
- ownership：program axis 0、static traversal；
- storage/layout：ABI views 为 global contiguous；
- primitive：loop 内 add/subtract/multiply/true-divide 显式绑定为 pointwise primitive；
- pipeline：单 stage，不启用 prefetch 或 async copy；
- boundary：整除时 exact，否则 masked；
- launch：`ceil_div(logical_extent, tile_size)` 的一维 grid。

Plan verifier 将 binding 限制在 entry kernel 内，要求恰好一个被 artifact 发射的 loop，完整覆盖 ABI storage/layout 和 loop primitive，并验证 domain extent、tail、grid、ownership 与 pipeline 均和 emitter 实际能力一致。unsupported realization 直接抛出 `NotImplementedError`，没有 Python fallback。

Triton emitter 按 Plan node ID 与 Kernel IR def-use 生成可独立阅读的 source。Artifact 封装 source、Intent MLIR、Physical Plan、launch configuration 和 callable entry；第一次真实 launch 后保存 Triton 暴露的 textual backend IR。

## 唯一可执行 repro

```bash
bash examples/repro/run_frontend_triton.sh
```

本次结果：

```text
frontend MLIR: 11/11 kernels PASS
frontend opcode coverage: PASS (50 opcodes)
backend IR levels: llir, ptx, source, ttgir, ttir
Triton numerical comparison: PASS (1000003 f32 elements, max error 0.0)
```

数值执行只对应 `vector_add`：三个静态一维 contiguous/noalias f32 views，逻辑 extent 为 1,000,003，因此真实经过 masked tail。比较对象是同一组 CUDA tensor 上的 `lhs + rhs`。

## 当前边界

以下内容尚未被当前结果证明或实现：

- canonical GEMM、softmax、FlashAttention、reduction 与 MoE 已证明可生成并解析 Intent MLIR，但尚未进入 Triton backend；
- Triton realizer 尚不支持 helper、多维或动态 view、stride/alignment/alias contract、reduce、contract、ragged、gather/scatter、atomic、state stream 与复杂 control flow；
- Physical Plan 尚无搜索、cost model、多 target policy 或可序列化的 MLIR dialect；
- Intent dialect 的 logical type 仍以 canonical spec 承载部分结构，MLIR 内还没有把所有 logical type 字段拆成可独立查询的参数；
- 50/50 opcode coverage 证明当前设计节点都存在 producer/verifier/emission 路径，不等价于所有类型、shape、constexpr 分支和边界组合的语义穷举；
- 当前唯一数值证据是一个 Triton pointwise kernel，不能外推为其余十个 kernel 或完整后端的数值正确性。

这些边界决定后续 backend 工作必须继续扩展 Realizer、Physical Plan legality 与 target emitter，而不能绕回按示例名称生成代码，或把 frontend MLIR 可解析误当成 backend 已完成。
