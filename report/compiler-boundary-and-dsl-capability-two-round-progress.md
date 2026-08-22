# 编译器职责边界与 DSL 缺失能力：两轮推进报告

## 1. 范围与结论

这两轮分别解决两个不同层次的问题：

1. 第一轮重新核定一个 `@intent.kernel` 的职责边界，删除 compiler-private multi-launch stage；
2. 第二轮补作者算法真正需要的低位数据语义，并按 upstream 算法补缺失 DSL source。

两轮之后，主链的边界是：

```text
一个 @intent.kernel
  → 一个 Kernel IR entry
  → 一份 single-launch Physical Program
  → 一个 provider kernel entry
  → 一次 target launch

多个 launches 的算法
  → 作者写多个 @intent.kernel
  → wrapper 显式分配/传递中间张量并顺序 launch
```

第一轮删除的是越过算子编译器边界的自动 kernel fission；第二轮增加的是会改变数值、必须由作者写下的 SSA 算法。两者方向一致：编译器不替作者发明算法阶段，但作者写下的 packed format、bit decode、索引关系和多-kernel 编排不能在 lowering 中丢失。

本报告只记录设计、实现和验证状态。稳定设计语义同步在 `doc/`；运行通过、性能差距和未闭合项只记录在本报告。

---

## 2. 第一轮：编译器职责边界自查

### 2.1 核心判断

对 Triton 与 TileLang 编译器结构的核查没有发现“把一个作者 kernel 自动拆成多个 GPU launches”的通用 compiler pass。它们会改变同一 kernel 内部的 layout、blocking、copy、pipeline、storage 和 collective realization，但不会因为 contraction 后接 scatter，就替作者创建第二个 runtime-visible kernel 和 host workspace。

因此，原来的 `intent_plan.stage` 不是普通物理 realization，而是在改变作者程序的 launch topology。它超出了 single-kernel operator compiler 的职责，应该删除，而不是继续扩充同步、workspace 和 fusion 字段。

### 2.2 删除的 stage executable path

本轮从唯一执行链中删除了以下内容：

- Plan dialect 的 `StageOp`、`StageAxisOp`；
- stage node/axis 的 schema、索引、verifier 和 analysis cache；
- `StageDecision`、`contractionStages`、`reconcileStages` 等 shared stage 推导；
- `FormStagesPass` 与 `Structured/StageFormation.cpp`；
- staged contraction 的 `lhs_form`、`rhs_form` 和 staged-only operand forms；
- 三个 provider 中 stage kernel header、stage wrapper、跨 launch workspace、workspace scatter、stage synchronization 与 `activeStages`；
- terminal source 中由 compiler 生成多个 kernel entry 并顺序 launch 的路径。

删除后，三个 materializer 只生成一个 provider kernel。单 launch 内仍可使用 replay、private workspace、persistent traversal、copy/pipeline 和 target-native collective；这些不会增加 runtime-visible callable 数量。

### 2.3 保留的自动决定

自查没有把所有自动决定一并删除。下面这些仍属于同一作者程序的物理 realization，因此保留：

| 决定 | 为什么仍属于编译器 |
|---|---|
| 从完整 logical domain 建立 physical blocking | 不改变 body 的 logical workset，只决定同一工作怎样映射到机器 |
| persistent traversal / worker folding | 不改变逻辑 iteration identity，只改变 program/grid 遍历方式 |
| pointwise lane promotion | body 仍观察同一个 scalar instance，只改变多个实例怎样装入物理 lanes |
| private buffer residency | buffer 的逻辑生命周期由作者固定，寄存器/local/global 的选择属于物理存放 |
| boundary neutralization | logical validity 与 reduction identity 已由算法确定，compiler 负责保证制造出的 physical tail 不影响结果 |
| provider form selection | 只选择该 target 上兑现同一个 Plan fact 的 native access/collective/primitive form |

这条边界不是“参考编译器做了就机械照抄”，而是用它们检验最终判据：改动是否改变作者写下的 program topology、logical workset、state、effect 或 ABI。改变这些的是作者算法；只改变同一结构的机器实现的是 compiler realization。

### 2.4 count partition 与 stream 的显式绑定

删除 compiler-private stage 后，split-K decode 不能再依赖隐式 stage。作者写下的 `partition(count=P)` 必须完整进入 Plan，并能与 part 内 `state_stream` 组合。

本轮因此补齐：

- count-partition region argument 的稳定 provenance；
- `StreamBindingOp.partition_node`；
- stream 引用的 partition 必须存在、来自同一 logical axis、对应精确 source region value；
- provider 从 Plan 读取 part begin/end，stream 只遍历该 part 的连续 key range；
- cuTile 对非零 part begin 使用 element-space gather form，不把 tile-space `ct.load` 错当成任意 segment load。

这不是重新引入 stage。每个 partial kernel 仍是独立作者 kernel；Plan 只实现该 kernel 内作者可观察的 part identity 和 streamed region。

### 2.5 文档收敛

当前设计文档已明确：

- 一个 Kernel IR entry 对应一个 target kernel entry 和一次 launch；
- compiler 不执行 source-kernel fission/fusion；
- 多-kernel pipeline、intermediate tensor 与调用顺序由 wrapper 表达；
- Physical Plan 不保存跨 launch stage graph；
- provider leaf 不生成 compiler-private host orchestration。

历史报告中关于 compiler-private stage 的旧叙述没有回写；历史报告保留当时事实，不继续作为当前规范。

---

## 3. 第二轮：DSL 语义与缺失能力

### 3.1 `I.bitcast` 的完整纵向链

本轮新增 `I.bitcast(value, dtype)`，并闭合：

```text
Python surface
  → frontend OperationKind.BITCAST
  → intent.bitcast
  → intent_plan.exec_bitcast
  → shared pointwise facts / verifier
  → provider form
  → Triton tl.cast(..., bitcast=True)
    / cuTile ct.bitcast(...)
    / TileLang T.reinterpret(...)
```

它的正式合同是：

- source/result shape 完全相同；
- element bit width 必须相同；
- bool/index 不允许参与；
- 不执行数值转换，只重解释相同 bits；
- closure、index provenance 和 logical validity不会因为 bitcast 被重新猜测。

`cast` 与 `bitcast` 因此是两个不同的 Kernel IR operation。

### 3.2 packed storage 的语言边界

本轮没有增加一个含糊的“通用 FP4/INT2 packed dtype”。采用的 Core 合同是：

```text
普通整数 ABI view
  + logical element → storage element 的索引关系
  + shift / mask / integer arithmetic
  + cast 或 equal-width bitcast
  + source-visible scale/zero-point/layout relation
```

以下内容属于作者算法：

- 一个 byte/word 装几个逻辑值；
- 每个逻辑值位于哪些 bit positions；
- interleave、nibble order、符号扩展、指数/尾数恢复；
- scale 的 group、编码与 swizzle ABI。

以下内容仍属于 compiler/provider：

- 一次物理解码几个逻辑值；
- 解码结果驻留在哪一级；
- 是否使用 `lop3`、`dp4a`、vector unpack 或 provider external intrinsic；
- tile、threads、pipeline 和 instruction selection。

这一区分也适用于未来 CPU/RVV backend：BitNet INT2 或 FP4 的逻辑 decode 不会因 target 改变，但向量解包和存储层级会改变。

### 3.3 已写入的低位算法

#### BitNet INT2

`bitnet_int2_matmul` 保留 upstream 的 4×4 interleave：

```text
storage byte = (k // 16) * 4 + (k % 4)
shift        = ((k // 4) % 4) * 2
value        = (packed >> shift) & 3
```

解码后的 i8 value 进入 i32 contraction。它不是按连续四个值简单打包的另一种 INT2 ABI。

#### Twiddled FP4 → BF16

`dequant_bf16_fp4_matmul` 按 upstream twiddled ABI：

- 每两个 bytes 形成一个 16-bit word；
- 四个 logical positions 使用不同 mask/shift 组合；
- 结果组成 BF16 bit pattern；
- `I.bitcast(u16 → bf16)` 后按 source bias 恢复数值；
- BF16 producer 再进入 f32-accumulating contraction。

这与普通 low/high nibble E2M1、NVFP4 的 swizzled E4M3 scale ABI不是同一个格式。

### 3.4 缺失 source entry 的 DSL 补齐

| Entry | 作者程序结构 | 本轮落点 | 当前验证状态 |
|---|---|---|---|
| `mamba3_siso_forward` | chunk preprocessing + carried SSM recurrence，shifted dt/trap、RoPE、多个 contractions | `streaming/mamba.py`，Triton provider/registry 已接线 | 已写入并接线；未单独数值运行 |
| `chunk_gated_delta` | intra-chunk prepare + inter-chunk recurrence 两个作者 kernels | `streaming/gated_delta.py`，cuTile provider/registry 已接线 | 已写入并接线；未单独数值运行 |
| `linear_attention_backward` | forward carried traversal 计算 dQ，reverse carried traversal 计算 dK/dV | `streaming/linear_attention.py`，TileLang provider/registry 已接线 | 已写入并接线；未单独数值运行 |
| `sparse_mla_backward` | delta preprocess + indexed sparse backward/many-to-one dKV atomic + dKV cast | `backward/sparse_mla.py`，TileLang provider/registry 已接线 | 已写入并接线；未单独数值运行 |
| `grouped_flash_decode` | count-partition partial + attention reduce | cuTile attention partial + `splitk_attention_reduce` | 数值通过 |
| `attention_sink_decode` | sink-aware count-partition partial + reduce | cuTile attention partial + `splitk_attention_reduce` | 已写入并接线；未单独数值运行 |
| `gemma_decode` | sliding-window/soft-cap count-partition partial + reduce | cuTile attention partial + `splitk_attention_reduce` | 已写入并接线；未单独数值运行 |
| `bitnet_int2_decode` | exact interleaved INT2 decode + contraction | `bitnet_int2_matmul`，TileLang provider/registry 已接线 | 数值通过 |
| `dequant_bf16_fp4` | exact twiddled FP4 decode + contraction | `dequant_bf16_fp4_matmul`，registry 已登记 | DSL 已写；source adapter 仍未闭合 |

### 3.5 两个算法/ABI取舍

#### Chunk gated-delta intermediate

初稿把 `C` 只放在输出 shape，compiler 无权从 `T` 猜 `C=ceil(T/64)`。最终没有增加 shape fallback，而是把 source 连续的 `(chunk, local)` intermediate view 写成一维 logical sequence ABI：

```text
(B, H, C, 64, K/V) 的连续 view
↔
(B, H, T, K/V) + source_positions = chunk * 64 + local
```

chunk relation仍由作者索引式表达；所有 output dimensions 均由 input ABI 唯一绑定。physical tail 的地址安全和 neutralization 继续由 compiler validity 机制负责。

同时撤回了初稿中把 f32 matrix operands 显式窄化到 bf16 的写法。Upstream 使用 f32→TF32 native matrix path；Intent source 保留 f32 operands，由 provider/backend 选择 TF32/native contraction，只有 source 指定的 intermediate ABI 落 bf16。

#### Sparse MLA backward

没有把 TileLang `atomic_addx4` 抬成共享 Core op。作者只写 typed `atomic_add` 的 many-to-one effect；是否四元素向量化属于 TileLang provider form。三个 runtime-visible kernels 也保持独立，没有重新引入 compiler-private stage。

---

## 4. 定向验证

两轮分别使用一条手动端到端 repro：DSL 经 compiler emit 到 provider source，provider 实际 JIT/launch，并和 upstream 做数值比较；没有建立 test 目录、pytest、fixture 或新验证脚手架。

| Provider / entry | Generated p50 | Upstream p50 | 结果 | 结论 |
|---|---:|---:|---|---|
| cuTile `grouped_flash_decode` | 0.222848 ms | 0.167184 ms | PASS | count-partition + partitioned state-stream + 两作者 kernels 已闭合 |
| TileLang `bitnet_int2_decode` | 0.042144 ms | 0.008192 ms | PASS | packed INT2 索引、位运算、i32 contraction 和真实 source ABI 数值一致 |

BitNet 的约 5.14× 性能差距没有被包装成完成。Generated 当前使用通用 bitwise decode + contraction；upstream 使用 `lop3/dp4a`。这说明 DSL/KIR 语义缺口已关闭，但 TileLang provider 仍缺少从 typed packed-coverage facts 选择 native decode form 的性能能力。修复位置应在 provider realization，不应改 DSL、按 kernel 名匹配或把 external intrinsic 抬到共享 Plan。

`grouped_flash_decode` 仍约慢 1.33×，说明 split-K 结构正确后还存在 provider form/tuning 差距；本轮没有为追数字修改算法或计时范围。

编译期间 `intent-compile` 完整构建成功；`git diff --check` 无错误。

---

## 5. 仍未关闭的事实

### 5.1 已写入但尚未逐项数值验证

以下 entry 已有 DSL 和 provider adapter，但本轮没有将它们写成“通过”：

- `mamba3_siso_forward`；
- `chunk_gated_delta`；
- `linear_attention_backward`；
- `sparse_mla_backward`；
- `attention_sink_decode`；
- `gemma_decode`。

Registry/adapter 接线只能证明入口存在，不能证明 provider JIT、launch 和 numerical comparison 已闭合。

### 5.2 FP4 upstream adapter

Twiddled FP4 DSL 已表达 registered source 的 packed ABI，但 vendored source 在 import 时依赖一个没有与该 source/runtime 相邻保存的 external decode-intrinsic module。当前保持明确 gap，没有改写 upstream、没有使用临时 PyTorch reference，也没有用不同 simple-nibble algorithm 冒充该 baseline。

### 5.3 尚未写成对应 source entry 的低位算子

`block_fp4_quant` 与 cuTile `nvfp4_quantize` 仍未闭合为对应 DSL source。原因不是统一的“FP4 不支持”：

- TileLang block FP4 使用 E8M0-like scale、32-element group 和 low/high packed pair；
- NVFP4 使用 E4M3 scale、16-element group、128×64 tile 与 512-byte swizzled scale ABI；
- 两者的 source-visible rounding、scale encoding 和 output shape relation不同。

本轮没有为减少空项而增加一个会丢失这些差异的 generic FP4 op。后续若补，应分别写出真实格式合同；共同复用的是 integer storage、bitcast、bitwise、reduce 与 shape/index semantics。

### 5.4 性能能力

本轮解决的是职责边界和算法表达，不等于 provider 生成质量已经完成。已暴露的明确性能项是：

- TileLang packed INT2 native decode / integer dot form；
- cuTile split-K decode 的 provider form 与 tuning；
- 其它新增大 kernel 尚未产生可归因数字。

这些问题不能通过恢复 compiler-private stage、修改 DSL 迁就单一 provider，或按 entry 名替换模板解决。

---

## 6. 两轮后的实际状态

已经完成的核心：

- single-kernel compiler boundary 在 schema、pass、verifier、三 provider materializer 和文档中一致；
- compiler-private multi-launch stage path 已删除，没有保留 fallback；
- source-visible `partition(count)` 可以与 part 内 ordered stream 机械组合；
- `I.bitcast` 成为正式、typed、跨 provider 的 Core operation；
- packed INT2/FP4 可以用 source-visible integer ABI、index relation 和 SSA bit arithmetic 表达；
- 缺失 entry 不再因为 compiler 自动 stage 被伪装成单 kernel，multi-kernel 算法由作者显式写出。

尚不能宣称的内容：

- 所有新增 entry 已在三个 provider 或两台设备数值通过；
- packed low-bit generated performance 已追平 provider-native upstream；
- FP4 quantization family 已统一闭合；
- 本轮已做全量回归或更新固定 baseline 表。

因此，这两轮得到的是更干净的编译器责任边界和更完整的作者语言，不是一份新的全量性能结论。
