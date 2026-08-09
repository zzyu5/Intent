# Intent Kernel 基本盘、反向与部署融合阶段报告

> 报告日期：2026-08-10
>
> 代码基线：`2e51c3f lower fused int8 GEMM epilogues`
>
> 报告性质：相对 `axis-scheduling-three-layer-audit.md` 的增量报告，不覆盖上一份报告
>
> 验收入口：`./examples/run/repro.sh <backend> <kernel>`

## 结论

这一阶段把上一轮的逐轴调度和三层表示真正压到了四类此前缺失的结构上：

1. 普通 GEMM 与 staged/ragged contraction 的 M/N/K 尾块；
2. BF16 contraction 的 DSL、IR、emission、ABI 和实际运行全链；
3. 多输出 backward，以及作者在 Python 外层编排的两 kernel LayerNorm backward；
4. GEMM、列偏置、ReLU、残差、per-channel scale、饱和和 i8 输出组成的完整部署 epilogue。

这些能力没有通过增加 kernel 类型或专用 emitter 得到。Realization 仍然只分析 logical axis、operation relation、consumer semantics 和 ABI；三个 target 只新增必要的 dtype、pointwise spelling、buffer/launch 接线。

最终验收集合为 15 个 kernel 名称、3 个 provider，共 45 个 backend/kernel 组合。45 条 repro 均完成以下真实链路并通过数值比较：

```text
Python DSL
  -> canonical Intent Kernel MLIR
  -> GPU physical realization + optional search space
  -> Triton / cuTile / TileLang source
  -> 下层编译器编译
  -> GPU 实际执行
  -> reference 数值比较
```

需要准确限定这份结论：当前已经证明作者可以分别编译两个 kernel，并在普通 Python 外层顺序编排它们；这不是一个 first-class multi-kernel compiler artifact。当前 `intent_plan.stage` 仍是 compiler 为 ragged contraction 构造的物理切片，不能与作者外层 launch graph 混为一谈。

## 一、本阶段提交边界

本阶段保持一个能力节点对应一个提交：

| commit | 能力节点 |
|---|---|
| `84b954e` | contraction tail 通过逐轴 validity 与 padding 兑现 |
| `e1e744a` | BF16 contraction 三后端全链贯通 |
| `ece0fff` | 通用多输出 wrapper 与 SwiGLU backward |
| `bd55d48` | 两阶段 LayerNorm backward 与 ordered reduction |
| `2e51c3f` | 融合 i8 GEMM epilogue 与 toward-zero 语义 |

代码提交后 tracked 工作树为空。构建仍写到 `/tmp/intentdsl-build`；运行生成的 `__pycache__`/`.pyc` 被 Git 忽略，没有纳入提交。

## 二、任意 contraction 尾块如何兑现

### 2.1 尾块不是作者的参数

作者只声明逻辑收缩：

```python
I.contract(lhs, rhs, reduce=((1, 0),), acc_dtype=I.f32)
```

tile 是 realizer 与下层 tuner 共同决定的，因此 M/N/K 不能整除 tile 时产生的无效 lane 不能要求作者手写 mask。当前责任分解是：

1. `KernelFacts` 从 view relation、domain 和 region argument 建立 logical-axis provenance；
2. GPU machine-plan 根据 consumer 的 reduction/contract 语义要求相应轴的有效区间；
3. `PaddingState` 生成 `intent_plan.padding`，记录 value、tensor axis、domain node 和填充值；
4. emitter 只读取 `PaddingOp`，将同一 validity 决策打印成目标语言的 predicate、padding mode 或边界 copy。

关键实现位于：

- `lib/Target/Common/Realization/KernelFacts.cpp:107`：region argument 到 logical axis 的统一绑定；
- `lib/Target/GPU/Realization/Plan/Build.cpp:70`：validity binding；
- `lib/Target/GPU/Realization/Plan/Build.cpp:231`：reduction identity 对 padding 的要求；
- `lib/Target/GPU/Realization/Plan/Build.cpp:288`：contract multiply/add 对 zero padding 的要求；
- `lib/Target/Common/Realization/Proofs.cpp:20`：padding 沿 pointwise consumer use-chain 的证明。

填充值来自最终 consumer：

| consumer 语义 | 无效 lane 的值 |
|---|---|
| add reduction | `0` |
| maximum reduction | `-inf` |
| multiply/add contraction | `0` |
| 只流向带边界的 store | 允许经过 pointwise 链，最终由 store validity 丢弃 |

`broadcast`、`cast` 和乘法的 padding 传播是共享证明规则，不属于 bias、LayerNorm 或 GEMM 特判。

### 2.2 动态验收形状

普通 GEMM 的额外 repro 使用：

```text
M = 4093
K = 4080
N = 14320
```

这三个维度对当前 32/64/128 tile 候选都会产生尾块。三个 provider 的 generated/reference 最大误差均为 `0`。

ragged grouped GEMM 的额外 repro 使用：

```text
members = 8191
K = 4080
N = 4080
```

它同时制造 member、K、N 尾块，三个 provider 的最大误差均约为 `1.973e-3`。这条路径还验证了 compiler-generated physical stages 下的 staged load、workspace 和 terminal store。

### 2.3 尾块性能

单位：ms。

| case | Triton p50/p95 | cuTile p50/p95 | TileLang p50/p95 |
|---|---:|---:|---:|
| GEMM M/N/K tail | 2.1727 / 2.1768 | 2.1531 / 2.1552 | 2.1326 / 2.1349 |
| grouped GEMM member/K/N tail | 1.3374 / 1.3461 | 1.4564 / 1.4595 | 1.2851 / 1.2911 |

## 三、BF16 基本盘

### 3.1 Canonical 算法语义

`examples/kernels/contraction/gemm.py:40` 中的 `bf16_gemm` 明确声明：

```text
BF16 lhs × BF16 rhs
  -> FP32 accumulation
  -> explicit BF16 narrowing
  -> BF16 output view
```

因此 accumulation dtype 是作者的算法语义；具体 BF16 cast 指令、寄存器布局和矩阵指令仍由目标语言及其编译器决定。

### 3.2 三后端贯通点

| 层 | 已贯通内容 |
|---|---|
| Python DSL | `I.bf16` input/output 与 `I.cast(..., I.bf16)` |
| Kernel MLIR | `tensor<...xbf16>` 与 FP32 contract result |
| realization | contract operand/result space 沿既有 per-op plan 生成 |
| Triton | `tl.bfloat16`，wrapper 使用 `torch.bfloat16` |
| cuTile | `ct.bfloat16`，wrapper 使用 `torch.bfloat16` |
| TileLang | `T.bfloat16`，显式 buffer/fragment dtype 与 `torch.bfloat16` |
| runtime | 输入、输出和 upstream 结果均检查真实 dtype |

没有增加 `bf16_gemm` emitter；所有选择都由 MLIR element type 驱动。

### 3.3 运行结果

| provider | generated p50/p95 (ms) | max error | upstream |
|---|---:|---:|---|
| Triton | 2.0640 / 2.0682 | 0.03125 | unavailable：仓内 Triton GEMM wrapper 固定 FP16 输出 |
| cuTile | 2.1513 / 2.1531 | 0.03125 | 2.1592 / 2.1593 ms |
| TileLang | 2.0507 / 2.0548 | 0.03125 | 2.2526 / 2.2535 ms |

当前 BF16 narrowing 使用各目标的 native cast。它规定了输入、累加和输出 dtype，但没有宣称三个下层在所有 BF16 舍入边界上 bit-exact。

## 四、Backward 如何被压出来

### 4.1 通用多输出 ABI

通用 wrapper 当前按同一套 ABI 规则处理一个或多个 output view：

1. 按 ABI 顺序收集任意数量的 `out` view；
2. 为每个 output 按 canonical shape/dtype 分配 `torch.empty`；
3. 单输出保持返回一个 tensor；
4. 多输出返回按 ABI 顺序排列的 tuple；
5. Triton `launch()` 仍返回 compiled kernel，保持 backend IR 收集协议不变。

这个规则只看 ABI access 与输出数量，不知道 backward 或 kernel 名称。

### 4.2 SwiGLU backward：最小多输出反向

`examples/kernels/backward/swiglu.py` 直接对照 Liger 的 Triton backward 结构：输入 `dc/a/b`，重算 sigmoid 与 SiLU，输出 `da/db`。

它验证：

- BF16 输入与输出；
- FP32 中间计算；
- 一个 kernel 写两个独立结果；
- 三个 wrapper 都能返回 tuple；
- 新增 pointwise backward 不需要新的调度模式。

| provider | generated p50/p95 (ms) | da/db max error | upstream |
|---|---:|---:|---|
| Triton | 0.0973 / 0.1452 | 0.001953 / 0.001953 | Liger 0.1196 / 0.1211 ms |
| cuTile | 0.0939 / 0.0959 | 0.001953 / 0 | unavailable |
| TileLang | 0.0979 / 0.1084 | 0.001953 / 0 | unavailable |

### 4.3 LayerNorm backward：作者两 kernel 编排

`examples/kernels/backward/layer_norm.py` 包含两个独立的 `@intent.kernel`。

第一阶段 `layer_norm_backward_rows`：

```text
每个 program 拥有一行
  -> 从 x 重算 mean / variance / rstd / normalized
  -> 计算 dx
  -> 写 dx + dw_partial + db_partial
```

第二阶段 `layer_norm_backward_reduce`：

```text
feature tile 进入 parallel program space
rows 进入 ordered state_stream
  -> 每个 stream tile 归约 partial rows
  -> carried state 累积 dw/db
  -> 写最终 dw/db
```

普通 Python 外层按作者顺序执行：

```python
dx, dw_partial, db_partial = rows_artifact.run(...)
dw, db = reduce_artifact.run(dw_partial, db_partial)
```

这压出了四个此前没有动态证据的接口：

1. 一个 kernel 多输出；
2. workspace 跨 launch 传递；
3. 第二 launch 完成跨第一阶段 program ownership 的归约；
4. 一个作者阶段内部使用 ordered streaming，而不是为整条 backward 增加模式分支。

### 4.4 cuTile state shape 的架构修正

cuTile state carrier 的 rank 现在只由 canonical result rank 和显式广播关系决定。attention 的 axis-1 reduction 需要列向量，而 LayerNorm backward 的 axis-0 reduction需要行向量，因此不能让 emitter 根据 `state_stream` 本身猜方向。

当前通用规则是：

- reduction 默认 `keepDims = false`，保持 canonical reduction result rank；
- `I.reduce(..., axis=...)` 决定数学归约轴；
- 只有 DSL 中显式的 `[:, None]` / `[None, :]` 才扩维；
- `state_stream` 不隐式增加 singleton 维度，也不跳过显式 expand-dims。

修正后同一机制通过了 cuTile LayerNorm backward、attention 和 online softmax，分别覆盖 axis-0 tensor reduction、axis-1 tensor reduction和 scalar stream state。

### 4.5 LayerNorm backward 结果

| provider | pipeline p50/p95 (ms) | dx/dw/db max error | upstream |
|---|---:|---|---|
| Triton | 0.3601 / 0.3625 | 0.001953 / 2.384e-6 / 9.537e-7 | official tutorial 0.2396 / 0.2601 ms |
| cuTile | 0.2434 / 0.2442 | 0.001953 / 4.768e-6 / 1.907e-6 | unavailable |
| TileLang | 0.2728 / 0.2741 | 0.001953 / 2.384e-6 / 9.537e-7 | unavailable |

Triton generated pipeline 当前约为 official tutorial 的 `1.50x`。数值链已经成立，但这里不能写成性能追平。

### 4.6 与 `intent_plan.stage` 的边界

需要区分两种“stage”：

| 概念 | 当前状态 |
|---|---|
| 作者将 backward 拆成两个 kernel，并在 Python 外层顺序 launch | 已证明 |
| compiler 为一个 ragged contraction kernel 自动建立 `intent_plan.stage` operation slices/workspace | 已有能力 |
| 一个 first-class compiler artifact 保存多个作者 kernel、workspace 生命周期和 launch graph | 尚未实现 |
| 同一个 physical StageOp slice 内动态证明 ordered state stream + staged contraction | 当前 corpus 仍未证明 |

因此本报告不会把 Python orchestration、compiler-generated StageOp 和同一 slice 的 stage/stream 组合写成同一件事。

## 五、部署 epilogue 与 i8 输出

### 5.1 DSL 写的是算法，不是后端技巧

`quantized_gemm` 的 canonical 算法为：

```text
acc_f32 = contract(a_f16, b_f16)
biased  = acc_f32 + bias_f32[N]
active  = max(biased, 0)
fused   = active + cast(residual_f16[M,N], f32)
scaled  = fused / output_scale_f32[N]
bounded = min(max(scaled, -128), 127)
output  = cast_toward_zero(bounded, i8)
```

这里同时压到：

- rank-1 bias 沿 M 轴广播；
- rank-1 per-channel quantization scale 沿 M 轴广播；
- FP16、FP32 和 i8 在同一个 kernel 中流动；
- contraction 后的多段融合 pointwise；
- i8 external output view 的分配、验证和 store。

### 5.2 Canonical rounding 语义

单纯写 `cast(i8)` 会让 float-to-int rounding 含糊。当前 frontend 在 float/bfloat → signed/unsigned integer 时自动附加：

```text
intent.rounding = "toward_zero"
```

GPU realization 会验证：

- float-to-integer cast 必须带 `toward_zero`；
- 其他 cast 不允许携带这个 rounding 属性；
- 饱和不是 cast 的隐藏行为，而是 DSL 中显式的 `minimum(maximum())`。

因此 saturation 已经完全落在 canonical pointwise 链中；三个目标的 float-to-i8 映射依赖当前 API 的 toward-zero conversion，并由三条数值 repro 共同验证。若上游 API 改变 conversion 语义，target capability/mapping 也必须同步拒绝或调整，不能静默接受。

### 5.3 三个 target 只补叶子能力

| canonical concept | Triton | cuTile | TileLang |
|---|---|---|---|
| binary minimum | `tl.minimum` | `ct.minimum` | `T.min` |
| signed i8 value | `tl.int8` | `ct.int8` | `T.int8` |
| torch ABI/output | `torch.int8` | `torch.int8` | `torch.int8` |
| narrowing | `tl.cast` | `ct.astype` | `T.cast` |

没有增加 `quantized_gemm` realizer、emitter 或 target handler 文件。新增 kernel 只组合已有 contract/load/broadcast/binary/cast/store，target 叶子只增加概念到语法的映射。

### 5.4 运行结果与 baseline 边界

| provider | generated p50/p95 (ms) | max i8 error | upstream |
|---|---:|---:|---|
| Triton | 2.0898 / 2.0939 | 1 | unavailable |
| cuTile | 2.1756 / 2.1788 | 1 | unavailable |
| TileLang | 2.1367 / 2.1440 | 1 | unavailable |

不同 MMA 累加顺序会使少量值跨过 toward-zero 的整数边界；本轮观测到的三条最大 i8 误差均为 1，输出 dtype 和饱和区间一致。

`source/` 中没有语义完整等价的 `GEMM + bias + ReLU + residual + output quantization` 单 kernel baseline：

- Triton source 中最接近的是 dense GEMM 加可选 activation；
- cuTile source 中存在 input block-scaled GEMM，但不是 output quantization；
- TileLang source 中存在 FP8/W4A8 GEMM，但不是这条 fused epilogue。

因此本项只报告 generated/reference，不拼接几个不同算法冒充 upstream fused baseline。

## 六、15 × 3 完整 repro

### 6.1 数值状态

| kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | PASS | PASS | PASS |
| layer_norm | PASS | PASS | PASS |
| layer_norm_backward | PASS | PASS | PASS |
| rms_norm | PASS | PASS | PASS |
| logsumexp | PASS | PASS | PASS |
| gemm | PASS | PASS | PASS |
| bf16_gemm | PASS | PASS | PASS |
| quantized_gemm | PASS | PASS | PASS |
| dual_gemm | PASS | PASS | PASS |
| attention | PASS | PASS | PASS |
| varlen_attention | PASS | PASS | PASS |
| online_softmax | PASS | PASS | PASS |
| moe | PASS | PASS | PASS |
| grouped_gemm | PASS | PASS | PASS |
| swiglu_backward | PASS | PASS | PASS |

`varlen_attention` 的每个 provider repro 内又分别执行 `causal=False` 和 `causal=True`；`gemm` 与 `grouped_gemm` 的每个 provider repro 内又执行额外 tail case。因此 `45/45` 指 backend/kernel 命令数，不是内部 GPU kernel launch 或数值 case 的数量。

### 6.2 Generated p50

单位：ms。数据来自本报告生成前连续执行的完整 repro 矩阵。

| kernel/config | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | 0.3560 | 0.3580 | 0.3492 |
| layer_norm | 0.1768 | 0.1799 | 0.1737 |
| layer_norm_backward | 0.3601 | 0.2434 | 0.2728 |
| rms_norm | 0.1768 | 0.1799 | 0.1741 |
| logsumexp | 0.1635 | 0.1696 | 0.1614 |
| gemm | 2.1174 | 2.3067 | 2.1643 |
| bf16_gemm | 2.0640 | 2.1513 | 2.0507 |
| quantized_gemm | 2.0898 | 2.1756 | 2.1367 |
| dual_gemm | 0.6965 | 0.7185 | 0.7380 |
| attention | 4.9981 | 5.0456 | 4.8924 |
| varlen_attention，causal=False | 0.4301 | 0.3832 | 0.4872 |
| varlen_attention，causal=True | 0.2894 | 0.2976 | 0.3505 |
| online_softmax | 0.3828 | 0.3519 | 0.3806 |
| moe | 8.9172 | 10.2437 | 11.4654 |
| grouped_gemm | 1.3106 | 1.4522 | 1.2690 |
| swiglu_backward | 0.0973 | 0.0939 | 0.0979 |

与上一份报告中同名 kernel 的紧邻结果相比，没有出现与本轮结构修改对应的系统性性能退化。单次 p50 仍可能有约 1% 左右波动，不能把这种差异解释成优化或回退。

### 6.3 不能写成性能完成的组合

以下 generated 路径数值正确，但当前仍慢于可用 upstream：

- Triton LayerNorm backward：约 `1.50x`；
- cuTile MoE：约 `1.07x`；
- TileLang MoE：约 `1.22x`；
- TileLang causal varlen attention：约 `1.20x`；
- Triton/cuTile GEMM 与 Triton attention 略慢约 0.5%–2%，处于接近持平范围。

cuTile attention 的 upstream 本轮测得约 68 ms，而 generated 约 5 ms。该比值明显包含 upstream wrapper/实现路径差异，不能据此宣称算法级 13 倍优势。

## 七、架构与冗余复核

### 7.1 Realization

`lib/Target/Common/Realization` 和 `lib/Target/GPU/Realization` 中没有：

- `softmax/gemm/attention/moe/backward/quantized_gemm` 名称分派；
- whole-kernel matcher；
- mapping-mode 枚举入口；
- 为 BF16、backward 或 output quantization 新建的调度策略文件。

当前唯一的选择结构仍是逐 logical domain 累积 `parallel/ordered/reduction/ragged_member/lane` 角色，再分别确定 program order、worker/fold、tile role 和 reuse。

### 7.2 Emission

三个 target 的新增内容只有：

- 多 output ABI allocation/tuple return；
- `minimum` spelling；
- BF16/i8 dtype spelling；
- canonical rank、broadcast 和 cast 的机械打印。

cuTile state-stream 按 canonical rank 发射，扩维只响应 Kernel IR 中显式的 expand-dims；不存在另一份 emitter-local state shape。

所有 tracked C++ `.cpp` 都仍在对应 CMake source list 中；没有发现空的旧 projection/target dialect 目录或未接线源文件。

`tools/intent-opt` 没有仓内 repro caller，但它仍是独立的 MLIR optimizer CLI，不属于本阶段被替代的 emission 路径，因此没有仅凭“仓内没人调用”删除。

## 八、当前明确边界

### 已兑现

1. 普通与 staged contraction 的真实非整除尾块；
2. consumer-derived validity/padding；
3. BF16 contraction 三后端生成与运行；
4. generic multi-output wrapper；
5. 一个真实 pointwise backward；
6. 作者 Python 外层两 kernel LayerNorm backward；
7. 第二阶段 ordered streaming reduction；
8. mixed-rank、mixed-dtype fused deployment epilogue；
9. canonical toward-zero float-to-int semantics；
10. 15 个 kernel × 3 个 provider 的真实执行。

### 尚不能宣称

1. first-class multi-kernel DSL/CompiledArtifact、workspace 生命周期与 launch graph；
2. physical `StageOp` slice 内的 staged contraction + ordered state stream 动态组合；
3. 三后端 BF16 所有舍入边界 bit-exact；
4. 完整 fused quantized GEMM 的 upstream 性能对照；
5. 所有 kernel/provider 都达到或超过上游性能；
6. autograd 接入、artifact cache 和跨 shape specialization cache 的完整库化接口。

这些边界没有用兜底路径伪装成已支持；当前未实现部分会在相应入口显式失败或根本没有 first-class API。

## 九、手动复核

唯一验收入口：

```bash
./examples/run/repro.sh <backend> <kernel>
```

参数集合：

```text
backend = triton | cutile | tilelang

kernel  = softmax | layer_norm | layer_norm_backward |
          rms_norm | logsumexp |
          gemm | bf16_gemm | quantized_gemm | dual_gemm |
          attention | varlen_attention | online_softmax |
          moe | grouped_gemm | swiglu_backward
```

该命令会重新构建 `intent-compile`，从 DSL 生成 canonical MLIR 与 target source，导入目标语言编译器，在 GPU 上实际执行，并进行 reference 数值比较；存在可调用上游时还会执行 upstream comparison，不存在时明确打印 `upstream baseline: unavailable`。
