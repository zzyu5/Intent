# 全量收尾与真实状态

## 结论

这一轮完成了 source inventory 与 registry 的逐项闭合，也在 5090 和 H100 上并行跑过六条完整 provider 矩阵；随后针对全量暴露的问题做了定向修复和复测，并把最终可信结果合回六张固定表。

当前 registry 覆盖 128 个真实算法 entry：Triton 54、cuTile 37、TileLang 37；每个 entry 都有真实、相邻且存在的 `source_runtime`。六张表共有 256 个 provider-device 行，其中 199 行数值与 source 对照通过，57 行在明确阶段失败；最终表里没有 `numerical_failed`，也没有把超时、adapter 失败或 target 能力边界伪装成通过。

但“全量通过”和“性能全部站住”都还没有达到。199 个通过行中，131 行在 source 的 1.05 倍以内，68 行超过 1.05 倍，34 行超过 2 倍，7 行超过 5 倍。最大差距集中在 cuTile 的 MLA/split-K/窗口 attention、TileLang 的 attention/INT2，以及 Triton 的 flash attention/paged MLA。它们已经能按责任层说明，但不能写成已经关闭。

因此这轮拿到的是一份完整、无静默数值错误、失败边界可定位的真实基线；不是一份为了收官而全绿的表。

## 证据口径

两台机器先并行执行完整矩阵，覆盖此前四轮的 DSL、KIR、Physical Program、shared/provider passes、stage 删除、低位数据语义和新 adapter。全量之后的代码修复只重跑受影响 entry，并把最新定向结果按 `(provider, device, kernel, case)` 覆盖回固定表。没有在最终修复后伪称又执行了一次第二轮全量。

实际验证链始终是：

```text
DSL -> KIR -> Physical Program -> shared passes -> provider passes
    -> terminal source -> provider compile/JIT -> GPU run -> numerical compare
```

定向复现仍使用同一个手动入口，例如：

```bash
./examples/run/baseline-v2.sh cutile /tmp/recurrent.csv recurrent_gated_delta
```

H100 使用同一入口和同一 kernel/case，只在远端对应环境执行。固定表只是数字和最终状态，不带检查逻辑。

## Registry 与 source inventory

### 最终分母

| provider | runtime 文件 | 真实算法 | support/wrapper | registry | 每台 CSV 行数 |
|---|---:|---:|---:|---:|---:|
| Triton | 55 | 54 | 1 | 54 | 54 |
| cuTile | 38 | 37 | 1 | 37 | 37 |
| TileLang | 37 | 37 | 0 | 37 | 37 |
| 合计 | 130 | 128 | 2 | 128 | 128 |

Triton 原先所谓“17 个未登记 runtime”里，16 个是真算法，已经全部接入；第 17 个是 Meta MoE projection 的 support loader，不是独立 callable，不能为了补分母再造一个 entry。新增的 16 个 Triton entry 覆盖：conv1d、triangular solve、embedding、roll、transpose、batch norm training、group norm backward、logsumexp、softmax backward、AdamW、addcmul、FP8 MQA logits、cumsum、histogram、max-pool-with-indices 和带 bias 的 legacy flash attention。

cuTile 多出的 runtime 是 host 侧 MoE multi-launch wrapper，不是独立 kernel；其 37 个真实算法已经全部登记。TileLang 37 个 runtime 与 registry 一一对应。

现在三家 registry 的每个 entry 都绑定了存在的 `source_runtime`。这意味着“未登记 source”不再缩小分母，但不意味着每一行都能在当前 target/device 完成首次 JIT。

## 两个 FP4 问题

### Twiddled FP4 upstream 依赖

`dequant_bf16_fp4` 上游依赖的 `quantize` 包已经按原始结构放回 source 相邻位置：

```text
source/tilelang/tilelang/gemm/dequant_bf16_fp4/quantize/
```

其中包括 `mxfp.py` 提供的 twiddled decode intrinsic。adapter 现在可以 import 完整 upstream，并真正进入 TileLang JIT；不再是“缺 external module”。两台机器上 `w4a8_gemm` 与 `dequant_bf16_fp4` 都在 300 秒 worker 上限内未完成，最终状态是 `worker_timeout`。这说明依赖闭合了，当前剩余的是 provider 首次编译/候选成本，不能把 import 成功写成 kernel 通过。

### 两个格式不同的 FP4 quantize

没有制造一个丢失格式差异的 generic FP4 op：

- TileLang `block_fp4_quant` 是 group 32、power-of-two absmax、E2M1 rounding、偶数 low nibble/奇数 high nibble；当前缺 typed E2M1 conversion 与 packed-nibble output ABI。
- cuTile `nvfp4_quantize` 是 group 16、runtime global scale、E2M1 byte pairs，并要求固定 128x4 到 512-byte 的 scale swizzle；当前缺 typed E2M1 packing 与 first-class swizzled scale-storage ABI。

两者都保留为 `intent_implementation_gap`。这不是 target 不支持，而是 Intent 还没有完整表达真实格式合同。

## 全量暴露后关闭的问题

### ordered axis 被错误分配 program ownership

`recurrent_gated_delta` 的时间轴既是有序 state stream，又被旧判据因为 `vectorDomain` 分进 program grid，导致每个 program 重复完整递推。修复落在 shared GPU decision：有序轴不因结果是向量就自动获得 program ownership。

修正所有权之后，cuTile 仍比 source 慢 170--284 倍。进一步并排读 source 和 generated code，关闭了四个 provider realization 问题：

- 单一 program axis 的 tuner 实际没有覆盖 source 使用的 16-wide tile；现在合法候选包含 16/32/64/128 与 occupancy 1--4。
- Plan 已选连续 store 且 bounds=false，terminal 却固定生成 bounds-checked scatter；现在完整连续片段使用 `ct.store`。
- `expand_dims` gather 的 validity 恒真时仍生成 `ct.where(True, ...)`；现在直接投影为展开值。
- load -> cast/pointwise -> reshape 进入 contraction 的等元素链，provider 现在记录最终 fragment shape，terminal 直接按该形状 load，不再先物化低质量中间 tile。

最终结果：

| device | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.789584 | 0.761456 | 1.036940 |
| H100 | 1.019184 | 0.993856 | 1.025485 |

这个修复没有按 recurrent kernel 名称分支；它分别修正 ordered ownership、合法 tuner 参数和 cuTile provider form。

### Contract 物理轴绑定缺失

TileLang `gqa_decode` 的逻辑 contraction row 只有 4，当前 native MMA form 不能把小于 16 的 row pad 成合法 fragment。以前 direct contraction 没保留 lhs/rhs/reduction 的 selected physical axis，导致 provider capability guard 看不见真实 row extent，错误一直拖到下层 assertion。

shared contraction refinement 现在给 direct form 也绑定结果轴。两台机器都在 source emission 前稳定得到 `provider_program_failed`，诊断为当前 TileLang form 不支持 subwarp row contraction；普通 `block_sparse_gemm` 仍通过：5090 ratio 0.894413，H100 ratio 1.028148。这里收回的是假支持，不是把失败做成通过。

### Boundary、region provenance 与 target projection

前四轮的多-domain、ragged、partition stream 和 region-value 改动在全量里暴露了几处事实消费问题。本轮把 region binding 收敛到 canonical value ID，修正 ragged program/member 轴的运行时 begin/end 投影，保留非零 partition begin，并让三家 leaf 读取已选 range/contract axis，而不是再按逻辑 shape 或角色重建。

Boundary neutralization 只删除 producer chain 已经结构性保证的 padding；常量真 validity 不再生成冗余选择。代表性 attention 数值复测在三家均通过，没有出现新的尾块数值错误。

### 算法层 unary

新增 `I.tanh` 和 `I.abs` 是作者可观察的数值算术，不是物理优化许可。它们进入 canonical unary 语义，再分别机械投影为 Triton、cuTile 和 TileLang 的原生调用。Gemma/GeGLU/GELU 的定向结果证明链路闭合；H100 `gemma_prefill` 的 8.713 倍差距也证明“有原生 tanh”并不等于 attention provider form 已经高质量，二者没有被混为一层。

## 六张最终表

| 表 | 总行 | pass | 非 pass | ratio <= 1.05 | ratio > 2 | ratio > 5 |
|---|---:|---:|---:|---:|---:|---:|
| Triton / 5090 | 54 | 50 | 4 | 37 | 2 | 1 |
| Triton / H100 | 54 | 52 | 2 | 41 | 5 | 1 |
| cuTile / 5090 | 37 | 34 | 3 | 23 | 7 | 1 |
| cuTile / H100 | 37 | 33 | 4 | 13 | 10 | 4 |
| TileLang / 5090 | 37 | 15 | 22 | 9 | 5 | 0 |
| TileLang / H100 | 37 | 15 | 22 | 8 | 5 | 0 |
| 合计 | 256 | 199 | 57 | 131 | 34 | 7 |

固定表为：

- `report/baseline-new/triton-5090.csv`
- `report/baseline-new/triton-h100.csv`
- `report/baseline-new/cutile-5090.csv`
- `report/baseline-new/cutile-h100.csv`
- `report/baseline-new/tilelang-5090.csv`
- `report/baseline-new/tilelang-h100.csv`

最终非 pass 状态直方图：

| 状态 | 数量 | 性质 |
|---|---:|---|
| `provider_program_failed` | 18 | provider form/capability 在 emission 前明确拒绝 |
| `provider_jit_or_initial_launch_failed` | 13 | target compiler/JIT、资源或首次 launch |
| `intent_implementation_gap` | 8 | Intent 当前没有表达真实算法/ABI 合同 |
| `worker_timeout` | 8 | 首次编译或候选执行超过 300 秒 |
| `adapter_preparation_failed` | 3 | source/adapter 准备失败 |
| `physical_program_failed` | 2 | shared Physical Program 不能合法表达 |
| `compiler_invocation_failed` | 2 | target compiler invocation 失败 |
| `source_compatibility_gap` | 2 | vendored source 与当前 target runtime 不兼容 |
| `source_provider_jit_or_initial_launch_failed` | 1 | source 侧 JIT/首次 launch 失败 |

没有统一的 `compile_failed`，也没有 `numerical_failed`。

## 性能状态与公平性

199 个通过行里，102 行 generated 不慢于 source；131 行在 5% 内。以下差距仍是编译器质量问题，不能归给随机抖动：

- cuTile：`splitk_mla_decode` 为 9.82x / 63.07x，`absorbed_mla_decode` 为 3.51x / 40.74x，`gemma_prefill` 在 H100 为 8.71x。共同缺口是 head/KV tile、可见范围收紧、TMA/MMA operand form、split partial 与 provider occupancy 没形成与 source 同质量的联合 realization。
- TileLang：`varlen_gqa_prefill`、`block_causal_attention`、`gqa_attention_backward` 和 `bitnet_int2_decode` 在两机约 2.7--4.7x。算法/ABI 基本一致；当前 provider form 仍没有达到 source 的原生 attention pipeline 与 lop3/dp4a INT2 路径质量。
- Triton：`flash_attention_forward` 两机约 2.31x，`paged_mla_decode` 两机约 15--16x，H100 `paged_gqa_decode` 2.58x，max-pool-with-indices 2.60x。FA/paged MLA 缺高质量 block/descriptor/warp/provider form，max-pool 缺二维 output tile 与 window arg-reduce 的紧凑投影。

有几类 ratio 不能当作单 kernel 结论：

- Triton paged GQA/MLA source 是 split-KV partial + reduce，多次 launch；generated 是作者当前写下的单 kernel nested ragged stream。表里是同一最终结果的端到端比较，不是“一个 generated kernel 比一个 source kernel”比较。
- cuTile mHC source 是 split-K partial + finalize 两阶段，Intent entry 是一个 state-stream callable；其 2.8--3.3x 只能作为端到端结构差异，不能归成 leaf 拼写慢。
- TileLang GQA backward 两边梯度公式一致，但 source 与 generated 的多 kernel ownership/atomic/split 分解不同。
- cuTile H100 dense GEMM 的 source 是旧 non-persistent official form，generated 使用 static-persistent TileGym form；0.093x 不能单独证明 shared compiler 优势。

这些行仍保留在表里，因为 source inventory 与最终输出 scope 是真实的；报告明确限制它们的结论范围。

## 仍然明确存在的边界

### Shared Physical Program

`chunk_gated_delta` 两机 cuTile 都停在 `physical_program_failed`。算法使用 `C = ceildiv(T, 64)` 构造运行时 domain。KIR facts 保存了真实 SSA start/stop/step，但 Physical Program 的 range 目前只存字符串 extent；`runtime_domain_<node>` 只是不可执行 placeholder，三家 wrapper 也只会把 canonical ABI shape symbol 翻成运行时表达式。

正确闭合需要 typed runtime-extent expression 进入 Plan，并由 wrapper 机械翻译；把 placeholder 当符号或只取 stop 都会丢 start/step 语义。因此没有用字符串公式或 ABI 猜测绕过。这是当前最清楚的 shared compiler 能力缺口。

### Intent 语言/KIR

八个 `intent_implementation_gap` 主要包括两种 FP4 格式、DeepSeek radix top-k 所需的 typed float-bit reinterpret/shared histogram/barrier，以及 cooperative persistent MLA 与现有作者程序结构不一致。它们不是一个统一的“低精度不支持”。

### TileLang provider 子集

TileLang 每台只有 15/37 行通过。GQA subwarp row、paged/varlen single-row contraction、native sparse forms、部分 loop-carried/state forms 和 FP8 runtime-lane form 都在 provider program 阶段明确拒绝；不再让下层长时间编译一条已知不合法的慢路径。其余 JIT/timeout 项保留为编译成本或下层失败，未错误升级成语言缺口。

### Source 与环境

legacy Triton bias source 在当前 Triton 版本下是 `source_compatibility_gap`；flash-attention backward 和少数 adapter/source 入口停在 source preparation/JIT。它们不证明 generated compiler 缺能力，也不计入通过。

## 职责边界复核

- 旧 compiler-private multi-launch stage 已从唯一执行链删除；Plan verifier 和三家 provider program 都要求单 launch。没有 `StageOp`、stage wrapper、跨 launch workspace 或 stage synchronization 以别名回来。
- 当前 diff 未发现按 kernel 名称或整算子形态分支。shared 修复读取 typed axis、def-use、index relation、range purpose 和 producer chain。
- target-specific INT2、gather/load/store、native contract operand 等 form 在 provider pass 选择并写入 provider attrs；terminal 虽然需要生成较厚的 TileLang/cutile 语法，但不再自行决定 shared ownership、range 或算法阶段。
- tuner 只枚举合法 tile/occupancy 参数；它没有重新选择算法 flow 或恢复被删除的 stage。
- `report/baseline-new/` 只保留六张 CSV，没有过程 Markdown 或检查脚本。

## 可作为下一阶段起点的状态

已经做完的是：完整 source/registry 分母、三家统一的执行链、无静默数值失败的跨设备固定表、stage 越界路径清除、明确的失败阶段，以及 recurrent/contract-axis/region-boundary 一批共享与 provider realization 修复。

尚未做完的是：typed runtime domain extent、两种真实 FP4 format contract、TileLang provider 覆盖面，以及 cuTile/Triton/TileLang 上若干已经由 source 数字证明的高质量 attention/MLA/INT2 physical form。它们都已有责任层和可复现 entry，不再混成“编译失败”或“下层就这样”。
