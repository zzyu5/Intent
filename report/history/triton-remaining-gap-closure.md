# Triton 剩余结构缺口收口报告

## 1. 范围与最终结论

本轮没有扩 registry，没有修改 public DSL、Kernel IR、shared GPU pass，也没有修改 cuTile 或 TileLang provider。最终保留的代码改动只有 Triton provider form、Triton terminal projection 和 Triton runtime tuning。

四项的最终结论并不都是“按原假设实现”：

| 项目 | 最终处理 | 5090 | H100 |
|---|---|---:|---:|
| Mamba3 stride descriptor | 新增独立的 strided-ND descriptor form，并保留原 linear form | `2.006445x → 1.784455x` | `2.382852x → 2.329580x` |
| padded RoPE if-conversion | 未实现；当前 DSL 的三臂 effect schema 不满足通用合法性 | `1.803714x` | `1.679058x` |
| block-sparse 资源失败 | 原“必须 sub-tile”判断被实测推翻；补 pipeline 参数候选并把调优移出计时区 | `failed → 1.002216x` | `0.900851x → 0.848171x` |
| BatchNorm shared blocking | 未修改；Plan 已有 B blocking，source 也不是 B-program ownership | `1.465636x` | `1.322214x` |

其中 block-sparse 是本轮最重要的责任纠正：5090 的失败不是“128-token logical page 只能再切成 2×64”，而是旧 kernel 固定使用 Triton 默认三阶段 pipeline。保持完全相同的 128-token body，只把 `num_stages` 放入合法 provider 参数候选，`num_stages=2` 即可把 shared-memory footprint 从 `133120 B` 降到约 `66816 B`，无需改变程序结构。

## 2. Mamba3：linear 与 strided descriptor 分形

### 2.1 新增的 provider 表示

Triton provider pass 不再用一个笼统的 `pointer_or_descriptor` 表示所有 descriptor。每个候选 transfer 现在额外记录：

```text
intent_plan.triton.descriptor_layout = linear | strided
intent_plan.triton.descriptor_block_axes = [...]
```

两种 form 的责任不同：

- `linear`：原有高质量路径。一个外层 block axis，加连续完整末维；外维按 row-major 线性化成二维 descriptor。
- `strided`：新增路径。保留 ABI view 的 N 维 shape 和 runtime stride；标量索引维的 block 为 1，两个或更多物理 block 维直接成为 descriptor block shape。Mamba 的 sequence×feature 二维窗口因此可以在原四维 ABI view 上机械表达，不需要 materializer 重新猜一个临时二维 view。

strided form 只接受受限的矩形 affine index：常量、精确标量索引、单一物理来源轴以及常量乘加；不接受 data-dependent gather、多个非标量来源混合或非矩形关系。唯一允许穿过的 gather 是 valid 恒真、索引关系只含 `full_slice/new_axis` 的 shape-only projection；它不读取新的下标，不改变 operand 0 的数值或 affine 来源。运行时继续检查 base 16-byte 对齐、末维 stride 为 1、外维 byte stride 16-byte 对齐和末维字节宽度。

terminal projection 只做三件事：

1. 按 provider pass 已记录的 axis mapping 拼 descriptor shape、stride 和 block shape；
2. 沿 SSA use-def 机械打印精确的 affine origin；
3. 将 descriptor 返回的 block reshape 回作者值的逻辑结果 shape。

它不按 Mamba 名称匹配，也不从邻近 contraction 反推 transfer form。

### 2.2 为什么必须保留两种 form

本轮中间 A/B 原型曾让所有 descriptor 都改走 N 维拼写，H100 的 `flash_attention_forward` 在该原型上从约 `1.96 ms` 退到 `2.05–2.19 ms`（这不是固定表的最终数字）。并排源码后确认，旧 linear form 被多余的 reshape/validity 物化覆盖了。

最终实现把 form 明确分开：linear 恢复原生二维 load/store 拼写，strided 才使用 N 维 shape/stride/origin。最终回归结果：

- 5090 FlashAttention：`2.687552 / 2.756608 ms = 0.974949x`；
- H100 FlashAttention：`1.943008 / 1.744928 ms = 1.113518x`。

两台机器都不弱于本轮前固定表，说明新 form 没有覆盖旧路径。

### 2.3 Mamba3 的实际收益与剩余差距

当前 Mamba3 生成源码中有 6 个 strided descriptor，5090 tuner 选择 `USE_TMA=1`。最终数字：

- 5090：`0.304528 / 0.170656 ms = 1.784455x`；
- H100：`0.364160 / 0.156320 ms = 2.329580x`。

descriptor 表示缺口已经闭合，但它没有消除全部差距。上游一次载入完整 Q/K feature block，再在片上拆成偶数/奇数两半；当前 DSL 明确写成了两次不同索引的 transfer。把这两次作者 transfer 自动合并为一次，再重写后续 use-def，已经不是 transfer form 的机械投影。本轮没有在 provider pass 中做这种算法表达改写。

## 3. padded RoPE：当前程序不满足 effectful if-conversion

### 3.1 对原假设的核验

Triton reference 中没有一个通用 pass 会把任意带 store 的 CFG 分支变成“select pointer + 单份 effect”。现有相关变换只覆盖 masked load/select peephole 和 pointer representation canonicalization，不移动或合并任意 store effect。

更关键的是，当前 Intent DSL（`examples/kernels/position/rope_cache.py`）三个分支不是“同一份 body、只有地址不同”：

- Q 分支写 `query_output` 两次；
- K 分支写 `key_cache` 两次；
- V 分支写 `value_cache` 一次，而且不执行 rotation 的同一数值路径。

source（`source/triton/xformers/position/rope_padded/rope_padded_kernels.py`）则先选 q/k/v 地址，始终形成两路结果；V 路用 select 取回原值后仍按统一的两-store schema 写出。把当前 DSL 变成 source 形态会投机执行作者没有写在 V 分支里的 rotation，并改变 effect 数量。这是程序结构改写，不是合法的 provider if-conversion。

### 3.2 最终决定

本轮没有添加 kernel-name matcher，也没有添加一个 legality 不完整的 effect fusion。当前数字如实保留：

- 5090：`0.054400 / 0.030160 ms = 1.803714x`；
- H100：`0.051312 / 0.030560 ms = 1.679058x`。

以后若要支持更窄的通用 if-conversion，至少必须由 typed facts 同时证明：分支 predicate 互斥、每臂 effect 数量和 schema 一致、pointer/value type 与 shape 一致、无 barrier/volatile/alias/ordering 冲突。当前 kernel 不满足这组前提，因此不能拿它作为该 form 的首个消费者。

## 4. block-sparse：资源缺口实际属于 pipeline 参数

### 4.1 原 sub-tiling 方向为何没有保留

一次性原型曾把 128-token logical page 在 Triton provider 内 replay 为 32/64/128-token 子段。5090 可以运行，但 H100 从原来的约 `0.078 ms` 退到约 `0.094 ms`。该原型已经删除。

继续并排检查本轮两份实际 compiled-kernel metadata 后发现（下列容量是编译产物证据，不是静态估算）：

- 原 body，`num_stages=3`：shared memory `133120 B`；H100 可运行，5090 超限。
- 同一原 body，`num_stages=2`：shared memory 约 `66816 B`；两台机器都合法。

因此没有证据需要改变 physical body，更不需要触碰 logical page ABI。真正漏掉的是已有下层参数：该形态没有 search space，过去直接吃了 Triton 默认 pipeline depth。

### 4.2 最终 provider form

Triton provider pass 为满足以下 typed 条件的 stream 记录 `pipeline_candidates`：

- stream body 没有不可 replay 的写 effect；
- 恰好一个完整 fixed、power-of-two、至少 128 的 inner reduction range；
- 该轴承担 `contraction_n`，不承担 `contraction_k`。

这不是 attention 或 sparse 名称判断。form 不改变 body，只声明 `num_stages/num_warps` 的合法 provider 参数候选。

短 kernel 还有一个测量问题：若每次 launch 都经过 Triton `Autotuner` Python wrapper，CUDA event 会把 start event 与真正 kernel enqueue 之间约 `0.012 ms` 的 host dispatch 空档算进去。最终 wrapper 在准备阶段让 `Autotuner` 执行它自带的已选-config launch，然后返回直接调用已选 JIT function 的闭包；不再为了构造闭包重复 launch 一次，计时区内只剩所选 kernel launch。对 inout kernel 不启用这条直接路径，避免绕开 Triton 的 restore hook。

最终结果：

- 5090：从 `provider_jit_or_initial_launch_failed` 变为 `0.086848 / 0.086656 ms = 1.002216x`；
- H100：`0.071952 / 0.084832 ms = 0.848171x`，相对旧 generated `0.077920 ms` 继续提升。

## 5. BatchNorm：没有缺失的 B ownership

上一轮报告把 BatchNorm 归因为“Plan 没有独立 B blocking”，本轮代码与生成物核验推翻了该结论：

- source 也是每个 program 拥有一个 channel C；B 和 S 都在 program 内部处理，不是 B-program ownership。
- 当前 Physical Program 已给 B 轴记录 reduction/lane 的 `row_vector` physical range，当前 case 的 B tile 是 32。
- DSL 与 source 的主要差别是作者数值分解：DSL 用 S state-stream 做 chunk statistics/Chan 合并，source 在 B×S tile 上维护 Welford 数组；不是漏了一个 B program axis。

把 B 自动提升成 program ownership 会让多个 program 共同产生同一 channel 的统计量，需要 partial/atomic/第二次归约，等于替作者改变 reduction decomposition。它不属于 shared blocking 的合法实现自由。

一次性候选曾加入 source 相近的 S=512/16-warps 组合，5090 没有收益，已删除。最终没有修改 shared 层，因此也没有必要修改或重测 cuTile/TileLang 固定表。

当前数字：

- 5090：`0.057664 / 0.039344 ms = 1.465636x`；
- H100：`0.055808 / 0.042208 ms = 1.322214x`。

## 6. 实际验证与固定表

本轮实际执行了本地与 H100 的独立 C++ build，并在两台机器上定向运行：

- `mamba3_siso_forward`；
- `flash_attention_forward`（descriptor 回归消费者）；
- `block_sparse_gqa_decode`；
- `padded_rope_cache_update`；
- `flaggems_batch_norm_training`。

每次 runner 都包含 generated/source 实际 GPU launch、数值对照和 p50 timing。等价的手动复现命令为：

```bash
PYTHONPATH=python:examples /home/kingdom/.venvs/intentdsl-mlir20/bin/python \
  -m repro.v2.runner triton \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --output /tmp/triton-targeted.csv \
  --kernel block_sparse_gqa_decode
```

固定数字已更新：

- `report/baseline-new/triton-5090.csv`；
- `report/baseline-new/triton-h100.csv`。

未修改 cuTile/TileLang CSV。

## 7. 收口后的真实状态

- 二维带步长 descriptor 已成为通用 Triton provider form，但 Mamba 剩余差距不再是 descriptor 表示缺失，而是作者 DSL 与 source 的 transfer decomposition 不同。
- block-sparse 5090 资源失败已经关闭，而且没有改变 logical page、恢复 compiler-private stage 或加入设备架构分支。
- padded RoPE 不是当前 legality 下可做的通用 if-conversion 消费者；继续追这一格应先由作者写成统一 effect schema，不能由 compiler 改写。
- BatchNorm 没有缺失 shared B ownership；继续追性能应比较两份作者算法的 Welford decomposition，而不是再扩 blocking 权限。
