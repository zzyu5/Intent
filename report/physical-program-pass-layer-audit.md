# 从 Physical Program 出发的 pass 层审计

## 结论

这一轮没有沿着生成的 Triton 源码倒推缺什么 form，而是直接读取 Physical Program、修改其中由 shared policy 选出的结构决定，再在真实 kernel 上做数值与性能 A/B。

结论分成四条：

1. **当前 Physical Program 不是一张只含 decision 的旁表。**它同时包含 `intent_plan.exec_*` 形式的算法 SSA/region/effect，以及 `axis`、`range`、`transfer`、`contract`、`stream_binding` 等已选物理决定。因此，provider pass 从当前 physical function 读取索引关系或 use-def，并不等于回旧 Kernel IR 重建；把这些可以重算的关系再复制进 decision op，反而会制造第二份算法真理。
2. **Mamba3 的二维带步长 descriptor 不是 Plan 信息丢失。**Triton provider pass 从 Physical Program 中的 transfer、index relation、selected range 和 SSA provenance 推导 `linear/strided` descriptor form，并把最终 form 写回 provider attributes；terminal 只消费已选 form并机械打印 offset。这里缺的是 provider 派生分析的模块化程度，不是 shared Plan 再存一份 stride schema。
3. **shared worker/fold policy 存在一个真实、已实测的欠定维度。**把全标量 program space 折叠成一维，能让 paged MLA 在 5090 和 H100 上分别快约 11.3% 和 3.7%；同一个改法却让 H100 paged GQA 慢约 58.4%。这证明当前“最多分配三个 worker 轴”的固定规则不是普遍最优，也证明“只要全是标量轴就折叠”不是合法修法。
4. **不能把 Triton 的隐式 shared-memory/stage 资源模型复制进 shared pass。**当前 shared residency pass 只知道 Intent 显式 private buffer 的容量与生命周期，不知道 Triton lowering 为 `num_stages` 隐式分配多少 shared memory。Triton 本身通过真实 compile/run 把 `OutOfResources`、PTXAS 和编译断言失败候选记为无穷大；TileLang 普通 AutoTuner 也在编译/benchmark 阶段跳过失败候选。shared pass 应收窄结构与 shape 上可证明的合法范围，provider 编译器仍负责它自己产生的隐式资源合法性。

因此，这一轮只提交审计报告，不提交那个跨设备错误的 worker-fold 修改，也不新增一套重复 Triton 资源模型。

## 一、审计边界：Physical Program 里到底有什么

当前唯一执行链是：

```text
Kernel IR
  -> Construct Physical Program
  -> shared GPU refinement passes
  -> provider form passes
  -> terminal translation
```

Physical Program 中的事实分两类：

- `intent_plan.exec_*`：算法操作、SSA、region、effect 和精确索引表达式的 physical-program 版本。它们仍是算法语义与 provenance 的唯一来源。
- decision ops：多个合法物理实现里已经选中的答案，例如 program worker/fold、range purpose/tile、transfer materialization、contract form/residency、stream binding、persistent traversal。

因此本轮采用以下判据：

- provider 读取 `exec_*` 的精确地址表达式、effect 或 use-def，然后选择 provider-native form：合法的 provider analysis；
- provider 重新决定 ownership、range、worker/fold、stream end 或 shared residency：越过 Plan 边界；
- terminal 根据已选 provider form 翻译算法表达式：合法的机械投影；
- terminal 再判断应该选 pointer、descriptor、primitive 或 blocking：隐藏的第二个编译器。

这个区分修正了一个容易误判的说法：**“provider 读 relation”本身不是问题；问题是它读 relation 后是在重算派生表达式，还是在重新选择上层已经应当选定的结构。**

## 二、当前超标样本的 Physical Program 审计

样本取两张固定 Triton 表中 `ratio > 1.05` 的并集。详细展开的六个代表样本来自实际 emitted Physical Program；其余项核对了对应 physical op、range/axis 绑定和历史定向归因。

| entry | Physical Program 已有事实 | provider 是否必须补 shared 事实 | 本轮判断 |
|---|---|---|---|
| `flash_attention_forward` | B/H 标量 ownership，Q `program_m` ownership，K 同时是 ordered/reduction/lane/contraction-N；K traversal/reduction 为 `stream_contract`，lane 为 row-vector；两次 contract；prefix boundary 与 neutral mask 已绑定 | 否 | Plan 足够；全 worker 折叠 A/B 无收益，关闭 boundary neutralization 也未改变当前形态 |
| `mamba3_siso_forward` | B/head 标量 ownership，value 轴 `program_m`，多个固定 reduction/lane 轴，scan carry/accumulator residency，四个 direct contract，transfer 与 autotune roles | 否 | strided descriptor 是 provider 从 physical transfer relation 派生的 form；全 worker 折叠反而慢约 8% |
| `paged_gqa_decode` | B/HQ 标量 ownership，page ragged-member traversal，token ordered stream/reduction/lane，两层 stream binding，score/value contracts | 否，但 worker dimensionality 的性能选择欠定 | 5090 对一维/二维映射不敏感；H100 一维化显著退化 |
| `paged_mla_decode` | B/HQ 标量 ownership，page ragged-member traversal，token ordered stream/reduction/lane，content/rope/value 三次 contract | 否，但 worker dimensionality 的性能选择欠定 | 一维化在两台机器都更快；不能据此写 kernel/shape 特判 |
| `flaggems_batch_norm_training` | C ownership，S ordered stream/reduction/lane，B 独立 row-vector reduction，两条 stream binding | 否 | 过去“Plan 没有 B blocking”的归因不成立；DSL 与 source 的 Welford/分解结构不同，不能据此改 shared blocking |
| `scaled_fp8_splitk_gemm` | split ownership，K reduction，M/N grouped program ranges，loop-carried replay contract，atomic write语义仍在 `exec_*` | 否 | 两机相对 source 的方向不一致，没有 shared policy 错误证据 |
| `padded_rope_cache_update` | B/head ownership，half-dimension pointwise lane，带 offset 的 access range，三路 branch/store effect 保存在 `exec_*` | 否 | source 先选地址再执行一次 body，DSL 写了三路 effectful branch；这是作者程序结构差异，不是 Plan 丢事实 |
| `paged_gqa_decode` / `paged_mla_decode` 的 upstream 对照 | shared Plan 都能完整表达当前单-kernel算法 | 不适用 | upstream 是 split-K/多 launch 分解；大 ratio 不能直接归成 shared pass 性能差距，worker A/B 只用于检验当前算法自己的 policy |
| `mamba_chunk_state` | stream、reduction、contract 与选定 tile roles 均存在 | 否 | 5090 超 5%，H100 反向更快；没有稳定 shared 缺口证据 |
| `splitk_paged_attention` | split ownership、paged stream、partial contract 和第二个作者 kernel 的边界显式存在 | 否 | 两机方向反转，没有稳定 shared 缺口证据 |
| `flaggems_roll` | pointwise ownership/lane 与精确 index relation 存在 | 否 | 约 6% 且绝对值很小；未发现 provider 回猜 range |
| `qkv_projection_pipeline` | 三个作者 kernel 各自拥有完整 M/N/K contract plan | 否 | 约 6%；不是一个 Physical Program 内缺 stage 或 contract fact |
| `flaggems_softmax_backward` | row ownership、reduction/lane range 和 fill 已绑定 | 否 | H100 约 1.11，5090 不超标；绝对差为微秒级，未发现 shared 决定缺失 |
| `flaggems_triangular_solve` | batch ownership、ordered recurrence 与 value residency 均可读 | 否 | 仅 H100 约 1.09；没有跨设备稳定的 shared policy 证据 |
| `mamba3_siso_step` | B/head ownership与固定小 contraction axes 完整 | 否 | H100 1.051 临界超标；没有 Plan 信息缺口证据 |

### Mamba descriptor 的层次结论

Triton provider pass 当前做的是：

1. 读取 Physical Program 的 `TransferOp`；
2. 沿当前 physical function 的 SSA use-def 解析受限仿射索引；
3. 将索引源绑定到 Plan 已选的 ownership/traversal/reduction/lane range；
4. 选择 `pointer` 或 `pointer_or_descriptor`，以及 `linear` 或 `strided`；
5. 将最终 form、block axes 和 layout 写成 provider attributes；
6. terminal 按这些 attributes 生成 descriptor shape、origin 与 offset。

这里 terminal 仍有较厚的仿射表达式拼写代码，但它没有重新选择 descriptor form。把精确 offset 字符串或 stride 再存进 shared Plan 不会让架构更干净，只会复制算法表达式。若后续继续解耦，应当收敛成 provider-local derived analysis/IR，而不是扩大 shared decision schema。

## 三、shared decision A/B

所有实验都保持 DSL、Kernel IR、range、tile、contract、stream 和 provider form 不变，只改 shared worker/fold 或 boundary policy。数值对照均通过。

### A/B 1：把所有 program axes 折到一个 worker

5090：

| entry | 当前 policy | 全折叠 | 变化 |
|---|---:|---:|---:|
| `flash_attention_forward` | 2.691488 ms | 2.694752 ms | +0.12% |
| `mamba3_siso_forward` | 0.304800 ms | 0.329376 ms | **+8.06%** |
| `paged_gqa_decode` | 0.517760 ms | 0.517760 ms | 0.00% |
| `paged_mla_decode` | 5.444480 ms | 4.831760 ms | **-11.25%** |

这个实验同时证实两件事：

- 当前多 worker policy 不是普遍错误；Mamba 的 tiled value axis 与外层标量 axes 不能无条件折叠。
- 当前 policy 也不是普遍最优；MLA 的全标量 program space 明显受益于一维化。

### A/B 2：只在“所有 program axes 都是标量 ownership”时折叠

5090：

| entry | 条件折叠结果 | 与当前 policy 比较 |
|---|---:|---:|
| `flash_attention_forward` | 2.691424 ms | 不触发，持平 |
| `mamba3_siso_forward` | 0.304800 ms | 不触发，持平 |
| `paged_gqa_decode` | 0.517056 ms | 触发，持平 |
| `paged_mla_decode` | 4.829584 ms | 触发，约快 11.3% |

仅看 5090，这条规则似乎可以提交；H100 同轮 A/B 否定了它：

| entry | H100 当前 policy | H100 条件折叠 | 变化 |
|---|---:|---:|---:|
| `mamba3_siso_forward` | 0.364160 ms | 0.364160 ms | 不触发，持平 |
| `paged_gqa_decode` | 0.755680 ms | 1.197168 ms | **+58.42%** |
| `paged_mla_decode` | 5.804832 ms | 5.589472 ms | **-3.71%** |

所以这一规则已经被真实跨设备结果否决，未进入主工作树。

### 为什么不能再补一个 `HEAD_GROUP` 阈值

MLA 和 GQA 的 DSL 都是 `key_head = query_head // HEAD_GROUP`。当前 facts 能表达 program axes、contracts、selected ranges 和精确 SSA，但没有一条 typed physical fact 表达“这个 program axis 的相邻实例对关键输入有多少跨 program reuse”。

可观察到的算法差异是：MLA 的 `latent_block` 同时供 score contract 和 value contract 使用；GQA 的 key/value 来自两个独立 load。但把这个 use-def 差异直接当成“折叠 grid 必然更好”的证明仍然过强——寄存器内复用发生在单个 program 内，而 A/B 改变的是跨 program 调度与 cache locality。

因此当前只能得出：

- worker dimensionality 是一个真实 shared structural decision；
- 当前固定 policy 在 MLA 上选得不够好；
- 现有 typed facts 不足以写出经这两个反例验证的通用选择规则；
- 按 kernel 名、HQ/KVH 数值或某个魔法阈值补条件会把缺失的 cost/reuse 事实伪装成规则，不能接受。

### A/B 3：关闭 boundary neutralization

5090：

| entry | 当前 | 关闭 neutralization | 变化 |
|---|---:|---:|---:|
| `flash_attention_forward` | 2.691488 ms | 2.693632 ms | +0.08% |
| `paged_gqa_decode` | 0.517760 ms | 0.517808 ms | +0.01% |

这两个样本上该 pass 没改变最终关键形态，不能用它解释当前 ratio，也没有证据支持扩大或删除这条 policy。实验只说明：当前两格不是 boundary-neutralization 回归。

### 跨 provider 定向核验

同一个 shared worker/fold 变体在 5090 上还做了以下 A/B：

- cuTile `grouped_flash_decode`：旧 0.220832 ms，新 0.219664 ms，持平；
- cuTile `absorbed_mla_decode`：旧 4.544032 ms，新 4.550416 ms，持平；
- TileLang `paged_mla_decode`、`gqa_decode`：新旧 Plan 都在 provider legality 阶段以同一条“logical contraction row extent 小于 native MMA fragment 16”诊断拒绝，没有出现由 shared 变体引入的新路径。

这也说明固定 CSV 与当前定向运行之间的绝对差不能直接归因给本轮 Plan 变体；必须做同环境 old/new A/B。

## 四、tuner 参数合法范围审计

### 当前各层实际承担的内容

| 参数/候选 | shared/provider pass 提供的事实 | runtime tuner 当前行为 | 最终资源合法性责任 |
|---|---|---|---|
| tile roles：`stream`、`scan`、`stream_contract`、`program_m/n`、`group_m` 等 | shared Plan 只声明最终出现了哪些可调 role；静态/符号 extent可用于上界 | Python 中有手写候选表；按 runtime extent 做 power-of-two 上界裁剪 | provider 编译器验证具体实现 |
| `num_warps` | Plan 不选择数值 | 与 role profile 一起形成候选 | Triton compile/JIT；超资源候选被淘汰 |
| `num_stages` | Plan 不选择数值；private residency pass只估算 Intent 显式 buffer | 与 role profile 一起形成候选 | Triton lowering/编译器知道隐式 pipeline shared memory，实际编译淘汰 |
| `USE_TMA` | Triton provider pass证明 transfer 可形成 descriptor候选 | runtime 检查 alignment、rank、stride、contiguity 等调用时条件 | Triton 编译器验证 descriptor 与资源 |
| `USE_NATIVE_SCALED` | provider pass证明 native scaled primitive候选 | 作为二值 provider-local候选 | Triton 编译器验证 primitive |
| row occupancy / pipeline stages | provider row form决定是否需要这类参数 | 静态 profile，实测选优 | provider compiler/JIT |

### 哪些范围是算出来的

- selected ranges 的参数名来自最终 shared Physical Program，不是 runner 猜出来的；
- 静态 extent 会把候选裁到 `power_of_two_ceil(extent)`；
- runtime symbolic extent 通过 `early_config_prune` 裁剪；
- descriptor 候选还会检查 runtime view alignment、rank、末维连续性、stride 对齐和可选的整体 contiguous 要求；
- provider form pass 先决定某个二值 form 是否结构上合法，非法 form不会进入候选。

### 哪些仍是经验候选表

- 每种 tile role 的具体数值集合；
- role 组合对应的 `num_stages`/`num_warps` profile；
- row occupancy 与 pipeline stages profile；
- 某些 provider-local 等价拼写的候选集合。

这些表是性能候选 surface，不是“设备合法空间”的形式化模型。

### 为什么本轮没有在 shared pass 里静态裁 `num_stages`

用户提出的怀疑是合理的，但对当前实现的事实前提只成立一半：

- `RefinePrivateBufferResidency` 的确读取 `registersPerUnit`，并按显式 logical buffer 的 shape、dtype、lifetime owner 选择 scalar array、private vector 或 global workspace；
- 它不知道 Triton 把某个 `tl.dot`、descriptor、软件 pipeline 和 `num_stages` 组合 lowering 成多少隐式 shared memory，也不知道 PTXAS 最终 register allocation。

ref 中 Triton 的真实做法是：

- `early_config_prune` 和 `perf_model` 是可选 hook；官方 attention 示例的早裁只处理 `BLOCK_M <= N_CTX`、causal 等 shape/语义条件；
- `Autotuner._bench` 实际 compile/run 每个候选，捕获 `OutOfResources`、`CompileTimeAssertionFailure` 和 `PTXASError`，将候选时间记为无穷大；
- 再从其余候选中选择最小实测值。

TileLang 普通 AutoTuner 同样逐候选编译，编译失败或 benchmark timeout/error 就跳过；其 Carver/Roller 有可选的资源感知候选生成，但不是所有 TileLang autotune 的统一静态 pass。

因此本轮的边界结论是：

- **应由 Intent 提前裁掉**：从算法与 Physical Program 可证明的 shape 不合法、结构不合法、provider capability 不满足；
- **不应由 Intent 复制模型**：Triton/TileLang lowering 自己产生的隐式 shared memory、register allocation、PTXAS 资源失败；
- **当前真实缺口**：role/profile 候选表仍是经验集合，未记录为什么这些值构成合理性能 surface；但把它改成伪精确的跨 provider 资源公式不是修复。

## 五、本轮实际运行范围

没有做全量，没有更新六张固定 CSV。所有输出写在 `/tmp`，只用一条 runner 路径完成 emit、provider compile/JIT、真实 GPU launch、数值比较与计时。

5090：

- Triton：`flash_attention_forward`、`mamba3_siso_forward`、`paged_gqa_decode`、`paged_mla_decode`；
- cuTile：`grouped_flash_decode`、`absorbed_mla_decode`；
- TileLang：`paged_mla_decode`、`gqa_decode`，确认相同 provider capability 诊断；
- 另做 worker 全折叠、全标量条件折叠、关闭 boundary-neutralization 三组临时编译器 A/B。

H100：

- 在远端 `/tmp` 独立展开当前提交和临时 Plan 变体，构建独立 `intent-compile`；
- Triton：`mamba3_siso_forward`、`paged_gqa_decode`、`paged_mla_decode`；
- 对 paged GQA/MLA 用同一个 checkout、同一个环境、同一轮分别运行旧/新 policy，排除了固定表漂移造成的假归因。

所有临时 probe 都不进入语料、registry 或测试设施。

## 六、最终状态

### 已确认成立

- shared pass 的结构不是“完全没问题”：worker dimensionality policy 在 paged MLA 上确有稳定性能损失；
- provider 读取 Physical Program 内 `exec_*` use-def 与 relation 是合法派生分析，不应机械复制到 shared decision schema；
- Mamba3 strided descriptor 缺口已经位于 provider form，而非 Plan 丢失二维访问语义；
- tuner 的 extent/descriptor legality 已有可证明裁剪；provider 隐式资源失败仍应由 provider compiler/JIT 淘汰；
- 关闭 boundary neutralization 不能解释当前两个 attention 样本的差距。

### 仍未关闭，但性质已经明确

- **program-space dimensionality 的 shared policy 欠定。**现有 facts 不足以在 MLA/GQA/H100/5090 四个结果之间写出无魔法常数的唯一规则；当前架构又只允许固定输入产生一份 canonical Physical Program，不能擅自把它变成结构 autotune。
- **Triton provider 的 affine descriptor analysis 仍较厚。**它当前职责正确，但派生逻辑与 terminal 的表达式翻译可以进一步模块化；这不是 shared Plan 字段缺失。
- **runtime profile 是经验性能 surface。**它不是资源 legality 证明；当前也没有证据要求 Intent 复制下层资源模型。

这三项不能混成一句“leaf 还不够好”：第一项是真 shared policy 缺口，第二项是 provider analysis/translation 组织问题，第三项是参数候选质量与下层资源责任边界。
