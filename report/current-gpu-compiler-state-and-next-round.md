# 当前 GPU 编译器状态与下一轮范围调查

## 0. 调查边界与结论

本报告针对提交 `5ec72faf2a700db9815e6f7830269c6c1b51918f` 取证。调查完整阅读了
`AGENTS.md`、`doc/` 当前规格、`report/triton-foundation-completion.md`、
`report/shared-gpu-reconstruction-completion-audit.md` 和 `report/history/` 下九份历史报告。
历史报告只用于恢复“当时声称过什么”；下面的状态判断全部重新来自当前代码、当前编译产物和
当前同语言 source。

这次调查得到的首要结论不是“119 个 Triton provider gap 需要逐项补齐”，而是更上游的一处
shared construction 不变量破坏：当前 KIR-to-GPU construction 会把 launch-visible runtime
dimension 直接写进 `FragmentType`。规格和 `FragmentType::verify` 都禁止这样做，但
`RelWithDebInfo -DNDEBUG` 构建绕过了 MLIR storage/type construction 的断言，随后
`VerifyGPUProgram` 又只检查 runtime symbol 是否已声明，于是非法类型在 release 构建中继续流过
shared pipeline。当前结果因此依赖构建模式：

- 开启断言的当前构建：`46/217` 通过 shared gate，`171` 失败；
- `RelWithDebInfo -DNDEBUG` 当前构建：`214/217` 通过 shared gate；
- 同一个 release 编译器继续走到 Triton terminal：`98` 个 kernel 到 terminal source，`116`
  个停在 provider program verification，`3` 个停在 shared physical program。

也就是说，历史 `217/217` 和当前 release `214/217` 不能解释成“shared physical program 已经
完整”；它们首先证明的是 release 构建没有执行一条已经存在的 type-construction invariant。
历史报告中的 `100 + 7 + 5 + 3 + 1 + 3 = 119` 分类在 release 路径上仍能重现，但其中最大的
`100` 类是非法 shared representation 的下游症状，不是 Triton 语言能力缺口。

下一轮主线必须先恢复**构建模式无关、从 construction 起就合法且完整的 shared physical
program**，再谈 Triton provider 横向补齐。现在不应开始 cuTile 第六轮，也不应跑 H100 全量。

## 1. 当前 HEAD 的取证方式

调查没有运行 GPU、source runtime 或 benchmark，也没有更新 baseline CSV。

### 1.1 开启断言的 shared probe

使用当前工作树构建的 compiler：

```bash
cmake --build /tmp/intentdsl-build -j4

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:examples:. \
/home/kingdom/.venvs/intentdsl-mlir20/bin/python \
python/intent/compiler/inventory.py \
  --kernel-root examples/kernels \
  --bindings-module examples.shared_compile_bindings \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --device 0 --jobs 4 --verbose
```

结果为：

```text
files=93 kernels=217 passed=46 failed=171
```

其中 `163` 个首诊断相同：

```text
fragment extents must be constant or physical-parameter expressions
```

其余 `8` 个不是可诊断失败，而是编译器断言：

- `nested_jagged_mean_pool`、`nested_document_pool`、
  `nested_jagged_mean_pool_identity`、`nested_sentence_pool` 在把非 fragment type
  `cast<FragmentType>` 时终止；
- `mamba3_siso_forward`、`mamba3_siso_step`、`max_pool2d`、
  `max_pool2d_with_indices` 在构造逆序 `llvm::iota_range` 时终止。

这八个断言不是新的能力边界，而是 transformation 在前置条件不成立时没有 fail-closed diagnostic。

### 1.2 release shared probe 与 terminal probe

`/mnt/hdd/intentdsl-build-native-05b/CMakeCache.txt` 明确记录：

```text
CMAKE_BUILD_TYPE=RelWithDebInfo
CMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DNDEBUG
```

用该编译器跑同一 inventory，结果是：

```text
files=93 kernels=217 passed=214 failed=3
```

三个失败均为 FlashAttention variant 的多 source physical traversal 不 lockstep。之后用一次性 Python
遍历调用现有 `lower_to_mlir` 和 `toolchain.run_compiler`，让同一批 217 个 kernel 继续通过
Triton legalization/serialization，得到：

```text
physical_program                 3
provider_program_verification  116
terminal_source                 98
```

逐 kernel 结果保存在本次调查的临时文件 `/tmp/intentdsl-current-terminal.tsv`，不进入仓库。
这个 probe 不实例化 provider JIT、不 launch GPU。

### 1.3 为什么两种构建会得到不同答案

规格在 `doc/compiler/gpu-program-ir.md:58-72,167-182` 和
`doc/compiler/physical-parameters.md:18-31` 明确规定 fragment shape 只能包含常量和 compile-time
physical parameters。实现中的 `FragmentType::verify` 也准确执行了该规则：
`lib/Dialect/GPU/IR/GPUDialect.cpp:184-205`。

冲突来自 construction：

- `lib/Conversion/KIRToGPU/KIRToGPU.cpp:899-935` 的
  `fragmentExtentForDimension` 在 dimension 能从 launch ABI 看见时，直接返回
  `PhysicalExprKind::Dimension`；
- `KIRToGPU.cpp:995-1052` 的 `convertTensorType` 随后把这个 runtime expression 放进
  `FragmentType`；
- `lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:19-60,238-241,350-362` 对 view、fragment、
  buffer shape 共用 `verifyExpressionSymbols`，只验证 `Dimension`/`ScalarABI` symbol 在 launch
  ABI 中存在，没有再次执行 fragment-specific kind restriction。

`git blame` 给出了明确时间关系：fragment type invariant 来自 `9549983e`；把 launch-visible
dimension 写进 fragment 的分支来自 `bded6edd`，提交说明恰好是
`compiler: establish complete shared GPU corpus gate`。因此不是 verifier 最近“收紧后误伤”，而是
construction 后来新增了一条与既有 type invariant 相冲突的路径。开启断言时 MLIR type storage
construction 抓到它；`-DNDEBUG` 时该断言消失，而 generic module verifier不会重新验证一个已被
构造出来的 type 参数。

这也解释了为什么断言构建有 `163` 个 runtime-fragment 失败，而 release 路径最终只剩 `100`
个：其余非法中间 fragment 恰好被后续 shared rewrite 改成了 compile-time shape。规格要求每个
shared pass 前后程序都合法；“后面可能修好”不能使非法 initial program 合法。

## 2. 119 个未到 terminal source 的真实性质

### 2.1 100 个 runtime-dimension fragment：这种形态根本不该出现

这 100 个在 Triton verifier 的首诊断都是：

```text
'intent_gpu.make_range' op Triton tl.arange physical extent must be a
compile-time physical expression
```

它们不是“已经实现但未触发”，也不是“Triton 不支持动态 shape”。Triton 当然可以在 runtime
extent 上用 static block 加 mask；不能做的是让 `tl.arange` 的 block extent 本身成为 runtime
value。Intent 当前把完整 logical dimension 当作 fragment physical extent，丢失了
“runtime logical extent 与 compile-time owned block”之间本该由 blocking/program mapping 表达的
关系。

所以这类必须在 shared construction/decision 层修：initial fragment 应当从一份合法的
compile-time physical baseline 开始，runtime dimension 只进入 program-space、loop bound、
coordinate 和 validity。不能在 Triton leaf 把 `D4` 猜成某个 block 常数，也不能放宽
`FragmentType::verify`。

### 2.2 七个 mutable buffer：shared form 合理，Triton realization 未实现

七个 kernel 是：

- `smith_waterman_score`；
- `viterbi_decode`；
- `insertion_top_k`；
- `bitonic_sort_rows`；
- `radix2_fft`；
- `varlen_gqa_decode_with_sink_logits`；
- `greedy_nms`。

它们均在 `lib/Target/Triton/Transforms/Legalize.cpp:662-665` 命中一条无条件拒绝：任何
`gpu::BufferOp` 都报“requires provider-local mutable-buffer realization”。这里 shared buffer
本身有算法/资源意义，不应在 shared 层消失；缺的是 Triton-local functionalization、SSA carry
或显式 local allocation 形式。历史第三轮已经把它登记成 implementation gap，当前仍原样存在。

### 2.3 五个 masked gather：已有 gather 路径，但 safe-index legalization 没有实现

五个 kernel 是：

- `compact_nonzero_rows`；
- `unique_consecutive_rows`；
- `varlen_aligned_causal_depthwise_conv1d`；
- `moe_prefix_routes`；
- `mamba3_siso_forward`。

`lib/Target/Triton/Transforms/Legalize.cpp:786-794` 对任何带 `valid` 的 `gpu::GatherOp` 直接拒绝。
原因是 `tl.gather` 的坐标即使 lane 被 mask，仍需要先成为内存安全的坐标；正确实现应在
provider pass 中显式形成 safe coordinate，再在 result 上保持原 fill/validity。当前没有这条
rewrite，因此属于 provider implementation gap，不是 target capability。

### 2.4 三个 provider candidate legality：parameter domain 为空

三个 kernel 是：

- `softmax_backward`；
- `ordered_product_prefix`；
- `max_pool2d_with_indices`。

`lib/Target/Triton/Transforms/Legalize.cpp:142-248` 已实现 physical/provider parameter domain 的
Cartesian instantiation和 typed-fragment legality 过滤，但这三项的所有候选都被过滤掉。它们的
性质是“当前 physical form 与候选域没有交集”：可能缺一个合法 candidate，也可能上游
fragment/ownership 决定本身错误。现有证据不足以把责任推给 tuner，更不能称为 target 不支持。
下一轮应在修复 shared fragment authority 后重新计算这三项；现在的 candidate legality 依赖一份
本来就可能非法的 shared shape。

这里还要区分 CSV 中七个 `candidate_contract_failed`。那七项的 underlying compiler kernel
已经到 terminal source；失败发生在 `examples/repro/v2/measurement.py:72-122`，因为 generated
candidate set 没有符合 source comparison predicate 的配置。它们不是上述三个 provider verifier
失败，也不属于 119 个 terminal-source 分类。

### 2.5 一个 scaled contract：已有 provider form，但当前 schema 不在其 legality 子集

`block_scaled_matmul` 到达 `gpu.scaled_contract`，随后被
`lib/Target/Triton/Transforms/Legalize.cpp` 的 `tl.dot_scaled` rank/format/e8m0 scale/group/axis
legality 拒绝。当前只能说 Triton provider 对这份 canonical scaled schema 的 realization 没有闭合。
它不在 Triton baseline registry，因此仓库里没有同格 source 用来判定应该落 native
`tl.dot_scaled`、显式 decode + ordinary dot，还是明确不接这一格；不能把未经同格 source 验证的
结论升级成 target capability boundary。

### 2.6 三个多 source 不 lockstep：shared alternative decision 缺失

三个失败是：

- `flash_attention_full_causal_stream_fwd`；
- `flash_attention_inline_fwd`；
- `flash_attention_select_fwd`。

`lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp:81-98,1430-1451,1638-1667` 已经不再像旧实现那样
静默拿第一个 source 当 master；它会验证 physical start/extent/step lockstep 并 fail closed。这一
修正是正确的，但当前只存在“所有 source 已经 lockstep”这一种 physicalization，没有先形成共同
segment relation、分别 slice 或显式证明等价的 alternative decision。因此这三项是 shared
region realization 未闭合，不是 provider 问题。

### 2.7 同语言 source 对“target 不支持”的直接反证

registry 中每一个 Triton entry 都带一个同语言手写 source。只要 source 能形成该 program，当前
generated 路径失败就不能归类为 Triton 表达能力边界。三个具有代表性的当前失败说明了差异：

- `scaled_fp8_splitk_gemm`：Intent 在
  `examples/kernels/contraction/block_scaled.py:64-94` 写 split identity、双 reduction-axis
  contract 和 atomic add；手写 source 在
  `source/triton/meta-applied-ai/gemm/fp8_scaled/scaled_fp8_gemm.py:38-92` 用 compile-time
  `block_m/block_n/block_k`、static `tl.arange`、runtime K loop、`tl.dot` 和 `tl.atomic_add`。
  当前 generated component 停在 runtime fragment，不是 Triton 缺 FP8 split-K。
- `block_sparse_gqa_decode`：Intent 的 partial kernel在
  `examples/kernels/streaming/block_sparse_attention.py:23-121` 保存了 selected-block loop、safe
  block、logical slice、attention summary 和 partial scatter；手写 Triton source 在
  `source/triton/vllm/attention/minimax_m3/sparse_attn.py:43-214` 使用 static
  `BLOCK_SIZE_{Q,K,D,H}`、runtime selected-block loop、block pointer、masked load 和两个 dot。
  当前 partial component 是 runtime fragment，combine component已经到 terminal source。
- `flaggems_fp8_mqa_logits`：Intent 在
  `examples/kernels/routing/mqa_logits.py:11-49` 表达 head/reduction/key axes、contract、head reduce
  和 key interval mask；手写 source 在
  `source/triton/flag-gems/routing/fp8_mqa_logits/fp8_mqa_logits.py:27-118` 用
  `BLOCK_M/BLOCK_N/BLOCK_D`、H/D loops、`tl.dot` 和 masked store。当前 failure 同样是把 runtime
  Q/K dimension 留进 fragment。

这些 source 不意味着 Intent 应复制手写 block 数值；它们证明 target 已有表达，缺的是 Intent 从
logical relation 形成 compile-time blocked program 的能力。

## 3. shared verifier 与规格不变量

### 3.1 verifier 应同时承担两种不同职责

局部 dialect/type/op verifier 应保证“当前 IR 自洽”：类型参数合法、operand/result schema、region
和 layout 局部一致。construction/shared whole-program verifier 还必须保证“这是一份完整 physical
program”：KIR effect/result coverage、program ownership、access relation、buffer obligation、
structured realization 和 unresolved decision 都已闭合。

`doc/compiler/passes-and-analyses.md:154-184` 明确把这两层分开：construction verifier 验证第一份
完整 physical program，shared verifier 在每个 pass 后验证 ownership、access、buffer、structured
op 和 parameter domain。当前 `VerifyGPUProgram.cpp` 已超过普通 MLIR local verifier，但还没有达到
规格定义的 whole-program completeness。

Triton 的 verifier 不能直接作为“只需自洽”的依据，因为 TTIR 已经是一份作者写好的 block
program。`ref/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp:638-670` 验证 blocked encoding rank、
power-of-two 和 order；`ref/triton/lib/Dialect/TritonGPU/IR/Traits.cpp:13-69` 验证 memdesc shape、
memory space、mutability 和 layout compatibility。这些局部检查不需要证明一个更高层 KIR 被完整
physicalize。Intent 自己选择了 program mapping 和 blocking，所以 coverage 证明不能委托给
Triton。

TileLang 也不是只做 schema check：
`ref/tilelang/src/transform/verify_buffer_init.cc:1-15,46-71` 单独检查 local/shared buffer 的
write-before-read；`ref/tilelang/src/transform/verify_parallel_loop.cc:28-111` 构造另一逻辑 iteration
并证明 store 不发生 data race。成熟实现把具体 physical hazard 作为独立 analysis/pass，而不是用
一个“所有 ID 都出现过”的集合等价替代。

### 3.2 十二条完整性不变量的当前覆盖

`doc/compiler/gpu-program-ir.md:167-182` 的十二条不变量与当前实现逐项对照后，结果如下。

1. **Executable kernel body。**`VerifyGPUProgram.cpp:139-149` 要求 module 中恰好一个 physical
   kernel；这保证当前 per-specialization tool invocation 的单 kernel 形状，但比规格允许一个 module
   含多个 specializations 更窄，也没有证明 origin map 覆盖整个 KIR specialization。
2. **Program/effect/result coverage。**`:285-317,367-377` 只检查 program axis 不重复、group ID
   dense、同组 offset/length 相等、effect-origin ID 集合相等。它没有证明 segment coverage/不重叠、
   logical workset 到 program instance 的双射或 guarded mapping，也不检查 result/member ownership。
3. **SSA type legality。**generic `mlir::verify` 会检查 SSA/region；fragment custom verifier存在，
   但 release type construction 可以绕过，whole-program verifier没有重复关键 type-kind invariant。
4. **Fragment compile-time extent。**规格和 type verifier有，construction违反，shared verifier又把
   runtime和compile-time expression混用。这是本轮已实证的最大缺口。
5. **Access completeness。**`:325-340` 要求 `PhysicalProgramAnalysis::footprint` exact，且非 scalar
   coordinate 有 range provenance；没有证明 active member set 与 validity 等价、fill/effect完整、
   range narrowing 是原 relation 的 subset、或不同 effects 无冲突。
6. **Control completeness。**generic MLIR verifier能检查 block arguments/yields/dominance；当前没有
   检查 physical control 是否保持 canonical ordered dependency、effect/result coverage。
7. **Buffer obligation。**`:99-134` 检查 instance ID、scope/lifetime pair 和 dataflow fact exact；
   没有证明 full initializer 每 instance 恰好一次、ownership/visibility、invocation workspace
   slice owner 或跨 program sharing。
8. **Structured-op completeness。**`GPUOps.cpp` 的 reduce/scan/fold/scan schema verifier检查类型和
   helper region局部形状；没有证明它覆盖原 KIR result/effect，或 segmentation 不引入可观察差异。
9. **Non-atomic conflict。**当前没有 whole-program alias/conflict check。
10. **Runtime grid 只能读 launch-visible metadata。**`:19-45,168-241` 对受限 expression tree 和 ABI
    symbol执行了这条；这是当前闭合程度较高的一项。
11. **Range narrowing proof。**analysis能返回 exact/unknown footprint，但 verifier没有验证每一条
    narrowing 的 subset/identity proof 已和 loop、access、validity 同步改写。
12. **无 KIR clone/side decision authority。**`:257-266` 禁止 `intent` dialect 和未列入白名单的
    dialect；但不检查 unknown executable attribute、unresolved decision record、origin/role side
    field 是否仍影响执行。

另外两项规格要求也未执行：`gpu.ParameterOp::verify` 只检查 candidates 非空，没有检查正数、去重、
role/constraints 与 resource legality；atomic address verifier在
`lib/Dialect/GPU/IR/GPUOps.cpp:991-1009` 明确 `(void)ordering; (void)sharing`，即 memory order 与
logical sharing domain 当前没有被验证。

所以 `217/217` 最多曾表示“在某个 release 构建里，这些 module 的现有 symbol/schema/部分
footprint 检查通过”，不能表示“217 份完整 physical program”。

## 4. 四个 blocking pass：真实 transformation，但决策空间仍是形状门

四个文件当前共有精确 `231` 处 `return std::nullopt` 或 `return failure()`：

- `RealizeRegionFold.cpp`：65；
- `RealizePointwiseBlocking.cpp`：41；
- `RealizeContractionBlocking.cpp`：62；
- `RealizeReductionBlocking.cpp`：63。

这个计数本身不能说明实现错误；大量返回只是 helper 的精确失败传播。真正的问题出现在 pass
entry 的可选空间和失败语义。

这些 pass 不是“假 pass”：它们确实创建 `scf.for`/`gpu.make_range`、改变 fragment types、重写
producer SSA 和 access graph。例如 region fold 在
`RealizeRegionFold.cpp:1480-1528` 创建 physical loop/carry，pointwise 在
`RealizePointwiseBlocking.cpp:1670-1695` 改写 ownership graph。

但它们目前也不是一般的 physical decision procedure：

- contraction 在 `RealizeContractionBlocking.cpp:1351-1448` 要求 direct loads、恰好一个 reduction
  pair、无 batch、lhs/rhs 各一个 free axis、明确 unit-step ranges 和特定 tail；scaled 路径
  `:1953-2259` 更窄；
- reduction 在 `RealizeReductionBlocking.cpp:418-471,2307-2374` 要求 root fragment load、单一
  reduction axis、exact unit-step source；
- region fold/scan 在 `RealizeRegionFold.cpp:81-137,1430-1451,1638-1667` 要求 immutable external
  load、replayable graph、unit-step range 和全 source lockstep；
- pointwise 在 `RealizePointwiseBlocking.cpp:751-823,1285-1304,1741-1755,2188-2261` 只能 replay
  一组受限 pure graph，并要求唯一 delinearize workset 和 launch-visible ownership authority。

当这些前置条件不满足时有两种后果：

- region fold/scan 以及“必须 physicalize”的 dynamic contraction/reduction/pointwise 会直接诊断并
  终止 shared pipeline；`lib/Dialect/GPU/Transforms/Passes.cpp:14-63` 在每个 pass 后再次验证，
  不会把它原样交给 provider；
- `requiresPhysicalRealization == false` 或已有 static/native coverage 时，contraction/reduction/
  pointwise 可以 no-op，只标记/保留现有 form，后续阶段负责消费。

因此这里的主要缺口不是“早退太多”，而是**一个合法输入通常只有一个已编码的 physicalization；
前置形状不匹配时没有另一份合法基线 decision**。不少条件来自当时覆盖过的 corpus 形状，而不是
typed analysis 对所有合法结构给出的决策。

这和第五轮修掉的裸 `operator_kind` 不是同一种局部错误：后者用整数代替语义 enum；这里的语义
类型大多明确，但合法 physical decision 被实现为 family-specific recognizer 的单一成功路径。
二者共同暴露的是 decision authority 不完整，而不能通过“代码里没有 kernel 名”自证成熟。

ref 只能提供组织方式，不能替 Intent 做这个独有决定：

- Triton 作者已经写出 block program；其最接近的通用 baseline 是
  `ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:27-58`：任何尚无 encoding
  的 ranked tensor 都得到一个 typed default blocked encoding，已有 encoding 则保留，需要变化时
  产生 `ConvertLayoutOp`。它不是先识别“root load + single reduction pair”才允许 tensor进入
  physical IR。
- TileLang 作者同样显式写 tile/loop；它没有与 Intent automatic blocking 完全对应的 pass。因此
  “ref 没有同名 pass”不意味着 Intent 不该做，而意味着这项额外责任必须有一个对所有 KIR 都合法
  的 baseline representation，再由 typed analyses选择更好的结构，不能靠四个 family matcher充当
  完整决策空间。

## 5. 第五轮的 31 个损伤到底在哪里

旧 `51/54` 表来自提交
`268b7165b87dadbaeb5e737ed389473767bfcc7c`（2026-08-25，`report: restore baseline v2 matrices`）。
当前表是 `20/54`。新 backend 首次全量记录 `4463693` 已经只有 `45` 个 pass；随后 CSV 一度仍保留
旧 `51/54` 快照。当前 31 个 non-pass 状态是在 `4ac68b5` 重新跑表时首次被记录，不等于全部由
`4ac68b5` 单个提交引入。

当前 HEAD 重新编译每个 registry component 后，可以把这 31 个逐项落到真实阶段。

### 5.1 二十个 entry 含 runtime fragment

十九个 entry 的所有 component 都停在 runtime fragment：

`fused_softmax`、`flash_attention_forward`、`layer_norm`、`flash_layer_norm`、`cross_entropy`、
`fused_add_rms_norm`、`rms_norm`、`xformers_rms_norm`、`scaled_fp8_splitk_gemm`、`jagged_mean`、
`causal_conv1d`、`mamba_chunk_state`、`mamba_chunk_scan`、`paged_gqa_decode`、
`splitk_paged_attention`、`paged_mla_decode`、`flaggems_batch_norm_training`、
`flaggems_logsumexp`、`flaggems_fp8_mqa_logits`。

`block_sparse_gqa_decode` 的 partial component 是 runtime fragment，combine component已经到 terminal
source，因此它是第二十个受该根因影响的 entry。

这二十项与 `KIRToGPU.cpp:903-914` 引入的 runtime fragment branch 有直接代码和当前 diagnostic
证据。不能把它们描述成 20 个独立 Triton leaf gap。

### 5.2 两个 entry 是 masked gather

- `varlen_causal_conv1d`：主 component 是 masked gather，final-state component已到 terminal；
- `mamba3_siso_forward`：masked gather。

### 5.3 两个 entry 在 provider candidate legality

- `flaggems_softmax_backward`；
- `flaggems_max_pool2d_with_indices`。

它们是前述“所有 typed fragment candidate 均非法”的两项 registry component。

### 5.4 七个 entry 已经到 source，失败在 measurement candidate contract

`rotary_embedding`、`swiglu`、`scaled_index_add`、`fp8_groupwise_quantize`、
`mamba_state_passing`、`flaggems_group_norm_backward`、`flaggems_addcmul` 的所有 component 当前都能
生成 terminal source。它们在 CSV 中退化，是 source/generated 公平比较 predicate 找不到同角色
candidate，不是编译器 terminal failure。

因此“31 个可能都是那 100 个 dynamic fragment”被当前证据推翻；精确分布是 20 runtime
fragment、2 masked gather、2 provider candidate legality、7 measurement candidate-contract。

## 6. 历史 gap 的当前状态

历史报告中消失的 gap 并没有全部被关闭。当前代码给出的状态如下。

### 6.1 canonical/KIR 留下的三项仍原样存在

`report/history/canonical-kir-reconstruction.md:151-157` 当时列出的三项目前都未闭合：

- tuple/record logical-buffer element：`lib/Dialect/Intent/IR/IntentOps.cpp:1661-1666` 的 buffer
  element仍依赖 ranked tensor，`KIRToGPU.cpp:3646-3650` 仍直接 cast ranked tensor；
- scaled-contract scale-axis relation：仍由
  `RealizeContractionBlocking.cpp:1977-2014` 根据 rank/reduction pair/group extent重建；
- 一般调用前置条件：public/KIR 仍主要只有 `assume_in_bounds`，见
  `IntentOps.cpp:1621-1627` 和 `KIRToGPU.cpp:4324-4332`。

它们没有被第五轮修复，只是移出了当轮 shared analysis 的报告范围。

### 6.2 第三轮 shared gaps 是部分闭合

- region fold/scan 已经从 shell 变成真实 physical loop/carry/output rewrite；但 multi-source
  lockstep alternative 未实现，本轮三个 variant 实证失败；
- multi-axis reduce 有 decomposition，但 runtime free-axis 与联合 chunking仍是窄路径；
- multi-contract、多个 execution groups 的联合 ownership/reuse/effect realization仍没有一般
  decision，`RealizeContractionBlocking.cpp:2080-2088` 等路径仍要求单一可独立改写形态。

### 6.3 第四轮 provider gaps 多数仍在

mutable buffer、safe gather、sparse physical relation、runtime while、record/multi-output/InOut state、
atomic/scatter 的 provider forms并未横向完成。本轮七个 buffer、五个 gather就是直接证据。
历史 `operator_kind` 裸整数问题已经真实关闭：cuTile 和 TileLang 当前读取 typed
`BinaryOperator`，这是一项已关上的 gap。

但 TileLang 的 `Bufferize.cpp` 和 `FormPipeline.cpp` 仍从 current op/use graph重建 storage/copy/
parallel/pipeline 候选；这不是本轮要修的实现，却说明 provider reconstruction 尚未被 shared
authority 完全取代。

### 6.4 第五轮“一条 gap 都没有”是范围缩窄，不是全部关闭

第五轮确实建立了 `PhysicalProgramAnalysis` 的 current-IR query authority，也修掉了几处重复
推导；但它的验收终点是 release shared verifier。build-mode divergence、100 个 terminal 前的
runtime fragments、既有 canonical gaps 和 provider拒绝证明，“没有遗留 shared design blocker”
是验收边界漏掉问题，不是当前事实。

## 7. 与 ref 的双向差异

### 7.1 ref 有而 Intent 还没有的实际能力

- Triton `ref/triton/lib/Analysis/BufferIndexAnalysis.cpp:17-46,147-204,273-317` 保存
  `base + constantOffset` 和 modulus，能证明 ring-buffer slots 不 alias；Intent 当前 footprint没有
  同等级 modular index relation。
- Triton `ref/triton/lib/Analysis/Allocation.cpp:191-201,340-455,495-526,678-735` 明确执行
  liveness、interference graph 和 offset allocation；Intent 的 shared buffer dataflow主要检查
  init/dominance，不能替代 allocation legality。
- Triton software pipeliner在
  `ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp:20-27,93-167`
  形成 modulo schedule、prologue/epilogue；当前 TileLang `FormPipeline.cpp:99-137` 主要是发现
  candidate loop 后包 `PipelineOp`。
- TileLang `ref/tilelang/src/transform/pipeline_planning.cc:36-143,298-338,634-673` 收集 precise
  BufferRegion read/write、replayability 和 stage dependencies；当前 provider pipeline没有同等级
  typed dependence carrier。

这些差异会在换成多 producer、ring stage、aliasing 或 conditional write 的 kernel 时产生真实
后果，不是命名差异。

### 7.2 Intent 有而 ref 没有、但确实需要的结构

`PhysicalSourceAxis`、`AxisMapAttr`、provider-neutral access footprint、`RegionFoldOp/RegionScanOp`
没有 Triton 中的一一对应物。这不是自动可疑：Triton/TileLang 作者已经写了 block/tile program，
Intent 作者没有写，所以 shared compiler必须保存 logical-to-physical provenance、segment和effect
origin。

真正的问题是这些 authority 是否一路保留并被 provider消费。当前 runtime fragment construction、
四个 family 对 source range/replay 的重复 helper，以及 TileLang provider reconstruction说明它们尚未
完全成为唯一 authority。Intent 多出来的 IR 只有在能替代 leaf 重建时才是能力；目前仍是部分地基。

## 8. 结构纪律与取证装置

### 8.1 空目录

当前 HEAD 的事实与 prompt 中的 `43` 略有差异。调查开始时共有 `59` 个空目录：

- `44` 个是 Plan、Target/Common、旧 shared GPU Realization/Transforms、三家旧 Lowering 及
  include 镜像的空架构叶目录；多出的一个是
  `include/Intent/Target/GPU/Config`；
- `15` 个是 `source/` 下尚无 vendored source 的 corpus leaves。

本轮按目录职责从底向上删除了 44 个架构空叶及由此变空的 12 个父目录，共移除 56 个目录节点。
`source/` 下 15 个叶目录保留，因为它们属于 corpus inventory，不是已删 compiler architecture。
空目录不被 Git 跟踪，因此这项清理不会出现在提交 diff 中。

### 8.2 八个大文件的职责和拆分时机

当前超过 1500 行的八个产品/compiler 文件是：

- `KIRToGPU.cpp` 4729 行；
- `RealizeContractionBlocking.cpp` 2643 行；
- `RealizeReductionBlocking.cpp` 2567 行；
- `RealizePointwiseBlocking.cpp` 2520 行；
- `TileLang/Transforms/Bufferize.cpp` 2074 行；
- `RealizeRegionFold.cpp` 1813 行；
- `IntentOps.cpp` 1809 行；
- `GPU/Transforms/Utilities.cpp` 1675 行。

按行数拆现在不合适。四个 family 文件中重复存在 `sourceRange`、producer replay、`SourcePlan`、
fragment-axis erase/retarget；先按文件切开只会冻结四份 decision authority。下一轮修 runtime
fragment 时，应先把“logical runtime extent → legal physical fragment/ownership/range”形成一个
typed construction/analysis authority，再让各 family只保留各自 mutation。

之后存在自然职责边界：

- `KIRToGPU.cpp` 可分为 physical ABI/workset construction、scalar/region op lowering、typed
  value/access construction；
- contraction/reduction/pointwise/region 文件可各自分离 source/replay analysis consumer 与具体
  structured mutation，但不合并成万能 family pass；
- `Utilities.cpp` 可在 authority稳定后分为 type/projection、relation retarget、cleanup；
- TileLang `Bufferize.cpp` 应等第六/七轮明确 provider-local form 后，按 storage/copy/sync/pipeline
  职责拆；现在拆会把仍然重选 shared structure 的代码永久化。

所以结构清理应伴随下一轮权威重建发生，不能作为主线前的独立大重构。

### 8.3 `inventory.py` 和其它测试装置

`python/intent/compiler/inventory.py` 是 131 行 compile-only corpus gate：它发现 examples、读取固定
constexpr bindings、并发调用 `run_shared_compiler`。它没有被 `python/intent/compiler/__init__.py`
导出，生产 `pipeline.py` 也不依赖它。本轮完成取证后已删除。

对 `python/`、`lib/`、`tools/` 的 test/audit/inventory/probe/check 驱动和调用者检索没有找到第二个
同类装置。`tools/intent-compile`/`intent-opt` 是生产工具；`examples/repro` 是正式手动 repro/
measurement 路径，不属于产品树中的测试脚手架。`__pycache__` 和 `autotuner.log` 是 ignored runtime
artifact，不是 tracked compiler path。

以后要遍历 corpus，应使用临时脚本直接组合公开 frontend/toolchain API，保存命令与结果后删除，
不再往 `python/intent/compiler/` 放 inventory runner。

## 9. H100 是否现在运行

现在不运行 H100，既符合本轮“不跑 GPU”的边界，也有技术原因：同一 HEAD 在 assertion/release
构建下对 shared legality给出互相矛盾的答案，31 个 5090 历史退化尚有 20 个落在这条非法
representation 上。此时生成 H100 表只会把 build-mode/compiler-state 混入硬件差异，下一轮修复
construction 后立刻作废。

H100 有意义的最早时点是：

1. assertion 和 `-DNDEBUG` 构建对 217 个 kernel 给出完全相同的阶段与诊断；
2. construction 不再创建 runtime-shaped fragment或断言终止；
3. shared verifier 的完整性项至少覆盖当前 program mapping/access/buffer/parameter authority；
4. Triton terminal source gate 在同一提交上稳定；
5. 之后才用同一 candidate contract 同时跑 5090/H100。

## 10. 下一轮范围建议

### 10.1 主线：重建合法 initial physical program 与 completeness verifier

下一轮不应以“把 119 个报错补绿”为目标。主线应是一次 shared construction/verifier 横向修复：

1. **消除 runtime FragmentType。**KIR-to-GPU construction 对所有 tensor value先产生一份合法的
   compile-time fragment baseline；runtime logical extents只进入 launch/program mapping、loop、
   coordinate和validity。具体 baseline 是 scalar owned member、已有 physical parameter还是由
   structured op创建的 block，应由 typed axis/use facts决定，不能由 target或 kernel family猜。
2. **让 transformation fail closed，而不是 assertion。**修复四个 fragment cast 和四个
   `iota_range` crash，使任何不满足前置条件的 current IR 给出所属 analysis/decision authority 的
   diagnostic。
3. **补 construction/shared completeness。**至少让 fragment expression kind、program ownership/
   segment coverage、access active set/validity/fill/effect、buffer owner/init/lifetime、parameter
   candidate domain和 non-atomic conflict 成为当前 IR 可验证事实。不要把所有规则塞进一个文件；
   analysis负责事实，verifier只消费并检查。
4. **把四个 family 的共同 source-range/replay/ownership 推导收回统一 analysis。**family pass只
   选择和执行 mutation；遇到 coverage failure先归类“缺哪个 analysis/decision authority”，不在
   失败点加更窄 matcher。
5. **在这之后重新分类 Triton 116。**runtime-fragment 类应随 shared修复消失或变成准确的新
   decision diagnostic；随后才分别实现 safe gather、mutable buffer、scaled contract 和合法
   candidate domain。

### 10.2 完成状态不能只看通过数

下一轮完成应同时满足：

- assertion 与 release 构建对同一 corpus 的结果逐 kernel一致；
- 任何 `FragmentType` 在创建时和 shared verifier 中都只能含 constant/physical parameter；
- 每个 runtime logical extent都通过显式 program-space/loop/coordinate/validity连接到 physical
  fragment，而不是被字符串或 side record解释；
- 每个失败都是 typed stage diagnostic，没有 C++ assertion/abort；
- verifier能说明 mapping/effects/results、access、buffer、structured ops 和 parameter domain为何
  完整，而不是只比较 origin ID集合；
- 至少选取不同结构的真实 kernels检查：换一种等价 producer graph或多 source structure时，
  decision来自同一 analysis authority，不因 recognizer形状改变而静默 no-op/失败；
- Triton provider只消费完整 shared facts，不从 KIR relation/logical shape重建 blocking。

217 的数字仍应记录，但它只能是这些性质成立后的覆盖结果，不能单独作为验收。

### 10.3 可以顺手做与必须后置的事项

可以随主线一起做：

- 在修改到相应文件时按稳定 authority拆 `KIRToGPU.cpp` 和四个 family 的 analysis/mutation边界；
- 修掉因非法 initial fragment暴露的八个 assertion；
- 删除由新 authority取代的重复 range/replay helper和宽松 no-op path。

必须后置：

- safe gather、mutable buffer、scaled contract 的 Triton provider实现，除非 shared主线修复后它们仍以
  同一 typed form准确留下；
- 1.05× 性能、candidate winner、TMA/pipeline等 provider optimization；
- cuTile/TileLang 横向实现；
- H100/5090 全量和 baseline更新；
- 纯粹按行数拆大文件。

### 10.4 第六轮 cuTile 现在不能开始

cuTile 可以继续做只读 capability/source 对照，但不能开始 provider implementation 轮。当前 shared
IR 会在 release 构建中携带规格禁止的 runtime fragment；让 cuTile 在这上面继续 legalization，只会
迫使它从 logical shape/KIR relation重建 block和coordinate，重复第五轮之前的厚 leaf 问题。

开始第六轮的必要前提不是 Triton 某个通过率，而是：shared initial program在所有构建模式下合法，
Triton 已证明能仅消费这份 authority形成 terminal source，剩余 provider gap明确是 target-local
form而不是 shared representation缺口。当前还没有达到这个状态。

