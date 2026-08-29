# 第五轮 shared GPU 重构完成度核查

## 结论

第五轮没有真正关上 shared GPU 重构。

`93` 个文件、`217` 个 `@intent.kernel` 全部通过 full shared verifier 是真实结果，本轮重跑后仍然是：

```text
files=93 kernels=217 passed=217 failed=0
```

但它只证明当前 verifier 接受这 217 份 GPU Program。继续沿同一条生产编译链进入 provider 后，Triton 只有 5/217 到达终端源码，cuTile 为 98/217，TileLang 为 75/217。失败并不全是 shared 层的问题：其中既有 provider transformation 破坏 shared 不变量，也有真实 provider capability 子集，还有 provider-local physicalization 未完成。因此这些计数首先否定的是“217/217 足以支撑完整、可执行、provider-neutral 这一强结论”；再结合下文找到的 shared 语义保持缺口，强结论本身也不能成立。

更严重的是，本轮在 shared family 实现内部找到两处可能改变数值语义的残留：scaled contraction 接受原 validity 后，重写时没有把它合入新 access；region fold/scan 对多个 source 没有证明 traversal 一致，却固定拿第一个 source 作为 master。它们都能穿过当前 217/217 gate，说明 gate 的覆盖与 verifier 的不变量仍然比实际语义窄。

因此，准确状态是：第五轮建立了一条真实、全 corpus 的 shared-stage coverage gate，也收拢了一部分 analysis authority；但 basic coverage gate、shared program 的 provider sufficiency、family rewrite 的语义保持以及 provider 边界还没有同时闭合。第五轮报告中消失的 gap，一部分已修，一部分只是被移出“shared gate”的观察范围，另有一部分原样留在实现中。

本轮没有继续重构。只删除了两处有明确 def-use 证据、不会改变行为的残余代码；结构性问题仅记录。

## 核查范围与方法

本轮没有运行 GPU、JIT、benchmark，也没有更新 baseline CSV。

核查使用的是当前生产链，而不是另造测试入口：

- `tools/intent-compile/intent-compile.cpp:92-144` 依次执行 canonical KIR verification、KIR-to-GPU construction、全部 shared passes、provider legalization 和 serialization，并用不同 exit code区分阶段；
- `python/intent/compiler/toolchain.py:14-23,26-76` 将这些 exit code 保留成 `physical_program_verification`、`provider_program_verification`、`terminal_translation` 等阶段；
- shared gate 继续使用 `intent.compiler.inventory` 和 `examples/shared_compile_bindings.py` 的完整 `93/217` specialization inventory；
- provider probe 对同一批 217 个 specialization 分别调用生产 `run_compiler`，只到 terminal source，不 materialize provider artifact；
- 另按 baseline-new registry 的 runtime-visible entries 做了一次 entry 级汇总，用来与旧链曾覆盖的结构对照。这个汇总不是 runtime pass rate，也没有拿它替代 217-kernel structural probe。

shared gate 的实际命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:examples:. \
  /home/kingdom/.venvs/intentdsl-mlir20/bin/python \
  -m intent.compiler.inventory \
  --kernel-root examples/kernels \
  --bindings-module shared_compile_bindings \
  --compiler build/tools/intent-compile/intent-compile \
  --jobs 8
```

所有 provider probe 都使用同一个当前设备 capability resolution；没有为某家修改 Kernel IR、constexpr binding 或 shared Program。

## 历史 gap 的当前状态

### 第二轮留下的 canonical 表面

`report/history/canonical-kir-reconstruction.md:151-157` 明确留下三处未闭合表面。这三处没有在第五轮被修掉，只是它们不属于 217/217 shared gate 的分母。

第一处是 tuple/record logical-buffer element。当前 verifier 仍在 `lib/Dialect/Intent/IR/IntentOps.cpp:1661-1666` 要求 `BufferType` 内部是 `RankedTensorType`，KIR-to-GPU 又在 `lib/Conversion/KIRToGPU/KIRToGPU.cpp:3646-3650` 直接 cast 成 ranked tensor。它没有变成另一种表示，也没有 provider realization；原 gap 仍在。

第二处是 scaled-contract scale-axis relation。当前 shared contraction pass 在 `RealizeContractionBlocking.cpp:1977-2014` 从 reduction pair、rank 和 group extent寻找唯一 inner pair；遇到多个可能 pair 时才报 ambiguous。也就是说，作者侧缺少的 typed scale-axis relation没有被补齐，shared pass仍在从当前形状恢复它。原 gap 没有关闭，而是落进了 family-local推导。

第三处是一般调用前置条件。当前 canonical surface仍只有 `assume_in_bounds`；`IntentOps.cpp:1621-1627` 和 `KIRToGPU.cpp:4324-4332` 只保存这一类 typed assertion。sorted/unique、dynamic-extent equality、encoded-format validity等没有正式 surface。它们没有被第五轮验证，也不能由 217/217 推出已经支持。

这三项本来就不应在第五轮擅自发明 API；问题不在“第五轮为什么没修”，而在最终报告把 shared compiler写成无设计卡点时没有说明它主动缩窄了结论范围。

### 第三轮的 shared executable gaps

`report/history/gpu-program-ir-triton-reconstruction.md:119-129` 列出的 gap 只有一部分关闭。

Region fold/scan 已经不再只是 shell。`RealizeRegionFold.cpp:1408-1612` 真实建立 segment loops、source slices、summary/combine carry；`1615-1748` 建立 scan summarize/apply/emit、output assembly 和 final state。这一块从“完全未物化”变成了真实 executable rewrite。

但是它没有横向闭合。fold 在逐个分析所有 source 后，于 `RealizeRegionFold.cpp:1456-1461` 无条件用 `plans.front().ranges.front()` 决定 stop；scan 在 `1637-1662` 同样用第一个 source 的 extent作为循环上界。这里没有检查其余 source 的 start、extent、step 与 master一致。canonical KIR verifier曾证明 logical extents lockstep，不等于 family rewrite 后的 physical source ranges仍相同。这个 gap 已从“没有 region physicalization”变成“只有单-master physicalization”。

Multi-axis reduce 已增加 normalization 和分解：`RealizeReductionBlocking.cpp:1248-1847,2528-2563` 会构造 inner/outer reduction并要求结构前进。因此原来的“所有 multi-axis reduce 未实现”已经关掉。仍未闭合的是 runtime free axes：`1871-1877` 和 `1912-1915` 明确拒绝尚未 physicalize 的 runtime free axis。历史 gap 变窄了，但没有消失。

Multi-contract joint realization 与 execution-group联合 physicalization 原样存在。当前 driver仍逐个遍历 contract并局部改写；scaled path还在 `RealizeContractionBlocking.cpp:2195-2200` 要求“one complete current program segment”。多个相关 contract、多个 execution group无法共同决定 mapping、reuse 与 effects。第五轮报告把 ordinary/scaled/sparse 分开称为合法 family rewrite，却没有证明它们能形成联合决定；这正是历史报告所说的 structural gap。

Buffer realization没有在 shared层闭合成统一物理存储程序，而是明确移到了 provider。这个层次选择本身可以成立，但当前 provider并未接完：Triton在 `lib/Target/Triton/Transforms/Legalize.cpp:662-665` 直接拒绝 `gpu::BufferOp`；TileLang则在 2075 行 `Bufferize.cpp` 中重新决定 storage/copy/sync。历史“buffer realization gap”不是已修，而是拆成了 shared buffer dataflow 和仍未完成的 provider-local storage realization。

Captured/multi-axis reduce、exclusive/captured scan 和一般多轴 gather也没有因 shared gate自动关闭。它们现在主要表现为 provider legality拒绝，性质从 shared shell不足变成 provider form覆盖不足。

### 第四轮的 provider gaps

`report/history/cutile-tilelang-provider-lowering-reconstruction.md:125-134` 的记录仍基本成立。

Sparse contraction 已有 shared `SparseContractOp`，但三家 provider没有形成完整消费。当前 Triton在 `Legalize.cpp:667-670` 明确拒绝；cuTile与TileLang lowering里没有 `SparseContractOp` consumer。TileLang upstream有 `gemm_sp` 并不能替代 Intent 中缺失的 compressed value/metadata physical relation。

TileLang atomic/scatter仍未横向闭合；runtime `scf.while` 仍不在 cuTile/TileLang封闭 surface；record、多输出与 InOut state仍没有完整 provider evidence。这些都不是 target language 已证实无能力，而是 provider implementation coverage gap。

因此第四轮的 gap没有被第五轮 shared report解决。它们被归为“后续 provider 轮次”，这属于范围移动，不是能力关闭。

### 上一份全量报告的未闭合事项

`report/history/full-gpu-regression-and-triton-performance-closure.md:629-664` 的几类结论，本轮核查后的状态如下。

验证事实和 Triton 1.05× 性能证据没有在第五轮产生；这是第五轮主动不覆盖的范围，仍然未闭合，但不能反推成 shared bug。

Shared GPU Program 类别中，raw `operator_kind` 跨层承担语义权威已经修掉，当前使用 generated typed enum accessors；这一项确实关闭。Dependence/access analysis也比旧报告显著前进，`PhysicalProgramAnalysis` 已成为 current-IR query入口。

其余几项只部分关闭：region fold/scan存在单-master问题；runtime axes仍明确拒绝；multi-contract联合 realization仍不存在；sparse/ragged联合结构仍不能横向消费；row/blocking规则仍有 single-segment 和固定 candidate 前置条件；family内仍有重复 dimension binding与局部 candidate authority。它们在第五轮报告里消失，不是都被修掉。

Provider类别仍未关闭。当前纯编译 probe没有 JIT timeout干扰，却已经在 provider legality/serialization阶段大面积失败，说明此前不能只把问题归因于 timeout或下层工具链。

旧能力回归也没有注销。旧链的厚 materializer曾覆盖 cuTile 35/37、TileLang 19/37；当前架构正确地删除了旧链，但相同 entry在新唯一执行链上没有恢复到 terminal source，因此这些仍是能力迁移未完成的证据。

## Shared Program 进入 provider 后发生了什么

### 完整 217-kernel probe

Triton 结果是 5 个成功到达 terminal source，212 个停在 provider program verification。成功的 5 个 specialization 是：

- `sparse_mla_grad_kv_cast`；
- `mla_head_projection`；
- `rotary_embedding_bf16`；
- `ragged_grouped_gemm`；
- `ragged_grouped_gemm_bf16`。

212 个失败中，186 个是同一句：

```text
physical kernel program mapping/effect coverage is incomplete
```

其余 26 个主要是动态 `make_range` 无法成为 Triton compile-time fragment，以及一个 provider candidate/form问题。

cuTile 结果是 98 个到达 terminal source，89 个停在 provider verification，30 个停在 terminal translation。首个诊断最集中的形态是：45 个 store coordinate不能被转成完整 tile domain，12 个 load同类失败，29 个 NaN-propagating min/max没有 cuTile等价 spelling，7 个 `scf.if` 超出 serializer封闭 surface，7 个 logical buffer load没有 provider form。这里既有真实语义子集，也有 provider legalization/serialization没有接完；不能统称 shared失败。

TileLang 结果是 75 个到达 terminal source，134 个停在 provider verification，8 个停在 terminal translation。首个诊断中，49 个是 native reduce只接受一个 builtin component，15 个是 native GEMM集合找不到同时合法的线程数，8 个是 NaN-propagating min/max没有等价 spelling，7 个是 external logical buffer view未实现，5 个仍留下 unbufferized shared fragment；其余分布在 gather/scan/scaled/sparse/random/control和axis歧义。TileLang有明显 capability/form子集，但也有 provider Program未完整物化。

这些计数不是性能结果。它们只回答同一份 shared Program能否穿过现有 production provider compiler直到源码。

### Registry 级结构对照

按当前 baseline-new registry 的 runtime-visible entries 汇总同一次纯编译结果：

- Triton 54 个 entry中，2 个所有 component kernels都到达 terminal source，52 个停在 provider verification；
- cuTile 37 个 entry中，17 个到达 terminal source，15 个停在 provider verification，2 个停在 terminal translation，2 个 multi-kernel entry内部阶段不一致，1 个没有 DSL；
- TileLang 37 个 entry中，6 个到达 terminal source，28 个停在 provider verification，3 个没有 DSL。

这和旧表的 runtime pass数不是同一测量：当前 probe没有 provider JIT、launch、adapter和numerical comparison，entry inventory也有调整。但方向是明确的：旧链曾经真正消费过的一批 execution/access/storage事实，在新路径里没有全部迁移成可验证 provider Program。有些被 shared Program表达后又被 provider transformation破坏，有些仍由厚 provider代码从 shape/use graph重建，有些直接缺失。

### Triton 的主失败不是 shared输入不足，而是 provider改坏当前程序

`lib/Target/Triton/Transforms/ProgramGrid.cpp:11-129` 只处理恰有一个、rank不超过3的 `gpu::DelinearizeOp`。它读取 `coordinateRoles`决定 Triton grid顺序，创建多个 `gpu::ProgramIdOp`，替换 coordinate uses，然后在 `126-128` 删除原 `DelinearizeOp` 和一维 `ProgramIdOp`。

问题是，execution-group的 typed group/offset/length attributes仍附着在被删掉的 `DelinearizeOp` 上。`lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:293-317` 正是遍历这些 operation attributes建立 `executionGroups`；`372-376` 要求 execution groups非空、program axes齐全且 effect origins完整。

Triton pipeline在 `lib/Target/Triton/Transforms/Legalize.cpp:881-890` 先验证 shared program，运行 `legalizeProgramGrid`，随后立即再次运行同一个 full verifier。因此普通两维 pointwise shared program在进入 ProgramGrid前合法，mapping被删除后随即报 `physical kernel program mapping/effect coverage is incomplete`。

这说明两件事：

第一，217/217产生的 shared mapping至少在这些 case里是完整的；不能把186个失败写成“shared没有program mapping”。

第二，provider-local grid legalization没有定义一种在改写后仍承载 execution group的 provider Program形态。它违反“complete program到complete program”的不变量，是第五轮后仍存在的架构断点。第五轮唯一的 grouped GEMM repro没有触发这一失败，因此不能覆盖 ProgramGrid 的 `<=3 coordinates` 分支。

本轮没有修这个问题。修法需要决定 execution group在provider grid IR中的正式载体，属于结构设计，不是audit轮可当场删除的局部残余。

### cuTile 与 TileLang 显示的 shared/provider边界

cuTile在 `lib/Target/CuTile/Transforms/Legalize.cpp:173-220` 从 `MakeRangeOp.start`重新识别 `tile_index * extent`、loop IV/step等形状，再构造 tile index。它不是单纯消费已经选定的 access form；57个load/store coordinate失败说明当前 shared access relation没有直接落成cuTile可消费的tile coordinate form，而provider-local recognizer只覆盖少量表达式。

这不自动说明 shared IR必须增加cuTile字段。更准确的缺口是：当前provider legalization没有把shared typed coordinate relation系统地lower成cuTile tile index，而是在表达式形状上重识别；换一种等价coordinate SSA写法就会失败。

TileLang更明显。`lib/Target/TileLang/Transforms/Bufferize.cpp:423-501` 收集全部load/contract/reduce/scan/store后重新做bufferization；`510-523` 沿use graph判断一个value是否“feedsDirectContractOperand”，据此选择Shared或Fragment storage。后续同一文件还决定access offset、padding、copy、sync和native axes。`FormPipeline.cpp:65-137` 再扫描loop body中是否同时出现Gemm/CopyIn，现场包成PipelineOp。

这些都是合法的provider-local职责种类，但当前实现依赖邻接、use graph和op presence重新做决定，而不是消费一份完整的TileLang provider Program。`Verify.cpp:41-98` 只检查允许的op/type、没有残留FragmentType和loop result；它没有检查TileLang真实要求的parallel nesting、local buffer index ownership或pipeline dependence。因此“通过TileLang provider verifier”也不等于provider program可由下层合法接受。

## Shared family 中仍存在的第二份推导和语义缺口

### Scaled contraction 丢失原 validity

`RealizeContractionBlocking.cpp:2160-2172` 明确允许四个load和result store带三类validity：无validity、scalar validity、或可识别tail predicate。随后rewrite在 `2429-2456` 只重新构造row/block/column tail mask，在 `2497-2510` 创建新loads；原来的scalar或residual validity没有replay，也没有与新tail mask相与。结果store同样在 `2527-2555` 只使用新row/column mask。

因此，这段代码的acceptance predicate比它能保持的语义更宽。只要原validity不是恒真、却被`scalarSource`接受，rewrite就可能读取或写入作者谓词已经排除的位置。217/217没有覆盖能把这个差异暴露成数值错误的specialization；shared verifier只看到重写后的access自洽，无法知道原谓词被丢了。

这是明确的shared correctness缺口。本轮没有局部补一个`and`，因为scalar validity怎样broadcast到data/scale/result各自fragment、哪些谓词可以合法replay，应由统一predicate/replay authority定义，不能在scaled family里再造一套。

### Region fold/scan 的多source master假设

`RealizeRegionFold.cpp:1412-1419` 与 `1619-1626` 确实逐source建立`SourcePlan`，但fold在`1456`、scan在`1637`直接选第一个range为master。构造loop时分别用`master.getExtent()`；没有证明其余plans的physical start/extent/step与master相同。

scan还有第二个单source假设：`1701-1708` 对所有output consumer cloning都传入`plans.front().sourceIdentity`。如果不同output对应不同source component，地址切片仍绑定第一个source。

helper extent binding也有一条宽松路径。`collectExtentBindings`在`442-469` 找不到typed source identity时，只要expected/actual rank相同就按axis position配对；随后`590-603`直接改写cloned result和block argument类型。这不是机械保存provenance，而是“同rank即同axis”的恢复规则，换一次transpose或两个同rank source就可能绑定错误。

另外，`findParameter`在`719-725` 只按name找physical parameter，没有同时核对role、canonical node、dimension或source；`summaryMembershipPredicate`在`956-984` 有多个满足条件的bool时保留最后一个；`predicatePartition`在`1327-1405` 有多个合法compare时返回第一个。这些都是输入语料只有一种候选时看不出来的single-input branch。

### Contraction 的 stale `indirectRow`

ordinary contraction在`RealizeContractionBlocking.cpp:1412-1450` 先根据最初是否找到row range设置`indirectRow`。若随后producer-range fallback成功找到exact range，`rowRange`被补上，但`indirectRow`没有重算。后续`1483-1485`、`1723-1736`仍按indirect路径收集assertion并replay coordinate graph，`1623-1627`也把role记成IndirectTraversal。

这不会必然算错，但它使同一份已恢复exact range仍走更厚的replay和不同role，是“先失败一次就永久改变physical decision”的第二条路径。

### Family-local parameter与dimension authority仍重复

Pointwise在`RealizePointwiseBlocking.cpp:72-82`、reduction在`RealizeReductionBlocking.cpp:832-847` 各自遍历kernel arguments寻找同一个dimension binding。Pointwise又在`903-943`本地创建`POINTWISE_CHUNK`候选，reduction在`774-830`和runtime chunk路径本地维护另一套coverage/chunk候选。

Shared pass创建physical parameter本身是正确职责，不能因为它出现在family文件就判错。问题在于这些candidate domains与binding lookup没有从统一legality/resource/access facts导出，也没有证明换一类输入会产生不同合法域。它们仍是family-local decision authority，而第五轮报告把physical parameter binding写成已经唯一收拢。

Pointwise对rank-lifted value还在若干路径直接构造unit extent和owner=1；这可能是合法的conservative baseline，但当前没有typed relation证明这些新axis确实是unit/broadcast axis。它同样需要由换输入后的IR事实证明，而不是由217个当前case都没报错来背书。

## 本轮删除的明确残余

本轮只改了两处能够由当前def-use直接证明无效的代码。

`RealizeReductionBlocking.cpp` 的 `collectRangesAndLoads` 同时收集ranges和loads，但调用者排序、去重loads后从不读取它们。现在改成`collectRanges`并删除完整的unused load traversal。它不改变任何rewrite输入或输出。

`RealizePointwiseBlocking.cpp` 在替换program coordinate后，把新的scalar `index` value传给`retargetSourceExtent`。而统一实现`Utilities.cpp:928-932` 对非`FragmentType` root立即返回，这条调用在所有输入上都是no-op。已经删除该分支中的extent/source构造和调用；其它真正以fragment/range为root的retarget路径保留。

删除后重新构建成功，`git diff --check`通过，完整shared inventory仍为217/217。没有删除provider terminal spelling，没有改verifier，也没有用fallback让任何case通过。

## 与 Triton/TileLang ref 的双向对照

### ref 有而当前 Intent 没有：不是每项都该进入 shared

Triton的`ref/triton/lib/Analysis/BufferIndexAnalysis.cpp:17-46,122-204` 把buffer slot index表示成`base + constant offset`及可选modulus，并对loop-carried ring counter做有界归纳，从而证明不同pipeline slot不alias。Intent的`PhysicalAccessFootprint`目前只保存resource、coordinates、source axes、ranges、validity和fill；没有modular index relation。

实际后果不是“Intent必须复制Triton analysis”，而是：一旦provider选择ring-buffered async pipeline，当前shared footprint不能替它证明slot independence，provider必须有等价local analysis。TileLang当前`FormPipeline`只按Gemm/CopyIn和use依赖包loop，没有该证明，因此多个producer或环形stage不能仅凭当前verifier判合法。

Triton的`ref/triton/lib/Analysis/Allocation.cpp:175-201` 建立liveness intervals、interference graph并计算storage offsets；`BufferRegion.cpp:70-77,93-175` 再按已选layout、padding和CTA计算真实byte footprint。Intent的`PhysicalProgram.cpp:1349-1388` 只证明first-write/full initialization，没有live intervals、interference或offset。

这不是shared层遗漏layout：`doc/compiler`已经把layout和physical storage overlap留给provider compiler。真正差异是当前TileLang provider没有一套与Triton allocation同等级的local analysis，却已经在Bufferize里做storage选择。因而shared verifier不能替TileLang证明storage合法，TileLang verifier也没有补上这层。

Triton的`SoftwarePipeliner.cpp:21-27,93-143` 先形成modulo schedule，再由pipeline expander重写loop并生成prologue/epilogue；TileLang的`ref/tilelang/tilelang/cuda/pipeline.py:118-142` 也明确分开PipelinePlanning、InjectSoftwarePipeline、LayoutInference和lowering。当前Intent TileLang `FormPipeline.cpp:65-137` 只有“loop里有Gemm且有transfer”这一candidate predicate，然后直接包`PipelineOp`，没有schedule/dependence/prologue/epilogue carrier。换成两个producer、cross-iteration value或需要显式wait的loop时，当前形态没有地方表达差异。

TileLang ref还在`tilelang/analysis/nested_loop_checker.py:34-62,80-109` 检查parallel/pipelined nesting，在`parallel_local_index_checker.py:24-57` 禁止local buffer index依赖parallel loop variable。Intent的TileLang verifier `Verify.cpp:25-36,55-98`只做surface/type清单。这是具体的provider legality缺口：一份Program可以通过Intent verifier，却违反TileLang自己的ownership/nesting规则。

### Intent 有而 ref 没有：有一部分是合理的跨provider差异

Intent的typed `PhysicalSourceAxis`、`AxisMapAttr`、structured `RegionFoldOp/RegionScanOp`和cross-provider access footprint不是Triton TTGIR的复制品。Triton作者已经在source里写下program grid、block shape和pointer arithmetic；Intent作者没有，所以KIR-to-GPU与shared passes必须显式保存source relation、compiler-selected segmentation和effect origin。这些结构在ref里没有同名物，不代表它们是变通。

成立的条件是provider只消费这些typed facts或在真实target-only结构上扩展。当前Triton ProgramGrid删除execution-group carrier、cuTile从arithmetic shape重认tile index、TileLang从use graph重选storage，说明这个条件尚未满足。问题不是“Intent比ref多了一层”，而是新增的shared authority没有一路保存到terminal boundary。

Intent的generic reduce/scan/region fold semantics也比TileLang native builtin集合更宽。这是合理差异；正确结果是provider明确形成合法子集或拒绝，而不是在shared层把generic combine猜成某个builtin。当前TileLang的49个single-component native reduce失败属于这类真实provider surface gap，不能为了提高通过数去收窄shared semantics。

## 最终判断

第五轮关上的，是一件重要但更窄的事：当前93个文件、217个specialization都能从canonical KIR构造出一份通过当前full shared verifier的GPU Program，而且若干过去分散的source/range/replay/access查询已收敛到current-IR analysis。

它没有关上的，是“这份Program已经横向完整，后续provider只需合法化和序列化”这一目标。证据不是没找到问题，而是：

- 同一生产链进入provider后，Triton 5/217、cuTile 98/217、TileLang 75/217才到达terminal source；
- Triton provider会删除shared verifier唯一读取的execution-group carrier；
- scaled contraction能在rewrite时丢失原validity；
- region fold/scan把多source程序缩成第一个source的master traversal；
- multi-contract、runtime free-axis、sparse、mutable buffer和多种provider form的历史gap仍在；
- cuTile和TileLang仍通过表达式/use-graph recognizer重建tile/storage/pipeline，而不是完整消费provider Program；
- 与ref相比，TileLang缺少与其storage/pipeline决定相配套的legality和dependence analyses。

所以答案是：shared重构没有做完；一部分能力真实完成，一部分gap被缩窄，一部分只是从第五轮报告的shared定义中移出，还有少量新的correctness/authority缺口被217/217掩盖。继续实现之前，应先把这些问题按shared语义保持、provider-program carrier和target-local legality三层重新归位；本轮没有替后续架构选择答案。
