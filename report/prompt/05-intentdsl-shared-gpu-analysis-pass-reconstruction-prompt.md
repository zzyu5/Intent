# 第五轮：重建 shared GPU compiler 的 analysis 与 pass authority

这一轮不做六表全量，不追 Triton source 的 `1.05×`，不处理 cuTile/TileLang provider timeout。

这一轮只解决一件事：让 canonical KIR 之后的 shared GPU compiler 横向成立。完成后，每个真实 kernel 都必须先得到一份完整、合法、provider-neutral、可独立验证的 executable GPU Program；shared physical facts 与 decisions各有唯一权威来源；Triton 只作为一条 end-to-end 投影验证，不能替 shared coverage背书。

完整阅读：

```text
doc/index.md
doc/compiler/README.md
doc/compiler/kir-to-gpu.md
doc/compiler/gpu-program-ir.md
doc/compiler/passes-and-analyses.md
doc/compiler/physical-parameters.md
doc/compiler/target-lowering.md
report/shared-gpu-compiler-investigation-and-round-recut.md
report/full-gpu-regression-and-triton-performance-closure.md
```

遇到结构性选择，直接对照：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

参考的是它们怎样用 current IR、typed operations、dataflow/access analyses、pass invariants与 verifier承载同类事实，不复制 TTGIR、CUDA pipeline、TileLang tile surface或它们的 pass 名称。

---

## 一、为什么必须重新做第五轮

原第五轮本应是最终全量与 Triton 性能闭合，却在 `669410b..bab7a40` 之间继续增加约一万行 shared/provider后端：

```text
RealizePointwiseBlocking   +2461
RealizeReductionBlocking  +2040
RealizeRegionFold         +1824
RealizeContractionBlocking +1561
KIRToGPU                  +2196
```

这不是 agent 执行不够努力，而是原轮次没有一道 shared compiler横向闸门。第三、四轮只需一条 kernel end-to-end通过即可结束；第五轮第一次全量才证明 region fold/scan、multi-axis reduce、multi-contract、buffer/access realization都没有横向完成。

当前结果也证明 Triton不能代替这道闸门：

```text
Triton registry      54
cuTile registry      37，其中 30 个 kernel 名不在 Triton registry
TileLang registry    37，其中 30 个 kernel 名不在 Triton registry
```

当前 cuTile 的 6 个 `physical_program_verification_failed`，以及 TileLang 的 2 个 `physical_program_failed` 和 12 个至少一机 `physical_program_verification_failed`，全部发生在这两组 Triton未覆盖 kernel 上。`tilelang-h100.csv` 是修复前快照，这个 union 只用来判定 shared shape 覆盖，不当作同一 HEAD 的最终通过率。Triton 51/54 只说明一小部分 shared shapes可用。

后面轮次已经重新切定：

```text
第五轮  shared GPU compiler；Triton 受影响路径不退化
第六轮  cuTile provider closure 与 timeout forensic
第七轮  TileLang provider/storage/copy/pipeline closure
第八轮  两机三家全量、公平 candidate 对照与 Triton 1.05×
```

原第五轮中公平比较资格、candidate set、measurement scope、两机并行、六表与 1.05× 的要求全部保留到第八轮，本轮不要重复做。完整移交索引在 `report/shared-gpu-compiler-investigation-and-round-recut.md` §13.4，包括 source 资格、算法/kernel/pipeline 对齐、`region_fold/region_scan` 与 multi-kernel wrapper、candidate 过滤/winner、fixed-singleton、JIT/autotune/CUDA Graph/reset/timed closure 和 CSV schema；这些都是第八轮必须继承的要求。

---

## 二、先修正“完整 physical program”这个承重矛盾

当前接口与实现互相矛盾：

- `include/Intent/Conversion/KIRToGPU/KIRToGPU.h:16-19` 声称 construction产生 complete conservative executable GPU program；
- `doc/compiler/kir-to-gpu.md:52-65` 也要求第一份 program完整、合法、可执行；
- 但 `lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:186-189` 明确拒绝仍存在的 `RegionFoldOp/RegionScanOp`，称它们在 segment traversal、source slicing、carry与 output flow物化前不完整；
- `runSharedGPUPasses` 中间只调用通用 `mlir::verify`，完整 `verifyGPUProgram` 只在末尾运行（`Passes.cpp:9-57`）。

本轮必须闭合这个矛盾。下列是规格已经要求的结果，不是预定的实现方案；具体 IR 分阶、analysis 组织和 verifier 落点必须由当前代码与 ref 对照决定。结束状态必须满足：

- construction结束时已经是一份完整 executable baseline，而不是等待某个 family pass解释的 structured record；
- 每个 shared transformation前后，当前 GPU Program 都满足该阶段声明的完整性不变量；
- verifier检查 effect coverage、program mapping、ownership、control/carry、access/validity、buffer与 structured flow，不以通用 MLIR well-formedness冒充 shared legality；
- 后续 pass可以把保守 program改得更好，但不能负责把“不完整”补成“第一次可执行”；
- 不保留另一条 legacy construction或 provider fallback。

不要通过删除 region fold/scan、把它们串行展开成数量级慢路径或给 verifier 加例外来过关。它们是正式 canonical semantics，initial program必须真实承载。

---

## 三、建立真正的 analysis authority，但不要照抄 Triton 的目录

当前 shared facts 不是空白：`FragmentType`、`RangeType`、`AxisMapAttr`、access coordinates、validity、owner与 buffer fields都是真实 current-IR载体。保留这些正确结构。

缺口是 dependence、access footprint、coordinate projection、lifetime/reuse、tail predicate、structured operand flow等事实被 family passes各自沿 producer graph重算：

- `RealizeAccessComposition` 自己组合 access provenance；
- `RealizeContractionBlocking` 自己收集 ranges、coordinates 与 tail predicates；
- `RealizeRegionFold` 自己建立 source plan；
- `RealizePointwiseBlocking` 有只服务自身的 reduction-dependency递归；
- `Utilities.cpp` 是 helper集合，不是可查询、可失效、可复用的 analysis result。

对照 ref 后，为 Intent 真正需要的事实建立唯一 authority。当前规格给出了不可变 canonical KIR 与 current executable GPU IR 的责任边界，但不预定必须创建哪些 C++ 类、目录或 pass。具体组织可以被 ref 与实现证据推翻，只要以下性质成立：

- canonical analyses只读 immutable KIR；
- physical analyses只读 current GPU IR；
- unknown必须是显式保守结果，不从 shape、origin、role名称或附近 op猜；
- transformation修改相关 IR 后，analysis要么明确保留，要么失效并从 current IR重算；
- analysis cache可以是 side information，但影响执行的 mapping、range、validity、ownership、buffer和loop必须写回 current executable IR；
- 多个 pass需要的同一事实只能由一个 analysis/typed query提供。

不要机械创建 `AxisInfo`、`BufferRegion`、`Alias` 六个同名模块。Triton使用 lattice，TileLang更多依赖 TIR `BufferRegion` 与 reject-only semantic checks；Intent 的结构由自己的 KIR/GPU IR需要决定。判据是事实是否唯一、可查询、可失效重算，不是有没有 `Analysis/` 文件夹。

---

## 四、收拢 decision authority，不做一个万能 op-family pass

按 op family 存在专门 lowering不是错误。Reduce、scan、region fold/scan 与 contract具有不同 accumulator/carry semantics，应该保留专门 rewrite patterns。

当前真正的问题是同一个 cross-family physical decision被独立实现多次。调查得到的起始数量是：

| fact/decision | 当前独立推导/实现簇 |
|---|---:|
| ownership / program mapping | 约 5 |
| zero/fill materialization | 4 |
| tail predicate / validity | 6 |
| fragment/value replay | 至少 5 |
| access composition/source-axis recovery | 4 层 |
| ordinary/scaled contraction mapping | 2 个高度同构实现 |

例如 `zeroFill` 已在 `Utilities.cpp`、Pointwise、Reduction、RegionFold各定义一份；tail predicate 的 range compare、projection、broadcast、与原 validity 合并，在多个 family各做一遍。

本轮完成态不是“所有代码只剩一个函数”，而是：

> 对每个 shared physical fact/decision，独立推导 authority只有一个。Family-specific patterns可以有多个，但只能消费统一 analysis/result，不得再从 source ID、shape、origin、op邻接独立恢复同一事实。

重切 pass边界时，以 IR invariant、analysis dependency和 mutation关系为依据。不要把现有四个大文件机械拆成更多小文件，也不要把所有 structured semantics塞进一个 universal blocking pass。

---

## 五、核对 construction 中没有事实支撑的 policy

当前 construction 已经在 analysis形成前写入窄 physical policy：

- workset主要从显式 `intent.parallel` 收集；没有 parallel 就退化为 singleton（`KIRToGPU.cpp:4306-4323`）；
- `pointwiseCoordinates` 通过 region 内是否出现 Reduce/Scan/Contract/Fold/For/If 等 op family分类（`:4262-4287`）；
- 所有 dynamic tensor dimension都预先得到 `FRAGMENT_D*`，统一使用 `OwnershipN` role和 `{64,128,256}` candidates（`:4359-4392`）。

这与规格定义的 logical workset来源不一致。Workset应由 parallel、unique effects、structured free axes、def-use、alias与 dependence共同形成；initial program应保守但完整，不应在缺少 access/reuse/structured facts时给所有 dynamic dimensions套同一 blocking policy。

本轮的行为判据是：construction 只能作出由它已有完整事实支撑的初始决定，无事实支撑的第二份 policy authority 不得留存。上述三处最终由哪个 analysis、IR carrier 或 transformation 承担，由 ref 对照与 current-program invariant 决定；不能只换成另一张固定阈值表，也不能把 op-family 名字换成数字 role 就宣称完成。

同时核对 target capability输入。当前 `GPUCapabilities` 只有 compute units、shared memory、registers、matrix units与 dynamic vector width；若某项 shared legality需要 dtype、grid、atomic或 structured primitive能力，必须从正式 typed capability获得，或明确留到 provider/hardware legality。不能在 shared pass缺信息时猜一个候选，再让 JIT判死。

---

## 六、把跨层 observable semantics 变成 typed authority

`operator_kind` 是已确认实例，不是唯一目标。

当前事实：

- `doc/dsl/types-numerics-and-effects.md:87` 已明确 propagating `maximum/minimum` 与 `maximum_num/minimum_num` 是不同语义；不要改写这条规格；
- Python enum使用 `7/8/9/10` 区分它们；
- canonical KIR与shared GPU IR都把它们降成裸 `I64Attr`；
- cuTile `nativeCombineKind` 把 `7/9` 合成 max、`8/10` 合成 min；
- 当前 cuTile float min/max lowering明确使用 `propagate_nan=False`，所以 canonical propagating max/min被静默改错；
- `RealizeContractionBlocking` 把 logical-and `11` 与 bitwise-and `13` 都当 tail conjunction，却没有把 `i1` 等价条件写进 typed legality。

修复目标：observable semantic categories在 canonical KIR、shared GPU IR、analysis query与 provider boundary中具有唯一 typed含义。可以使用 typed op classes、enum attrs或其它 MLIR-native表示；不要预设必须“一种操作一个 op”，但不允许多个 pass共享一张裸数字表。

不要只把这四个数字改成 enum 就结束。以它们的真实 consumer graph 为起点，找出其它会跨层决定 observable semantics 或 physical legality、且当前可能被多个 consumer 不同解释的载体。扫描边界由实际 use-def 与 ref 对照生成，不预先规定一张“必须改成某种类”的清单。

axis index、record field index、component count等结构性整数不需要为了形式全部改成 op class。Provider API enum在 terminal serialization中也可以是数字。判据是该值是否跨层决定 observable semantics或 legality，以及消费者是否可能各自解释成不同含义。

对 cuTile/TileLang 只做 typed interface迁移和明确的语义保持/拒绝，不在本轮开发新的 provider form。Provider-specific capability与性能闭合留到第六、七轮。

---

## 七、pipeline 可以有顺序，但顺序不能代替 analysis contract

不要把目标写成“passes 可以任意重排”。Triton与TileLang都有严格顺序，某些 pass必须在 source access region、tile op或layout仍存在时运行。

当前需要修的是：

- 每个 pass声明读取什么 current facts/analyses、改写什么 IR、不变量是什么、使什么 analysis失效；
- 顺序依赖来自明确的 produced/consumed facts，不来自“趁某个 op 还没被 erase赶紧扫一遍”；
- `decomposeMultiAxisReductions` 在 top-level 与 reduction pass 内部各调一次，是当前 transformation 职责不明的具体证据；完成态要么只有一个 normalization authority，要么有 ref 与 IR invariant 支持的明确 fixed-point contract，不预定通过搬哪次调用来修；
- access composition 两次运行当前有真实原因，因为中间 structured realization 会产生新 access。完成态必须让这种重复由明确的 produced/consumed fact、fixed point 或 canonicalization contract 解释，而不是无解释重复；
- shared verifier在有意义的阶段边界验证完整 executable invariants。

是否采用 MLIR `PassManager`、analysis manager或怎样拆文件，由实现证据决定；不要把换框架本身写成成果。成果必须表现为 current IR、analysis authority、失效规则和横向 coverage 的真实变化。

---

## 八、两道强制闸门

覆盖与重构有明确的先后关系：重构过程中遇到 coverage 失败，先将它归类为“缺少哪个 analysis/decision authority”，不在失败点就地增加规则补通。Coverage 是重构完成后的结果，不是与重构并行的另一项任务。

原因是：在当前四个 family 文件各自推导 tail/replay/ownership 的结构下，关闭 coverage 最省事的做法就是继续往对应文件增加更窄的规则；原第五轮新增的约一万行后端正是这种推进方式的结果。Decision-authority 闸门最终虽然会拦住它，但到那时又已经多了几千行需要拆除的窄路径。

两道闸门必须同时通过。任何一项没通过，本轮不能以“gap 已列出”结束。

### Shared executable coverage

执行时完整 `examples/kernels/` corpus中的每个 `@intent.kernel` 必须经过：

```text
frontend
→ canonical KIR
→ KIR-to-GPU construction
→ 全部 shared GPU passes
→ full shared GPU verifier
```

当前实时枚举 `examples/kernels/**/*.py` 得到 93 个文件、217 个 kernels；`WRITE-05.md` 中的 216 是写作时快照，本次调查没有修改 corpus。本轮必须在执行时重新枚举，若 corpus 有真实变化，报告起止数量与原因。不得：

- 使用 allowlist；
- 只跑 registry中的 54/37/37；
- 把 frontend parse成功或通用 `mlir::verify` 当作 shared gate成功；
- 依赖 Triton/cuTile/TileLang serializer才能判断 shared program完整；
- 为闸门保留 test-only lowering path。

这是本轮用户 prompt 对仓库 `AGENTS.md` 默认“唯一手动数值 repro”验证纪律的一项明确、限定例外：额外允许一条 compile-only compiler inventory audit，仅用于完整 corpus 的 shared-stage 闸门。它不运行 GPU、不做 benchmark、不新建测试基础设施；除此例外外，仍只允许第九节那一条 Triton 数值 repro。若当前 compiler CLI 不能在 full shared verifier后独立停止，使这一正式 compiler boundary可直接调用属于本轮工作；不要新建 pytest、fixture或 test目录。闸门使用一条生产编译路径的批处理命令遍历完整 corpus，结果写入本轮报告，临时文件随后删除。

### Decision authority

报告必须提供 before/after 表：

```text
fact/decision
authoritative KIR/GPU IR carrier
authoritative analysis/query
唯一 mutating transformation entry
family/provider consumers
invalidation/recompute rule
独立重推导次数
```

同一事实的独立重推导次数必须为 1。这里数的是 semantic inference authority，不是 helper调用数或 rewrite pattern数。

217/217 通过但继续靠四个 family文件各自推 tail/replay/ownership，不算完成；代码组织漂亮但任何 kernel过不了 full shared verifier，也不算完成。

---

## 九、Triton 只做受影响路径的 non-regression

shared 重构必须通过当前最成熟 provider的真实投影，但本轮不继续追 source ratio。

当前 Triton CSV事实：

```text
5090  51/54 pass；2 个 source_device_resource_gap，1 个 source_compatibility_gap
H100  52/54 pass；1 个 source_compatibility_gap，1 个 worker_timeout
```

H100 的 `flash_attention_backward` 只有宽泛 `worker_timeout`，当前 CSV 不能证明超时发生在 source 还是 generated；不要沿用“非通过全部在 source 侧”的未验证结论。本轮若 shared 改动触及该 entry，只定位 generated 是否退化，不把 source 侧完整资格审核提前从第八轮搬回来。

本轮不跑 54-entry Triton runtime/performance 全量，那是第八轮的任务。从实际 shared 改动中选一个能穿过被修改 authority 的真实 Triton kernel，使用一条现有手动 repro 命令完成 source generation、provider compile/JIT、launch 与一次数值对照。若改动触及已知的 grouped GEMM/MoE 共享路径，手动 repro 必须选该共享路径，不允许用不相干的简单 kernel 代替。

这条 repro 只判定新 shared 结构是否仍能投影并数值正确，不重新判定 source/generated 算法资格，不追 `ratio >=1.05`，不扩大 candidate set 优化数字。Triton 成功不替代 217-kernel shared gate。

---

## 十、本轮明确不做

- 不运行或更新六张最终 `baselinev2` 表；
- 不关闭 Triton 21 个 `ratio >=1.05` 项；
- 不重新审核 source/generated candidate set、winner与measurement scope；
- 不处理 cuTile 14+14 timeout；
- 不处理 cuTile provider forms、JIT或性能；
- 不处理 TileLang storage/copy/sync/pipeline/provider性能；
- 不恢复旧 KernelModel/Plan/materializer路径；
- 不新增 public DSL construct或改写作者算法；
- 不以 target-specific需求在 shared层加机制；
- 不新增 test目录、pytest、fixture、兼容开关或 fallback；
- 不写版本号、CHANGELOG、迁移指南或 deprecation状态。

如果 full shared gate暴露 canonical KIR本身缺少已定义 DSL semantics，修 canonical表示；如果需要新增作者可见语义才能继续，停止并报告设计歧义，不擅自扩语言。

---

## 十一、主要实现完成后的 ref 对照与收尾

主要结构改完后，暂停继续补 failing entry。不使用一张预写的“成熟编译器检查项”让实现自证通过；而是从本轮实际新增、修改或删除的语义 authority 和 physical decision 出发，去 `ref/triton` 或 `ref/tilelang` 追同一问题的真实实现。对照应由 ref 里的真实结构生成问题，而不是为预设结论找类比。

报告对每处对照保留双方 file:line，说清具体载体、决定阶段、unknown/legality 表达和 mutation 后事实处理的差别，然后用另一个真实 kernel、另一种结构或移除一项前置假设来观察这个差别的实际后果。找不到对应实现、具体差异和外部后果，等于没有完成自查；“与成熟实践一致”“泛化守住了”“emitter 变薄了”都不是证据。

最后沿已替代的 authority 和被改写的 consumer 回溯唯一 executable path，删掉旧推导、无人调用的 helper、不完整中间状态、临时 attr/side record、compatibility/fallback 和本轮临时产物。要删什么由实际 def-use 与 ref 对照决定，不用预写清单替代调查。自查不单独新建文档，证据进入本轮最终报告。

---

## 十二、`doc/` 与 `AGENTS.md` 权限

可以修改 `doc/`，但仅限 ref/实现调查证明规格缺少稳定语义或 compiler invariant的地方：

- 可以补 observable operator、atomic、capability或analysis boundary的正式定义；
- 不得把当前实现限制写成规格，例如“只支持 unit-step”“只支持一个 full-program segment”；
- 不得写进度、失败、性能数字、迁移状态或兼容概念；
- 现有规格已经明确的事实，修实现，不重写 doc迁就代码。

`AGENTS.md` 只增加一条简短、长期有效的纪律：自查时对照 `ref/triton`、`ref/tilelang` 的同类实现，给出双方 file:line、差异与实际后果；找不出具体差异等于没完成。不要加入本轮假设或长检查清单。

---

## 十三、验证与报告

只使用现有 compiler/runner和临时命令，不建立测试基础设施。

产出：

```text
report/shared-gpu-analysis-pass-reconstruction.md
```

报告必须写清楚：

- construction与 intermediate completeness矛盾如何闭合；
- 建立了哪些 canonical/physical analyses，它们的输入、unknown、consumer与 invalidation；
- 每类 physical decision 的 before/after独立推导次数；
- 哪些 family-specific lowering保留及为什么不是重复 authority；
- 哪些裸语义载体被 typed 化，NaN/atomic/format等语义如何保持；
- 完整 corpus的文件数、kernel数与 shared gate结果；
- 失败过的非 Triton重合结构如何被 shared层闭合；
- 唯一 Triton end-to-end numerical repro 的命令、选择理由与结果；
- ref 对照的双方 file:line、具体差异与实际后果；
- 删除了哪些冗余推导、旧 helper与 incomplete path；
- 是否发现规格歧义或仍无法闭合的设计问题。

最终只保留一条 executable compiler path。将实现、必要规格修改、`AGENTS.md` 一行纪律和报告整理成语义连贯的提交，并确认工作区没有临时产物。
