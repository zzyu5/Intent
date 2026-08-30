# 第 5c 轮：闭合 shared GPU 正确性并恢复 Triton 完整接纳

这一轮是第五轮的最后一次正确性收口。完成以后，第六、七轮只实现 cuTile/TileLang 的
provider-local forms，第八轮只处理公平 candidate、两机全量与性能；后续轮次不应再发现 shared
representation、semantic preservation、verifier completeness 或 build-mode correctness 问题。

这句话不能靠报告声明成立。它必须由三类外部证据共同支撑：三家 registry 覆盖的 shared program
都合法完整、54 个 Triton entry 真实生成并运行数值正确、shared GPU 规格中的每条完整性不变量都
有明确且经 ref 对照的执行归属。

本轮不追 `1.05×`，不选择 autotune winner，不更新六张 baseline CSV。

---

## 一、开始前完整阅读

从 `doc/index.md` 进入并完整阅读当前相关规格，至少包括：

```text
doc/compiler/README.md
doc/compiler/kir-to-gpu.md
doc/compiler/gpu-program-ir.md
doc/compiler/passes-and-analyses.md
doc/compiler/physical-parameters.md
doc/compiler/target-lowering.md
```

同时完整阅读：

```text
AGENTS.md
report/current-gpu-compiler-state-and-next-round.md
report/shared-gpu-compiler-investigation-and-round-recut.md
report/triton-foundation-completion.md
report/shared-gpu-reconstruction-completion-audit.md
```

`report/current-gpu-compiler-state-and-next-round.md` 是当前 HEAD 上关于 build-mode divergence、119 个
terminal-source gaps、31 个历史退化和 verifier 缺口的事实基线；
`report/shared-gpu-compiler-investigation-and-round-recut.md` 提供第五至第八轮的职责切分。历史数字若与
当前调查冲突，以当前 HEAD 证据和 `doc/` 规格为准，不复用过时结论。

结构性选择必须先对照：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

这里的“对照”不是在收尾报告加一个章节。每当要决定 initial physical representation、analysis
authority、unknown、legality、verifier、fallback 或 provider boundary 应是什么形态时，先找到 ref
中的同类实现，说明它如何承载、我们当前差在哪里、差别在换一个 kernel 或移除一项前提时有什么
实际后果，再决定本仓库的实现。

已经核实、可以作为调查起点的 ref 事实包括：

- `ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:27-58`：无 encoding 的
  ranked tensor 先获得 typed default blocked encoding；已有 encoding 保留，需要变化时产生
  `ConvertLayoutOp`。它不要求 tensor 先匹配某个 kernel family 才能进入合法 physical IR。
- `ref/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95-104`：
  pass manager 在 transformation pass 后插入整模块结构 verifier。
- `ref/triton/lib/Analysis/` 中 `AxisInfo`、`Alias`、`Allocation`、`Membar`、`BufferRegion`、
  `BufferIndexAnalysis` 是多个 pass 消费的正式分析；证不出来时返回显式保守结果，不切换到另一套
  shape matcher 猜答案。
- TileLang 作者显式写 tile/loop，因此没有与 Intent automatic blocking 同名的 pass；但
  `ref/tilelang` 有正式 `BufferRegion`、pipeline dependence planning，以及 nested parallel、
  fragment-loop indexing、buffer initialization、parallel race 等 semantic/physical legality checks。

这些是起点，不是要求照抄 Triton 类名或 TileLang TIR surface。Intent 作者没有写 block shape，
automatic blocking 是 Intent 独有责任；越是 ref 没有一一对应物，越要从它最接近的 typed default、
analysis result、legality 和 mutation 组织方式推导，而不是自己发明形状清单。

---

## 二、当前已经查实的根因，不要重新调查一遍

当前事实链是：

- `doc/compiler/gpu-program-ir.md:58-72,167-182` 和
  `doc/compiler/physical-parameters.md:18-31` 规定 fragment shape 只能包含常量与 compile-time
  physical parameter；
- `lib/Dialect/GPU/IR/GPUDialect.cpp:184-205` 的 `FragmentType::verify` 精确实现该规则；
- `lib/Conversion/KIRToGPU/KIRToGPU.cpp:899-935` 的
  `fragmentExtentForDimension` 对 launch-visible dimension 返回 runtime
  `PhysicalExprKind::Dimension`；
- `KIRToGPU.cpp:995-1052` 的 `convertTensorType` 随后把它写进 `FragmentType`；
- `lib/Dialect/GPU/Transforms/VerifyGPUProgram.cpp:19-60,238-241,350-362` 对 view、fragment、buffer
  shape 共用 symbol-existence 检查，没有在 whole-program verifier 中再次区分 runtime launch
  expression 与 compile-time fragment expression；
- 开启断言的同一 HEAD 只有 `46/217` 通过 shared gate；`RelWithDebInfo -DNDEBUG` 构建是
  `214/217`；release 构建继续到 Triton terminal 后是 `98 terminal_source / 116 provider
  verification / 3 physical_program`；
- fragment type invariant 来自提交 `9549983e`，launch-visible runtime fragment 分支来自
  `bded6edd`。后者的提交说明是 `compiler: establish complete shared GPU corpus gate`。

这说明 initial shared program 从构造时就可能非法，而 release build 把 type-construction assertion
关掉后掩盖了它。不能通过放宽 `FragmentType::verify`、把 runtime dimension 改名成 parameter、或在
Triton leaf 临时猜一个 block 常数来修。

当前 119 个未到 terminal source 的 release 分类也已经重新取证：

```text
100  runtime-dimension fragment
  7  mutable buffer
  5  masked gather / safe-index legalization
  3  provider candidate legality
  1  scaled-contract legality
  3  multi-source region traversal 不 lockstep
```

其中 100 个首先是 shared representation 错误；其余类别必须等 shared 主线修完后重新观察其
typed form，不能提前按旧诊断在 provider 里补分支。

当前 31 个旧 pass→non-pass 也不是一个原因：20 个 entry 含 runtime fragment，2 个 masked gather，
2 个 provider candidate legality，7 个已经到 terminal source、失败在
`examples/repro/v2/measurement.py:72-122` 的公平 candidate predicate。最后七个属于第八轮，不要在
本轮改 measurement 让 CSV 变绿。

---

## 三、执行顺序：先重建 authority，再关闭 coverage

这一轮的顺序是硬约束：

1. 先决定并实现合法 initial physical representation；
2. 再建立 shared analyses、decision authority、invalidation 与 verifier；
3. 让四个 family passes消费这些 authority，删除第二份推导；
4. 之后才运行完整 corpus/registry coverage；
5. coverage 暴露的失败先归类到缺失的 analysis、decision 或 provider-local form；不得直接在失败的
   family 文件里添加更窄 matcher；
6. shared gate稳定后，才实现 54-entry terminal-source 硬目标仍需要的 Triton-local forms；
7. 最后运行一台机器的 54-entry 数值正确性收口。

原因已经由前一轮实证：在四个 family 文件各自推导 range/replay/ownership 的结构下，最快关闭
coverage 的方法就是继续加更窄规则；第五轮约一万行后端正是这样产生的。Coverage 是新 authority
的结果，不是与重构并行、反过来定义重构的任务。

---

## 四、主线一：形成合法、保守、完整的 initial physical program

### 4.1 initial representation 的具体形态由 ref 对照决定

修复的外部性质已经由规格确定：runtime logical extent 不能成为 fragment static shape；它应通过
program space、physical traversal、coordinates 与 validity 连接到 compile-time physical fragment。

但 construction 的保守起点究竟是 scalar-owned fragment、一个正式 physical parameter、structured
operation 自带的 fragment，还是其它 typed baseline，不在 prompt 中预定。实现前必须对照
Triton default encoding、其 type conversion/materialization，以及 TileLang/TIR 对未调度 value 的
合法表示，回答：

- ref 如何保证任何合法输入先进入一份合法 IR，而不是先匹配一个优化形状；
- default representation 承载哪些事实，哪些在后续 pass 才 refinement；
- representation 未优化时如何保持完整可执行，而不是留下 provider 才补的洞；
- 当 rank、runtime extent、multi-source 或 effect 改变时，default path 是否仍成立。

根据这组证据选择 Intent 形态。不能把 `{64,128,256}`、统一 `OwnershipN` 或另一张固定表换个
位置继续使用；也不能把 runtime full dimension伪装成 compile-time parameter。

### 4.2 construction 完成即合法

`lowerCanonicalKIRToGPU` 返回后必须满足当前阶段声明的完整 physical-program invariant。后续 pass
可以改变 blocking、ownership、range、replay 或 structured realization，但不能负责把一份非法/
不完整 program 第一次补成可执行。

如果 `RegionFoldOp/RegionScanOp` 等 structured op 允许作为 construction baseline 留存，其 type、
source slicing、carry、effect/result flow 和 verifier 形态必须由 ref 中“合法高层 op 经过后续 lowering”
的做法支撑；如果不允许，就在 construction 中形成另一份完整 baseline。不能靠 verifier 例外或
串行数量级慢路径逃避。

### 4.3 build mode 不能改变 legality

类型/operation 不变量不能只依赖 C++ assertion。任何非法 input/intermediate state在 assertion 与
`-DNDEBUG` 构建中都必须产生相同的 typed diagnostic。修复当前四个 `cast<FragmentType>` 和四个
逆序 `iota_range` assertion 时，先确定它们缺的是哪项 analysis/decision authority，再在那个层次
fail closed；不要加“如果不对就跳过”的防御性分支。

---

## 五、主线二：建立 shared analysis 与 decision authority

### 5.1 不复制 ref 的模块名，复制其责任闭合方式

从本轮实际需要的事实出发，对照 Triton analyses 与 TileLang `BufferRegion`/semantic analyses，决定
Intent 需要哪些正式 query/result。至少要覆盖当前四个 family 重复推导的：

- physical source range 与 coordinate provenance；
- fragment/value replayability；
- ownership、program mapping 与 execution-group coverage；
- tail predicate、active member set、validity 与 fill/effect relation；
- access footprint、alias/conflict；
- buffer initialization、lifetime、visibility 与 owner；
- structured operands/results/carry/source traversal关系。

每项都必须回答：

- authority 读取 immutable KIR 还是 current GPU IR；
- exact/unknown/conservative 结果怎样表示；
- 哪些 passes消费；
- 哪种 mutation 使其失效；
- 何时从 current IR 重算；
- 哪些 execution-changing facts必须写入 current IR，不能只留 analysis cache。

unknown 的处理不能凭 Intent 自己定。对每类 unknown，去 ref 看同类分析是退化到合法保守程序、
拒绝 semantic illegality，还是把 legality留给下一层；说明我们的情况为什么属于同一类，再实现。

### 5.2 family passes保留，但只能消费统一答案

Reduce、scan、region fold/scan、ordinary/scaled contract 的 accumulator/carry semantics不同，可以保留
专门 mutation patterns。完成态不是所有代码合并成一个万能 pass，而是每类 shared fact/decision只
有一个推导 authority。

当前 `RealizePointwiseBlocking.cpp`、`RealizeReductionBlocking.cpp`、
`RealizeContractionBlocking.cpp`、`RealizeRegionFold.cpp` 共存在 231 处
`return std::nullopt/failure()`；数量本身不是问题。逐个处理到的关键是：前置条件不满足时，是
semantic illegality、analysis unknown、还是需要另一份合法 physical decision。这个分类必须用 ref
同类做法判断，不能把 direct load、单 reduction pair、unit-step range、all-source lockstep继续当成
唯一成功形状。

ordinary/scaled contraction、range/replay/tail/zero/ownership 等重复推导必须收敛到统一 authority。
多个 consumer和 rewrite pattern可以存在，但不得从 source ID、shape、origin、op邻接再算第二份。

### 5.3 文件拆分只随稳定职责发生

如果新 authority 已经形成稳定边界，可以随主线拆分 `KIRToGPU.cpp` 和 family analysis/mutation；
删除被替代的重复 helper和宽松 no-op path。不能先按行数拆大文件，也不能为本轮临时造一层目录。
TileLang `Bufferize.cpp` 的 provider/storage 拆分留给第七轮，除非本轮 shared authority迁移会直接
删除其中一份 shared 重推导。

---

## 六、主线三：逐条闭合 verifier 的完整性责任

`doc/compiler/gpu-program-ir.md:167-182` 定义的十二条完整性不变量，每一条都必须在报告中落到
以下两种结论之一：

1. 当前阶段由某个明确 verifier/analysis检查，并给出 current IR carrier与 file:line；
2. 本阶段明确不检查，并通过 ref 对照说明为什么安全、由哪个后续层拥有、提前检查会怎样拒绝合法
   program。

不能留下“规格写了但实现没说”，也不能为了完成清单一次性增加十二种 fail-closed matcher。

要处理的十二项是：

- executable physical kernel body；
- program space 与 logical effect/result coverage；
- SSA scalar/fragment/resource type legality；
- fragment extents 只含 constant/physical parameter；
- access resource、coordinates/provenance、active members、validity 与 fill/effect；
- control arguments、yields、dominance 与 carry；
- buffer initialization/first-write、lifetime、ownership 与 visibility；
- structured-op physical operands/results/semantic schema；
- non-atomic conflicting effects；
- runtime grid不读取 launch 后的 device data；
- physical range narrowing 的 subset/identity proof 与真实 IR rewrite；
- execution不依赖 KIR clone、axis/role string 或 side decision record。

另外明确处置 physical parameter domain legality，以及 atomic ordering/sharing。当前
`gpu.ParameterOp::verify` 只检查 candidate非空，`GPUOps.cpp:991-1009` 仍丢弃 ordering/sharing；
是否应由 shared verifier、provider verifier或 canonical verifier检查，必须依据 ref 对类似 typed
semantics与 hardware scope 的分层处理决定。

### 6.1 不把“证不出来”和“不成立”混在一起

三个 FlashAttention variants 当前因 multi-source physical traversal 无法证明 lockstep而失败。
方向上不能静默选择第一 source，但“analysis unknown”也不等于 sources一定不 lockstep。

对每项新检查，先在 ref 中回答：

- 哪些不满足意味着 program semantic/IR 本身非法，ref 会拒绝；
- 哪些只是 optimization fact unknown，ref 保留合法但保守的表示；
- 哪些属于 provider/hardware legality，ref 延后检查；
- unknown 如何在 IR/analysis中显式传播，mutation后如何重算。

不要在 prompt 的执行中套用预先写死的“legality拒绝、optimization退化”口号；以 ref 的真实实现和
当前 invariant决定每项归属。报告必须给双方 file:line及一个移除前提后的实际后果。

---

## 七、三个历史悬项必须在本轮有结论

`report/history/canonical-kir-reconstruction.md:151-157` 留下：

- tuple/record logical-buffer element；
- scaled-contract scale-axis relation；
- 一般调用前置条件。

本轮逐项检查当前 `doc/`、canonical KIR、shared construction、真实 corpus和 provider consumer，
并对照 ref 的 aggregate/buffer element、scaled operand relation、assumption/precondition表示。每项只能
落到：

- 当前语料/公开 surface 已能触发，因而本轮完整闭合；
- 当前作者 surface或 corpus确实触发不到，明确说明为什么不影响本轮 correctness closure、由哪个
  未来真实语言构造触发时才需要，并确保当前实现没有猜测该事实的隐式路径。

不能让它继续以“历史报告提过、这轮没再说”的方式消失。若发现现有 `doc/` 缺少作者可观察语义，
停止并提出具体设计歧义；不要根据当前实现或单个 target临时发明 public syntax。

---

## 八、shared 主线之后，恢复 54 个 Triton terminal source

`report/baselinev2/triton-*.csv` 的 registry 有 54 个 entry，每项都有同语言手写 Triton source。
source 能表达并运行该算法，因此这些格子不能以“Triton target不支持”结束。

本轮编译硬目标是：**54/54 generated entry 的全部 component 均到达 Triton terminal source。**

先重跑 shared 主线后的分类：

- runtime-fragment 类应随 shared representation修复消失，或转成准确的 shared decision diagnostic；
- multi-source lockstep若仍失败，回到 shared analysis/decision authority处理；
- safe gather、mutable buffer、scaled contract、candidate legality若仍以同一 typed provider-local form
  准确留下，再对照手写 Triton source和 `ref/triton` 实现对应 local legalization；
- 只在这时实现达成 54/54 所需的 Triton-local forms。

不允许为了 54/54 在 shared 层加 Triton-only机制，也不允许 serializer从 KIR shape/relation重建
blocking。每项 provider rewrite必须把完整 provider program变成另一份完整 provider program，
serialization前有唯一 spelling。

CSV 中七个 `candidate_contract_failed` 已经能到 terminal source，不在这项任务里修改；两个 source
侧 resource/compatibility gap不影响 generated lowering，也不能作为 generated terminal source 的
豁免。

---

## 九、三家 shared coverage 必须一起关闭

Triton registry不能代表 shared coverage。cuTile 和 TileLang 各有 37 个 entry，且各有约 30 个
Triton registry不覆盖的结构。

shared authority稳定后，对 37 个 cuTile + 37 个 TileLang entry 的全部 generated components运行
纯编译 probe：

```text
frontend → canonical KIR → KIR-to-GPU construction
         → 全部 shared passes → full shared verifier
         → 对应 provider legalization/verification（只用于阶段分类）
```

本轮要求：

- 74 个 entry 接收到的 shared program全部合法、完整、build-mode independent；
- 若继续失败，失败必须在准确的 provider-local form/capability/JIT之前的 legalization阶段，并能指出
  shared IR已经提供了什么、provider还缺什么；
- 任何 shared representation、ownership、access、buffer、structured flow缺口都属于 5c，必须修；
- 不实现 cuTile/TileLang provider forms，不处理 timeout，不跑它们的 GPU数值或性能。

这里不是用 cuTile/TileLang verifier替 shared verifier背书，而是用 Triton不覆盖的真实结构检验
shared representation。若 provider为了继续必须从 KIR/shape重建 shared fact，那也属于 5c 的 shared
boundary缺口，不能登记成“第六/七轮再说”。

---

## 十、正确性必须经过 54-entry 真实数值收口

纯编译通过不等于语义保持。shared 重写完成且 54/54 terminal source达成后，在一台当前可用 GPU
上把 54 个 Triton entries全部执行一次。

本轮只判：

- generated artifact 能 provider compile/JIT；
- generated kernel真实 launch；
- 使用现有 entry 的 numerical oracle/tolerance完成数值比较；
- multi-kernel entry按作者/registry定义的完整 callable closure执行；
- 每项失败准确落到 frontend/shared/provider/JIT/launch/numerical/adapter/source侧阶段。

不 benchmark，不比较 p50，不追 `1.05×`，不对齐或扩大 candidate set，不选择/报告性能 winner，不更新
baseline CSV。一个 entry只需完成必要 compilation/autotune后的一次真实数值运行；不要为测量精度
重复 benchmark。

source能在该机器运行的 entry，同时保留同语言 source 对照。已有 source resource/compatibility gap
只豁免 source侧 launch，不豁免 generated compile、launch和数值判定。若某个 entry当前没有独立判断
generated correctness 的 oracle，本轮不能把它记为通过；应闭合现有 repro adapter 的数值边界，
不得用 PyTorch composition冒充新的性能 baseline，也不得改作者算法或计时 scope。

这是本轮对 `AGENTS.md` 默认验证纪律的明确限定授权：使用一条可手动执行的现有 54-entry runner
命令完成整个 Triton 数值收口，不为每个 entry建立单测。除此之外不建 test目录、pytest、fixture或
长期 correctness runner。

---

## 十一、构建模式与三道验收闸门

本轮使用三道同时成立的闸门。任何一道失败，都不能以“gap 已记录”结束。

### 11.1 Build-mode legality

用开启断言和 `RelWithDebInfo -DNDEBUG` 两种构建，对执行时完整 `examples/kernels/` corpus逐 kernel
运行相同 production compiler边界。两者必须：

- 得到完全相同的成功/失败阶段；
- 对失败给出同一 typed diagnostic类别；
- 没有 C++ assertion、abort或 build-mode-only acceptance；
- 任何 `FragmentType` 在创建时与 whole-program verifier中都只含 constant/physical parameter。

执行时重新枚举 corpus；当前 `93 files / 217 kernels` 只是调查快照。数量有变化时报告起止数量和
真实 corpus原因，不写死 allowlist。

### 11.2 Shared decision authority

报告给出 before/after：

```text
physical fact/decision
→ authoritative KIR/GPU IR carrier
→ authoritative analysis/query及 exact/unknown 表示
→ 唯一 mutating transformation entry
→ family/provider consumers
→ invalidation/recompute rule
→ 独立重推导次数
```

同一 semantic/physical fact的独立推导次数必须为 1。多个 consumer和 rewrite pattern不计为重复；
任何 consumer重新从 source ID、shape、origin、op邻接得到第二份结论都算未完成。

### 11.3 External correctness

同时达到：

- 54/54 Triton generated entries 全部生成 terminal source；
- 74 个 cuTile/TileLang entries 的 shared stage全部合法完整，后续失败准确属于 provider local；
- 54 个 Triton generated entries在一台 GPU 上真实运行并通过数值判定；
- 十二条完整性不变量、parameter domain、atomic ordering/sharing和三个历史悬项逐项有明确归属。

217/217 或其它通过数可以记录，但单独不构成任何闸门。

---

## 十二、取证方式与临时工具

编译取证使用现有 `intent-compile`、`--stop-after-shared` 和 frontend/toolchain production API。
`python/intent/compiler/inventory.py` 已删除；需要批量遍历时写一次性脚本，运行后删除，不再把 runner
放进产品树。

本轮允许三类批量取证，因为它们直接对应上述闸门：

1. assertion/release 两种构建的完整 corpus compile-only对照；
2. cuTile/TileLang 74-entry compile-only shared/provider-stage分类；
3. 一条 54-entry Triton真实数值命令。

它们不得变成 test infrastructure。报告保存可手动复现的命令、阶段计数和逐 entry失败分类；临时
脚本、日志、cache和生成文件不进入仓库。

---

## 十三、主要实现完成后的自查

主要改动完成后先停止补 entry。从实际新增、修改、删除的 authority与decision出发，在
`ref/triton`/`ref/tilelang` 找同类实现；不要拿预写检查清单让代码自证通过。

每处对照必须给出：

- Intent 与 ref 双方 `file:line`；
- 两边的 IR carrier、analysis result、unknown/legality、mutation与 verifier职责；
- 具体差别；
- 换另一个真实 kernel、改变 source graph、增加一个 source、或移除一项 matcher前提后的实际
  后果。

新增的 default representation要证明不是只对触发它的 entry成立；新增 analysis要显示多个 family
consumer确实读取同一结果；新增 verifier要证明没有把 ref 中的保守 unknown错误变成拒绝。
找不出具体差异与实际后果等于自查未完成。“与成熟实践一致”“泛化守住了”“emitter变薄了”都
不是证据。

随后沿唯一 executable path回溯，删除被替代的 duplicate inference、旧 helper、宽松 no-op、
compatibility/fallback、临时 attr/side record和本轮取证产物。不恢复 KernelModel、Plan side attrs或
旧厚 materializer。

---

## 十四、本轮明确不做

- 不追任何 provider 的 `1.05×`；
- 不 benchmark，不更新六张 `baselinev2` CSV；
- 不审核 source/generated candidate公平性、winner或 measurement scope；
- 不处理七个 measurement `candidate_contract_failed`；
- 不处理 cuTile timeout、JIT或 provider forms；
- 不实现 TileLang storage/copy/sync/pipeline/provider forms；
- 不跑 H100/5090双机全量；数值只用一台机器；
- 不新增 public DSL construct，不改作者算法；
- 不为 Triton、cuTile或TileLang在 shared层加 target-specific机制；
- 不恢复 legacy path，不留 compatibility switch、默认值或 fallback；
- 不按 kernel名、entry名、op数量或 whole-region shape分支；
- 不按行数做无关重构；
- 不建 test目录、pytest、fixture或长期 inventory runner；
- 不写版本号、CHANGELOG、迁移指南或 deprecation标记。

如果实现发现 current `doc/` 缺少作者可观察语义，停止并报告具体设计分叉；只有 ref 与真实算法证明
规格缺少稳定语义/架构事实时才修改 `doc/`。现有规格已经定义的 invariant应修实现，不把当前
family限制写成新规格。

---

## 十五、产出与提交

产出：

```text
report/shared-gpu-correctness-closure.md
```

报告写清楚：

- initial physical representation最终如何确定，ref依据与 build-mode divergence如何消除；
- 新增/收拢的 analysis与decision authority、unknown、consumer、invalidation和独立推导次数；
- 十二条完整性不变量、parameter domain、atomic ordering/sharing的实际归属；
- 三个历史悬项的最终结论；
- runtime fragment、multi-source、safe gather、mutable buffer、scaled contract、candidate legality在
  shared修复后的重新分类；
- assertion/release完整 corpus逐 kernel一致性；
- 54/54 Triton terminal-source结果；
- cuTile/TileLang 74 entries 的 shared结果和真实 provider-local分类；
- 一台 GPU 上54-entry Triton数值结果与所有非通过项的准确阶段；
- ref双向对照的双方 file:line、具体差异和实际后果；
- 删除的第二份推导、旧路径与临时产物；
- 是否仍有会让第六、七、八轮重新处理 shared correctness 的问题。

实现过程中按语义完整的节点提交，不把半条旧/新 executable path留在提交边界。最后把实现、必要
规格修正、报告整理成连贯提交，删除全部临时脚本和日志，确认工作区干净。

