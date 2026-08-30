# 写第 5c 轮的 prompt

你这一轮**不改实现，也不重新调查**。产出是一份 `report/prompt/05c-*.md`。

---

## 最重要的一条：标准不由我们定

**遇到任何"该做成什么样"的问题，去 `/home/kingdom/phdworks/ref/triton` 和
`/home/kingdom/phdworks/ref/tilelang` 看同类问题他们怎么解，然后说清楚我们差在哪、
这个差别有没有实际后果。**

这不是 prompt 里的一个段落，是写 prompt 时的方法。**下面每一处"要达到什么性质"，都必须
在 5c 的 prompt 里以"去 ref 对照后决定"的形式出现，不能写成我们自己拍板的规则。**别人已经
解过的问题不要重新发明答案。

已经在这轮讨论里核实过的 ref 事实，可以直接写进 prompt 当起点（但不是终点）：

- `ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:27-58`：任何尚无 encoding
  的 ranked tensor 都先得到一个 typed default blocked encoding，已有的保留，需要变化时产生
  `ConvertLayoutOp`。它不要求一个 tensor 先被识别成某种形状才准进入 physical IR。
- `ref/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95-104`：
  在 pass manager 里每个变换 pass 之后都插一个整模块结构 verifier。
- `ref/triton/lib/Analysis/`：`AxisInfo`、`Alias`、`Allocation`、`Membar`、`BufferRegion`、
  `BufferIndexAnalysis` 是可复用分析结果，被多个 pass 共同消费；证不出来时是显式的保守结果，
  不换一套 shape matcher 猜答案。
- `ref/tilelang`：作者自己写 tile/loop，没有与 Intent automatic blocking 对应的 pass；它有
  reject-only 的 semantic checks（非法 nested parallel、fragment-loop indexing）和正式
  `BufferRegion`。

**"ref 里没有同名物"不等于我们不该做**——Intent 作者不写 block shape，这项责任是 Intent 独有
的。但那更要求去看 ref 在最接近的位置上怎么组织，而不是我们自己定一套。

---

## 这必须是最后一轮碰正确性的

后面只应该改性能 pass，不应该再解决正确性和结构问题。要让这句话成立，不能靠决心，只能靠
让 5c 的验证覆盖第六、七、八轮会施加的检验方式——否则那三轮会各自再暴露一次。

绕圈的机制是明确的：

```
第三轮   声称横向完成    验证 = 1 个 kernel          → 第五轮发现缺一万行
第五轮   声称无设计卡点  验证 = 217/217 shared gate  → 自查发现 ProgramGrid 回归
5b       声称地基修好    验证 = 98/217 + 1 个 repro  → 调查发现 217/217 是 -DNDEBUG 造的
```

**每一次的验证都比声明弱一个量级。**所以 5c 的 prompt 必须把下面三条写成本轮范围，不能留
给后面：

**一、shared 的合法性不能只用 Triton 验证。**cuTile 和 TileLang 各有 30 个 Triton registry
不覆盖的 kernel，调查已经确认当前所有 shared 失败都落在这两组不重合集合里。5c 要对
cuTile/TileLang 的 74 个 entry 也做纯编译 probe——**不修它们，只确认它们收到的 shared program
合法完整，失败全部是 provider-local form，没有一个是 shared representation 缺口。**如果有，
那就是 5c 的活。这样第六、七轮才只剩 provider 实现。

**二、正确性不能只用编译验证。**程序编得过不等于算得对。5b 那次 scaled contraction 丢失原
validity 就是编译完全正常、语义错误。**5c 要把 54 个 Triton entry 在一台机器上真实跑一次
数值对照**——只判数值，不判性能、不对齐 candidate、不追 `1.05×`、不更新 baseline CSV。那些
属于第八轮。这是"正确性收口"的最低证据，不做就没资格说后面只剩性能。

**三、verifier 的 12 条完整性不变量，每一条都要有明确处置。**调查报告已逐条对过当前实现，
多数只做局部检查。5c 不必全部收紧（那可能过度保守），但每一条必须落到"已检查"或"明确不检查
且说明为什么安全"——**不能留在"没说"的状态**，否则它就是下一轮暴露问题的入口。历史上悬了
四轮的三项（tuple/record buffer element、scaled-contract scale-axis relation、一般调用前置
条件）同样要各给一个结论：关闭，或者明确说明当前语料触发不到、什么时候才需要。

这三条让 5c 变大，但它换掉的是第六、七、八轮各自返工一次。**如果 5c 做不完，宁可缩主线，
不要缩这三条验证**——缩了验证，等于把这一轮也变成"声称完成、下一轮暴露"的第四次。

---

## 5c 要解决什么

### 主线：消除非法的 shared representation

调查报告 `report/shared-gpu-compiler-investigation-and-round-recut.md` 是这一轮的输入，不是
待验证的假设——它在当前 HEAD 上带 file:line、两种构建对照和 git blame 逐条取证。不要推翻它
重来，也不要为了写 prompt 再跑一遍同样的取证。

它的首要结论不是"119 个 Triton gap 要逐项补齐"，而是：

- `doc/compiler/gpu-program-ir.md` 与 `physical-parameters.md` 规定 fragment shape 只能含常量和
  compile-time physical parameter；
- `FragmentType::verify`（`lib/Dialect/GPU/IR/GPUDialect.cpp:184-205`）**精确实现了这条**；
- 但 `fragmentExtentForDimension`（`lib/Conversion/KIRToGPU/KIRToGPU.cpp:899-935`）在 dimension
  launch-visible 时直接返回 `PhysicalExprKind::Dimension`，`convertTensorType`（`:995-1052`）
  把它写进 `FragmentType`；
- `-DNDEBUG` 让 MLIR 的 type-construction 断言消失，`VerifyGPUProgram` 又只检查符号是否已声明，
  非法类型于是在 release 构建里流过整条 shared pipeline。

同一 HEAD 因此矛盾：断言构建 `46/217`，release 构建 `214/217`。而 `git blame` 显示 fragment
不变量来自 `9549983e`，写入 launch-visible dimension 的分支来自 `bded6edd`——提交说明是
`compiler: establish complete shared GPU corpus gate`。**建立那道 gate 的提交同时引入了绕过
它的路径。**

主线是调查报告 §10.1 的五项。**但每一项在 5c 的 prompt 里都要写成"对照 ref 之后决定形态"，
不要写成实现方案。**尤其：

- 消除 runtime `FragmentType` 之后，construction 拿什么当初始表示——`TritonGPUConversion` 的
  default-encoding 做法是最接近的参照，但 Intent 的 KIR 高于 TTIR，具体形态要由对照得出；
- 四个 family 共同的 source-range/replay/ownership 推导收回统一 analysis 时，`ref/triton/lib/Analysis/`
  的组织方式（谁产出事实、谁消费、unknown 怎么表达、mutation 后怎么失效）是参照，但不要照抄
  六个同名模块——TileLang 就没有那一套，它依赖 TIR/`BufferRegion`。

### 硬目标：`baselinev2` 的 54 个 Triton entry 全部生成 terminal source

这是本轮必须达到的外部结果。

它有现成的证伪基础：**这 54 个 entry 每一个都配同语言的手写 Triton source**，就在
`source/triton/` 下，编得过跑得动。**所以对这 54 格，"Triton 不支持"永远不是合法答案。**

目标是 **terminal source 生成**（纯编译，不需要 GPU），不是数值通过或性能达标。CSV 里两项
source 侧 gap（`source_device_resource_gap`、`source_compatibility_gap`）是 source 自己的问题，
不影响我们能不能 lower。

当前 CSV：`20 pass / 25 provider_program_verification_failed / 7 candidate_contract_failed /
2 source 侧`。**7 个 `candidate_contract_failed` 的 kernel 已经能到 terminal source**，失败在
`examples/repro/v2/measurement.py:72-122` 的公平比较 predicate，属于第八轮——不算进本轮编译
目标，也不要为了让它变绿去改 measurement。

### 构建模式无关

`217/217` 之所以无意义，是因为它来自关掉断言的构建。5c 验收必须包含：**断言构建与 release
构建对同一 corpus 的每个 kernel 给出完全相同的阶段和诊断**，并规定以后取证默认用开启断言的
构建。

---

## verifier 会不会变得过度保守——这一条必须由 ref 回答

这是这一轮最容易走偏的地方，**而且答案不能由我们拍板**。

风险是具体的：调查报告里那 3 个多 source 不 lockstep 的 FlashAttention variant，
`RealizeRegionFold` 现在会证明 lockstep、证不出来就 fail closed。方向对，但如果 source 事实上
lockstep 而分析看不出来，我们就拒绝了本来能跑的程序。**"证不出来"和"不成立"是两件事。**

5c 的 prompt 要把这个问题交给 ref 对照，具体要回答的是：

- **ref 把哪些东西做成"拒绝"，哪些做成"退化但仍产出合法程序"，依据是什么？**已知两边都有：
  TileLang 有 reject-only 的 semantic checks；Triton 的 `Alias` 证不出来时返回 `MayAlias`、
  `BufferIndexAnalysis` 证不出来时退保守路径——都是不优化，不是拒绝。这两类的分界在 ref 里
  是怎么划的，我们对应的每一条应该落在哪一侧。
- **ref 的 verifier 检查的是"IR 自洽"还是"程序完整"？**调查报告已经把
  `doc/compiler/gpu-program-ir.md` §10 那 12 条不变量逐条对过当前实现，多数只做了局部检查。
  一次性收紧 12 条很容易变成 12 种新的拒绝方式——**ref 在这件事上是怎么做的，加进去的每一条
  检查靠什么保证它不会拒掉合法程序。**
- **analysis 返回 unknown 时会发生什么？**`ref/triton/lib/Analysis/` 里 unknown 是显式结果。
  我们的对应位置现在是什么行为。

**不要在 prompt 里预先写死一条"legality 拒绝、optimization 退化"之类的分界规则。**那是我们
自己定的标准。要写的是：去 ref 看这条线在那里怎么划，我们的每一项落在哪一侧、依据是什么。

---

## "不要越修越乱"

调查报告 §10.3 已给出划分，照搬并写实。**同样地，这些也要能追到 ref 的做法，而不是我们的
偏好：**

**遇到覆盖失败，先归类"缺哪个 analysis/decision authority"，不在失败点就地加更窄的 matcher。**
当前四个 family pass 的前置条件（direct load、单 reduction pair、unit-step range、全 source
lockstep）不少来自当时覆盖过的语料形状。`TritonGPUConversion` 那种"先给默认、再改进"的组织
方式正是用来避免这种形状清单的——写 prompt 时把这条对照关系说清楚。

**必须后置**（调查已点名）：Triton 的 safe gather、mutable buffer、scaled contract provider
实现——除非 shared 主线修完后它们仍以同一 typed form 准确留下；`1.05×` 与 candidate winner；
cuTile/TileLang；H100 与 baseline 更新；纯按行数拆大文件。

**可随主线一起做**：改到相应文件时按已稳定的 authority 拆 `KIRToGPU.cpp` 和四个 family 的
analysis/mutation 边界；修掉 8 处 C++ assertion（4 处 `cast<FragmentType>`、4 处逆序
`iota_range`），让它们变成所属 authority 的 typed diagnostic；删掉被新 authority 取代的重复
range/replay helper 和宽松 no-op path。

**不要放宽 `FragmentType::verify` 让 kernel 通过**，也不要在 Triton leaf 把 runtime dimension
猜成某个 block 常数——两者都是把缺口合法化。三家后端都要求编译期定形的块，这一点在 ref 里
可以直接看到。

---

## 自查段落怎么写

生成式的 ref 对照，不是编号清单：**从这一轮实际动过的 authority 和 decision 出发，去 ref 找
同一问题的真实实现，给两边 file:line、具体差别、以及换一个 kernel 或去掉一项前提时的实际
后果。**对照要由 ref 里的真实结构生成问题，不是为预设结论找类比。

**找不出具体差异等于没做自查。**"与成熟实践一致""泛化守住了"不是证据。不要写"X → 1"这类
自评表。

---

## 验收

调查报告 §10.2 的性质清单，加上两条外部结果：

- 54 个 Triton entry 全部生成 terminal source；
- 断言构建与 release 构建逐 kernel 结果一致。

**217 可以记录，但不能单独作为验收**——它已被证明可以在不变量不被执行时毫无意义。

---

## 写 prompt 时的注意事项

- 每条断言带具体位置（文件、行、提交），让执行的人不用重新调查；
- 明确写出哪些不做、为什么；
- 边界完整重述：不留第二条路径、不加兼容分支/默认值/fallback、不按 kernel 名或形状特征分支、
  不建 test 目录/pytest/fixture、不写版本号/CHANGELOG/迁移指南；
- 取证可以用现有 `intent-compile`、`--stop-after-shared`、现有 frontend/toolchain API 组合的
  一次性命令。`python/intent/compiler/inventory.py` 已在上一轮删除；需要遍历 corpus 时写一次性
  脚本、跑完即删，不要再往产品树里放 runner；
- 编译取证（54 个 lower、74 个 cuTile/TileLang probe、两种构建一致）纯编译，不涉及 GPU；
  数值验证那一条要跑 GPU，但只在一台机器上、只判数值、不做 benchmark、不更新 baseline CSV。

产出就是那一份 prompt 文件。不要顺带改实现，不要再写一份调查报告。
