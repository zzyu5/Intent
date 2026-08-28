# 调查、对照 ref，然后重写第五轮 prompt

你这一轮**不执行重构**。你要做的是：先把当前编译器的真实结构调查清楚、跟 `ref/triton`
和 `ref/tilelang` 逐处对照，然后**写出一份新的 `report/prompt/05-*.md`**，替换现在那份。

后面还有第六、七、八轮（cuTile、TileLang、全量），这一轮只定第五轮。但你的调查结论会决定
后三轮怎么切，所以调查要按四轮的整体来做，结论写进报告，prompt 只写第五轮。

---

## 一、第五轮为什么失败：事实

原第五轮的任务是「在重构后的新链上做三家两机全量回归 + 把 Triton 严格可比项压到 1.05×
以内」。它没有完成，完成度自评 44%。

已经发生的事实（`report/full-gpu-regression-and-triton-performance-closure.md` 里有完整
记录，先读完）：

- 从 `669410b` 到 `bab7a40` 共 26 个提交、88 个文件、约 `+15589/-1474`；
- 其中 `RealizePointwiseBlocking` +2461、`RealizeReductionBlocking` +2040、
  `RealizeRegionFold` +1824、`RealizeContractionBlocking` +1561、`KIRToGPU` +2196；
- 也就是说，**一个「最终验收」轮里写进了约一万行新后端**。

原因很清楚：第三、四轮的报告**自己写明了**横向没有完成（`region_fold`/`region_scan` 的
physical segment loop、multi-axis reduce、multi-contract joint realization、buffer
realization 全部列为 gap），但那两轮的验证只有「一条命令跑通一个 kernel」，所以「列出 gap
清单」就足以算交付，序列照常推进。第五轮的第一次全量把这件事变成了事实。

**这是这次流程设计的失败，不是 agent 敷衍。**新的第五轮 prompt 必须自带一道不能用清单
糊弄的闸门。

当前状态（读 `report/baselinev2/*.csv` 自己核）：

```
triton-5090   51/54     triton-h100   52/54     三个非通过全部在 source 侧
cutile-5090   14/37     cutile-h100   13/37     各 14 个 worker_timeout
tilelang-5090  9/37     tilelang-h100  0/37     H100 表是修复前的旧快照
```

旧链曾达到 cuTile 35/37、34/37，TileLang 19/37、18/37。**这是能力回归，不能因为执行路径
换过就注销。**Triton 侧 21 行超过 1.05×，其中 `moe_splitk_expert_projection` 2.68×、
`flash_attention_forward` H100 2.27×、`paged_mla_decode` 2.17× 是结构性差距。

---

## 二、待验证的假设——这些是猜测，不是结论

下面几条是讨论中形成的判断。**它们可能是错的。**你的第一项工作是逐条验证或推翻，用当前
代码和 ref 的具体位置说话。推翻其中任何一条都是有价值的结论，不要因为它写在这里就往上凑。

### 假设 1：三家 registry 几乎不重叠，所以 Triton 通过不能证明 shared 层完整

初步统计是 cuTile 的 37 项里 30 项不在 Triton registry，TileLang 同样 30 项不在。如果成立，
那么 cuTile 的 `physical_program_verification_failed` 和 TileLang 的
`physical_program_verification_failed` / `physical_program_failed` 是**落在 Triton 从不
触碰的 kernel 上的 shared 层真实缺口**，而 Triton 51/54 对此毫无说明力。

自己重新统计，并确认这些失败到底发生在哪一层。

### 假设 2：shared 层没有 analysis authority

`lib/Dialect/GPU/` 下似乎没有 `Analysis/` 目录，只有一个约 693 行的 `Transforms/Utilities.cpp`
装着 `stripBroadcast`、`replaceSourceExtent`、`projectPredicate`、`zeroFill` 这类工具函数。
而 `ref/triton/lib/Analysis/` 有 `AxisInfo`、`Alias`、`Allocation`、`Membar`、`BufferRegion`、
`BufferIndexAnalysis` 六个真实数据流分析，被多个 pass 共同消费。

如果这个差别成立，含义是：**我们的每个 pass 自己就地推导它需要的事实**，第五轮报告里那句
「dependence/access region 尚未成为足够统一的 analysis authority」就是这个的具体形态。

### 假设 3：pass 是按 op family 切的，不是按 decision 切的

我们有 `RealizeContractionBlocking`(≈2700)、`RealizePointwiseBlocking`(≈2461)、
`RealizeReductionBlocking`(≈2673)、`RealizeRegionFold`(≈1824)——四个文件各自做 blocking。
Triton 的 TTGIR pipeline 是 `Coalesce`、`Pipeline`、`AccelerateMatmul`、
`OptimizeThreadLocality`——按变换切，每个跨所有 op family。

如果成立，含义是 blocking、ownership、fragment 形成、tail predicate、accumulator carry
**各被实现了四遍**，而且报告里那两个 gap（multi-contract joint realization、
execution-group 联合 physicalization）不是「还没做」，是**这个结构做不了**。

验证方法不是数行数（代码可以搬位置），是数**同一个决定被实现了几次**：tail predicate
处理出现在几处、ownership 形成出现在几处、accumulator carry 出现在几处。

### 假设 4：pipeline 顺序是承重的，因为每个 pass 破坏下一个需要的信息

`runSharedGPUPasses` 似乎是一个手写函数序列而不是 PassManager，中间穿插 `mlir::verify`，
`realizeAccessComposition` 被调用两次。代码注释里写着「要趁 X 还完整的时候做 Y」、「后面的
pass 不能被要求重建 range provenance」。

如果成立，含义是：事实活在 IR 形状里，形状一改事实就没了，所以 pass 不能重排、不能插入。
Triton 的 pass 可以重排是因为事实从当前 IR 按需重算。

### 假设 5：`operator_kind` 是「去掉名字改用编号」

`operator_kind` 在 `IntentOps.td` 和 `GPUOps.td` 里都是裸 `I64Attr`，约 75 处引用分布在
11 个文件，其中约 41 处是直接 `== <数字>` 比较。cuTile 的 `nativeCombineKind` 把 `7/9`
合并为 max、`8/10` 合并为 min，而 canonical verifier 里 `9/10` 是 **NaN-selecting 变体**。

对照：MLIR 把 `arith.maximumf` 和 `arith.maxnumf` 拆成两个 op class，Triton 在
`OptimizeThreadLocality.cpp` 里明确按 `isa<arith::MaxNumFOp, arith::MinNumFOp>` 分支；
TileLang 用 `ReduceTypeEnum` + `IsSum()/IsMax()/IsBitAnd()` 这样的 typed 谓词。

如果成立，那 cuTile 那处合并是**静默语义收窄**（含 NaN 输入结果不同），而不只是架构不好看。
`RealizeContractionBlocking` 同时接受 `11` 和 `13` 当合取但没证明 `13` 的操作数是 `i1`，
是同一问题的第二个实例。

---

## 三、判据与哲学——这几条比上面的假设重要

### 自查不用清单，对着 ref 看

**这是这次流程最重要的一条教训。**

「没有按 kernel 名称分支」在原来的自查清单上，而上面那段 `operator_kind == 12` 的代码
**通过了这一条**。清单只能查你已经想到的东西；想不到的正是最危险的。

而「对着 ref 看同类问题他们怎么解」能抓到——你打开 `OptimizeThreadLocality.cpp` 看见
`MaximumFOp` 和 `MaxNumFOp` 是两个类，立刻会问「那我们怎么区分的」。

所以你写的新 prompt 里，**自查段落不要写成编号清单**。要写成：

> 挑出实现中承载语义事实、或做出物理决定的地方，在 ref 里找到同一件事的对应实现，说清楚
> 载体/结构差在哪、这个差别有没有实际后果，给出两边的 file:line。

唯一的约束是：**找不出具体差异等于没做。**「与成熟实践一致」这种句子不算结论。

### 判据在外部，不在自评

「泛化守住了」「决定都在 pass 里闭合」「emitter 只拼写」这类形式判据不算数——一个东西可以
完全满足这些形式，同时那个 pass 只有一个候选、逻辑是硬编码的、换个输入就退化。

算数的判据只有一种：**换个东西还成不成立、去掉某个假设还通不通。**报告里少写「我守住了
什么」，多写「我试了什么、结果如何」。

### 成熟不等于没有名字

把 kernel 名换成 role、source ID、整数编号或某个局部 matcher，仍可能只是把特例换了一种
编码。成熟性要由「同类成熟编译器怎样保存语义、表达访问区域、建立依赖、选择合法 provider
form，以及我们与它们具体差在哪」来判断。

### 时刻怀疑自己在做玩具

假 pass、假泛化、假 emitter 分离——这三样都能通过形式审查。诚实地问：新增或修改的 pass，
换一个输入会不会走到不同分支，还是永远只有一条路？补的能力，除了触发它的那个 entry，还有
别的什么会用到？

### 不留第二条路径

不保留新旧双路径、不加 compatibility flag、不用默认值补缺失语义、不按 kernel 名/形状特征
分支、不为某个 provider 在共享层加机制。表达不了就明确不支持，不生成慢一个数量级的路径
冒充通过。

---

## 四、四轮的形状

```
第五轮  shared 层：对照 ref 重构组织方式；Triton 不退化
第六轮  cuTile
第七轮  TileLang
第八轮  三家两机全量 + 暴露问题的修复
```

第五轮的候选内容（**你的调查要确认这个划分对不对，可以推翻**）：

1. 建立 analysis authority——access region、dependence、axis/coordinate provenance 成为
   可查询、可失效重算的分析，而不是每个 pass 就地推；
2. 按 decision 而不是 op family 重切 pass；
3. 语义载体 typed 化（`operator_kind` 是已知实例，ref 对照会扫出别的）；
4. 覆盖：全部 216 个 kernel 都能产出通过 shared verifier 的 GPU program。

**第 4 条是闸门，不是任务。**它 provider-independent、不需要 GPU、跑得快，正是第三轮缺的
那道闸门。

**但闸门本身不充分。**在当前结构下，关闭覆盖最省事的办法就是往那四个 Realize 文件里再加
更窄的规则——第五轮那一万行很可能就是这么来的。所以新 prompt 必须同时要求那个可数的性质：
**同一个决定被实现了几次。**搬代码搬不掉这个数。

第六、七轮的真实含义也要在新 prompt 的边界里说清楚：cuTile 和 TileLang 各有 30 个 Triton
从不触碰的 kernel，所以那两轮不只是「接 provider」，而是**用两家 provider 和一百多个不重叠
的 kernel 去检验第五轮那层分析是不是真的通用**。第五轮做不出那一层，第六七轮就会重演第五轮
——一边接 provider 一边补 shared。

---

## 五、`doc/` 与 `AGENTS.md` 的权限

**这一轮的 prompt 可以赋予修改 `doc/` 的权限，但只限于改规格事实。**

- 可以：ref 对照证明规格有洞的地方。例如规格现在没有说 canonical 的算术 operator 集合是
  什么、NaN-selecting max 与 ordinary max 是不是不同语义——MLIR 把它们拆成两个 op class
  就是证据，补上这一条是修 spec。
- 不可以：把实现限制写进去当规格。「contraction blocking 只支持 unit-step」不是规格，
  写进去就是把缺口合法化。
- 不可以：新增中间状态、迁移期概念、兼容层描述。

`AGENTS.md` 只加简短的一条，不要长篇：自查不用清单，对着 `ref/triton`、`ref/tilelang` 看
同类问题怎么解，说清楚差在哪、为什么、给两边 file:line；找不出具体差异等于没做。

---

## 六、你的调查要回答什么

不要按上面的假设逐条打勾。以下是**问题**，答案由你的调查给出：

- shared 层现在到底由什么承担「知道事实」的职责？ref 由什么承担？差别有没有后果？
- 现在的 pass 边界是按什么切的？ref 按什么切？如果不同，我们因此做不了哪些事？
- pipeline 的顺序能不能改？如果不能，是什么让它不能改？
- 跨层传递的语义事实有哪些？每一个的载体是什么？有没有消费者可能读错？（`operator_kind`
  只是一个入口，不是全集。）
- 第五轮补进去的那一万行里，哪些是真实的通用能力，哪些是为了让某类结构通过而加的窄规则？
- cuTile/TileLang 的 14+14 timeout 到底是什么——候选空间非法导致的浪费，还是单个候选本身
  就编译不完？（第五轮 `d580ba8` 用「收缩到合法且有意义的 candidate」关掉了 dense GEMM 的
  timeout，这条线索值得追。）
- 旧链 cuTile 35/37、TileLang 19/37 覆盖的 executable structure，新链缺的是哪些？
- 上面五条假设，哪些成立、哪些不成立、有没有我们都没想到的更重要的问题？

---

## 七、产出

1. **一份调查报告**，写清楚上面那些问题的答案、证据（两边的 file:line）、以及你认为
   第五到第八轮该怎么切。如果你的结论与本文的假设不同，直接说，并给出依据。

2. **一份新的 `report/prompt/05-*.md`**，替换现有那份。它要：
   - 给目标和判据，不给实现方案；
   - 每条断言都带具体事实（数字、文件位置、之前哪一轮的结论），避免重新发明或重复劳动；
   - 明确写出哪些不要追、为什么；
   - 自查段落是生成式的 ref 对照，不是编号清单；
   - 带一道不能用清单糊弄的闸门，以及那个可数的结构性质；
   - 边界段落完整重述（不留第二条路径、不按名字/形状分支、不加兼容层、不建 test 脚手架、
     不写版本号/CHANGELOG/迁移指南）；
   - 说明 `doc/` 与 `AGENTS.md` 的权限边界。

3. 不要在这一轮改任何实现代码。调查可以读、可以跑现有 repro 取证据，但不提交实现改动。

4. 旧的 `05-intentdsl-full-validation-triton-closure-prompt.md` 里关于**全量、公平比较
   资格、candidate set 对齐、measurement scope** 的内容是好的，不要丢掉——它属于第八轮，
   在你的调查报告里说明它应该迁到哪一轮。
