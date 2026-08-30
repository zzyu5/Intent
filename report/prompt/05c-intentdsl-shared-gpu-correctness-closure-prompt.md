# 第 5c 轮：收口 shared 物理决定，并完成 Triton 全量正确性与性能

这一轮从当前工作树继续，不重新调查已经闭合的历史问题。目标是完成第五轮最后一处 shared
physical-decision 重复，随后把当前 54 个 Triton entry 全部真实运行并测出两机性能。

重点仍然是编译器本身。运行和性能用于暴露当前程序结构是否正确，不是拿固定通过数反过来定义
实现，也不把 autotune、候选数量或 winner 当作 Intent 的编译能力。

---

## 一、开始前阅读与事实边界

从 `doc/index.md` 进入，完整阅读本轮直接涉及的规格：

```text
doc/compiler/README.md
doc/compiler/kir-to-gpu.md
doc/compiler/gpu-program-ir.md
doc/compiler/passes-and-analyses.md
doc/compiler/physical-parameters.md
doc/compiler/target-lowering.md
```

同时阅读：

```text
AGENTS.md
report/shared-gpu-analysis-pass-reconstruction.md
report/shared-gpu-reconstruction-completion-audit.md
```

结构判断继续对照：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

ref 对照不是实现完成后的说明章节，而是本轮作出结构决定的方法。每当需要决定 default physical
representation、analysis exact/unknown、legality、parameter binding、provider boundary或 verifier
责任时，先找 ref 中最接近的真实实现，说明它怎样承载、我们为什么相同或不同、换一个 kernel 或
去掉一项前提会有什么实际后果，再决定 Intent 的形态。

`ref` 没有与 Intent automatic blocking 同名的 pass，不代表 Intent 不该做。Triton/TileLang 作者
已经显式写下 block shape与 loop，而 Intent 作者没有；这部分是 Intent 独有责任。参照的是成熟
编译器怎样建立 typed default、传递 analysis、表达 unknown、改写 current IR并验证，而不是寻找同名
模块或照搬 target surface。

下列事项已经完成，本轮不重新调查，也不再把它们写成待办：

- logical buffer element 已收紧为 ranked tensor 中的单一 scalar dtype；tuple/record 不进入
  logical buffer；
- public precondition 只保留 closed typed 的 `assume_in_bounds` 与 view `alias/noalias`，没有
  任意布尔 `assume` 或优化 hint；
- scaled tensor contraction 已在对照 `tl.dot_scaled` 与 TileLang scaled GEMM 后采用 closed
  positional schema，通过 rank、轴位置、group size、carrier extent 与 scale shape legality
  表达 scale-axis relation，不再从任意 rank/shape 反推作者语义；
- runtime fragment、safe gather、structured source traversal、range/replay、lockstep reduction、
  provider parameter role 等问题已有当前实现和定位，不再沿旧分类重新开工；若本轮修改直接触及
  其中某条 authority，再从 current IR观察实际后果；
- assertion 与 `-DNDEBUG` shared corpus、cuTile/TileLang compile-only probe、以及旧改动下的
  Triton terminal-source probe都已经提供过定位信息。只有本轮修改确实触及相应结构时，才做受影响
  的定向确认，不为了重新得到同一个数字而全量复跑。

旧失败分类和旧 CSV 不再作为当前实现任务。不得沿历史数字重新补 matcher。

---

## 二、当前唯一的 shared 收口：ownership 必须被精化，而不是重复建立

当前 dense GEMM 已暴露一条确定的第二份物理决定：pointwise realization 为动态维度建立
`FRAGMENT_D*` ownership，tensor-contraction realization 随后又为同一 M/N 维度建立
`BLOCK_M/BLOCK_N`。当前未提交工作树正在尝试让 contraction M/N 复用已有 execution mapping axes，
这不是 HEAD 已闭合事实；旧 `FRAGMENT_D*` parameter、引用它们的 type/range/SSA 图和 tuner
dimension也尚未完整消失。

旧生成源码已经把后果量化清楚：

```text
1,997,568 个 triton.Config
约 1,997,673 行
约 362 MB
```

其中两个失效的 `FRAGMENT_D*` 各有 17 个候选，把本已过大的参数组合再次放大 `17 × 17`。
这不是普通 source-size 问题，而是同一 physical axis 同时存在两份 decision authority。

完成态必须满足：

- pointwise pass 可以先产生一份合法保守 ownership；
- tensor-contraction pass依据 typed source/dimension relation精化同一 axes，不追加第二组 M/N
  ownership；
- 精化后，旧 parameter result、`PhysicalExprAttr`、fragment/range type、launch expression、死 SSA
  和 provider config dimension 全部消失；
- 删除必须来自明确的 decision replacement 与 use rewrite，不能只在 serializer 隐藏参数，也不能
  依靠一个宽泛 DCE 掩盖仍被 IR 引用的旧决定；
- pointwise、reduction、scan、region fold/scan、ordinary/scaled tensor contraction 对同一 axis
  不得各自产生互不相知的 ownership；
- current Physical Program在任一 pass 边界都只有一份 execution mapping，verifier不得靠忽略
  stale parameter让它通过。

首先完成当前 `RealizeContractionBlocking.cpp` 的修改，然后从实际 def-use 与 parameter consumers
检查是否还有同类重复。不要重新展开第五轮所有历史检查，也不要借这个问题按行数重构 family 文件。

---

## 三、把 physical parameter 与 autotune policy 分开

这次 config 爆炸还暴露了另一处职责混合：当前 `ParameterOp` 既表达“这里存在一个待绑定的
physical parameter”，又用相互独立的 candidate domains表达 autotune policy；Triton legalization
随后把所有 domains做笛卡尔积。

即使删除两个 `FRAGMENT_D*`，dense GEMM 当前仍可能产生：

```text
BLOCK_M(4) × BLOCK_N(4) × BLOCK_K(3)
× GROUP_SIZE_M(4) × NUM_WARPS(6) × NUM_STAGES(6)
= 6,912 个 config
```

这不是 Intent 的编译能力，也不是比 Triton 更大的有效编译空间。手写 Triton 使用的是少量完整、
相关联的 `triton.Config` 元组，而不是独立标量集合的全面组合。

本轮按以下边界收敛：

- shared GPU program只表达 physical parameter 的角色、它约束的 current IR以及 typed legality；
- compiler不预测 winner，不把候选数量当创新，也不动态生成大规模搜索空间；
- 普通编译/正确性路径只需要一组确定、合法、性能不荒谬的静态默认 config；
- 可选 autotune接收少量、预先给定的完整 config tuples；tuple内的 M/N/K、warps、stages等数值
  保持关联，不再从独立 domains做笛卡尔积；
- provider可以绑定 shared program 已经声明的 blocking parameters与 provider-local launch options，
  也可以根据 current typed program和设备资源删除非法 tuple；但它不能从 kernel名称、shape或
  source graph重新发明 execution axes、blocking topology，或自行扩张候选；
- config tuples必须在形成 `tritonConfigsAttr` 和 terminal source之前绑定/过滤。当前 repro adapter在
  artifact已经生成后才筛选 dense GEMM 16 个配置，位置过晚，不能继续作为真实路径；
- Triton autotune如果启用，只负责在给定静态 tuples中实测 winner。winner不写回 shared IR，也不
  宣称为 Intent compiler 的能力；
- 当前阶段不建设 shape/device驱动的动态 candidate generator；未来若出现这类需求，在
  provider/runtime边界重新评审，不能塞进 shared pass。

当前实现并不存在“默认单 config / 可选 autotune”两条明确模式；建立这条边界正是本轮工作。
具体是调整 `ParameterOp` schema、增加 provider-local complete-config carrier，还是通过 compile
invocation向 provider program提供默认 binding与静态 tuples，由当前代码依赖和 ref 对照决定。
外部性质必须是：shared parameter authority唯一、terminal source规模有界、默认路径无需 autotune、
可选 tuner只消费完整静态 tuples。

---

## 四、analysis unknown 不是 program illegal

本轮修改 ownership、range、replay、parameter relation 或 verifier时，必须保留 WRITE-05c 确立的
分界：analysis证不出来，不等于程序不合法。

每个 exact/unknown 分叉都先对照 ref：

- semantic/type/SSA本身不成立时，当前层给 typed diagnostic；
- optimization fact unknown时，保留一份完整、合法、较保守的 current program；
- provider/hardware legality在 shared IR无法判断时，由 provider verifier消费明确 carrier后判断；
- 不允许用另一套 shape matcher猜答案，也不允许把 unknown静默解释为第一 source、默认 range或
  任意 block常数；
- mutation改变所依赖的 current IR后，analysis必须失效或重算，不能继续消费旧结果。

具体哪项属于拒绝、保守表示或后续 legality不能靠口号预定，要给 ref中的同类行为和实际后果。

Coverage仍然是 authority正确后的结果。遇到失败先问缺哪个 analysis、decision或provider-local
carrier，不在失败的 family 文件中就地加入更窄的 direct-load、rank、shape或op邻接规则。

---

## 五、实现后的检查服务于改动，不重新建立门禁工程

shared ownership和 config路径完成后，先看实际产物：

- dense GEMM 的 program mapping只有真实 M/N axes；
- `FRAGMENT_D*` 不再出现在该 kernel 的 parameter、type、range、launch或 config中；
- 默认模式只生成一组 config；autotune模式只生成显式提供的少量完整 tuples；
- terminal source的主体是 kernel，而不是上百万行配置；
- ordinary 与 scaled tensor contraction走同一套 ownership replacement原则；
- ref 中 Triton 作者显式写 program-id/M/N/K blocking、Triton compiler负责 layout/MMA/pipeline的边界
  没有被 Intent 重复承担。

本轮改动发生在 shared 层时，选真正受影响的 cuTile/TileLang contraction entry做快速 compile-only
确认即可；不固定要求重新遍历全部 74 entries。检查可以并行，目标是尽快发现 shared regression，
不是追求实验仪式。

原先 assertion/release逐 kernel一致和 74-entry provider probe已经完成，不把它们变成每个修改节点的
重复任务。若本轮修改触及 type construction、assertion或跨 provider shared representation，才做相应
定向确认。无论是否复跑，assertion开关都不得改变 IR legality或 diagnostic类别。

收尾时用 `doc/compiler/gpu-program-ir.md` 的十二条完整性不变量审视完成态。十二条每一条都必须在
报告中落到以下两种结论之一：

1. 当前阶段由某个明确 verifier/analysis检查，并给出 current IR carrier与 `file:line`；
2. 当前阶段明确不检查，并用 ref说明为什么安全、由哪个后续层拥有、提前检查会怎样拒绝合法程序。

这不是要求机械增加十二种 verifier。对当前语料没有触发、ref也没有要求本阶段承担的能力，不为了
填满清单投机新增机制；但不能留下“规格写了、实现没说”。本轮直接触及的 program mapping、fragment
extent、range rewrite、effect coverage与禁止第二份 side decision必须真实闭合。

---

## 六、Triton 54-entry 全量运行

shared路径收口后，直接使用当前 54-entry Triton registry。不要再停在“terminal source生成成功”；
每个 generated callable都要经历：

```text
terminal source
→ Triton compile/JIT
→ launch
→ numerical comparison
→ benchmark
```

多 kernel entry按 registry定义的完整 Python orchestration执行。source侧 resource/compatibility gap与
generated失败分开记录，不允许互相冒充。

运行时优先并行推进，机器空闲就使用，不为了追求测量仪式把可以并行的 entry全部串行。明显离群值
单项复测一次；不要围绕微小扰动反复运行。

### 6.1 候选公平性

性能比较使用以下规则：

- source有 Triton autotune configs时，generated使用按 `BLOCK_M/N/K`、program grouping、warps、stages、
  CTAs 等 semantic roles 映射后的等价完整 config tuples；字面参数名称无需相同，但每个数值必须控制
  同一物理角色；
- source只有一个 config或没有 autotune时，generated也使用对应的单一静态 binding；
- 两边使用相同 input、dtype、算法结构、调用次数和计时 scope；
- 两边各自由 Triton autotune选择 winner；Intent不预选 winner；
- 不允许 source关闭 autotune而 generated搜索，也不允许 generated通过扩大候选集合获得优势；
- 不把“同一参数名的独立 domain相同”当作公平；比较单位是完整 config tuple；
- 若 DSL与 source算法不同，先按 source真实算法修改 DSL/example。不得靠换候选、缩小 scope或在
  compiler中按 entry名补规则掩盖算法差异。

### 6.2 性能处理

在 5090 与 H100 上并行运行，更新：

```text
report/baselinev2/triton-5090.csv
report/baselinev2/triton-h100.csv
```

表格格式保持现状，不加版本号、不增加检查列。

这是本轮对仓库默认单条 repro 纪律的明确授权：可以用现有 registry runner并行完成两机 54-entry
数值与性能，不为它另建 test目录、pytest、fixture或长期 runner。运行服务于当前编译器收口，不把
固定通过数写成新的实现门禁。

对算法、调用和候选均对齐的 entry，目标仍是 generated不慢于 source `1.05×`。超过的逐项从
Physical Program开始判断：

- execution decomposition、ownership、blocking、range、residency或 structured realization不对，
  修改 shared pass policy；
- shared program已完整，但 Triton DSL form没有使用目标已有原语，修改 Triton provider
  legalization/serialization；
- DSL算法结构与 source不同，修改作者 example使算法对齐；
- 生成结构和原语均一致，替代 config/form A/B也不能改善，才记录为下层或设备差异。

每个结论必须有当前 generated/source代码和定向运行证据。不要复用历史报告中的“target能力边界”、
“下层质量”或“算法不同”作为现成答案。有手写 Triton source且能运行的 entry不能被定性为 Triton
无法表达。

两机 winner反转、绝对差仅几微秒或确认是运行态离群的，可以不硬追；但要用当前同候选运行确认，
不能凭旧 CSV判断。

---

## 七、边界

- 不扩 registry，不新增 kernel；
- 不实现 cuTile/TileLang provider forms或性能优化；
- 不按 kernel/entry名称、op数量或 whole-region形状在 compiler中分支；
- 不为 Triton在 shared层加 target-specific机制；
- 不恢复 fallback、compatibility switch、旧 Plan、厚 materializer或第二条 executable path；
- 不把 DCE、serializer过滤或 repro adapter后处理当作删除旧 physical decision；
- 不把 autotune、candidate数量、winner或静态配置表包装成编译器创新；
- 不建 test目录、pytest、fixture或长期 inventory runner；
- 不写版本号、CHANGELOG、迁移指南或 deprecation标记；
- 不重新调查已经闭合的三项规格分叉和旧失败数字。

若实现发现 current `doc/` 的最终语义与上述 parameter/config职责不一致，先给出具体冲突和 ref证据；
不要为了保住当前笛卡尔积实现而修改规格。

---

## 八、自查、报告与提交

主要修改完成后，先停止追单个数字，从本轮实际改动出发对照 `ref/triton`：

- 手写 Triton怎样表达 program grid、M/N/K blocking和完整 config tuples；
- TTIR/TTGIR passes怎样精化已有 encoding/layout而不是并存两份 authority；
- Triton compiler怎样消费已有 K-loop做 MMA selection与software pipeline；
- provider/runtime怎样选择 config winner，为什么不属于 shared compiler。

给双方 `file:line`、具体差别和换一个 tensor-contraction kernel后的实际后果。不要写检查表式自评。

实现、运行和清理完成后，再完整执行 `report/prompt/AUDIT-5c-state.md`。它只核查最终工作树是否与
本 prompt一致，不重新调查已完成事项，也不要求为得到旧数字而重复运行。

产出：

```text
report/shared-gpu-and-triton-closure.md
```

报告只写当前事实：

- duplicate ownership最终如何被精化和删除；
- stale parameter、dead SSA、重复 config dimension与旧后处理路径删了哪些；
- physical parameter与静态 complete-config tuples最终如何分层；
- Triton 54-entry 的 terminal/JIT/launch/numerical状态；
- 两机性能、`1.05×`内外数量，以及每个剩余超标项的当前证据；
- shared改动在受影响的其它 provider结构上有没有实际后果；
- 十二条不变量逐条的真实执行归属，以及本轮直接触及部分的实现变化；
- ref对照的具体差异。

不重复抄旧报告的历史数字和已经关闭的调查过程。

按语义完整节点提交。最终删除临时脚本、日志和巨大生成文件，提交实现、必要规格修正、两张 Triton
CSV与报告，确认工作区干净。
