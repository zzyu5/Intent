# 第三轮：重构 Shared GPU IR 与 Triton 纵向闭合

这一轮重构编译器的第二层物理IR；它是整个重构流程的第三轮：

```text
canonical KIR
→ shared executable GPU IR
→ shared GPU passes
→ provider-local legalization
→ terminal provider source
```

依据是：

```text
doc/compiler/README.md
doc/compiler/kir-to-gpu.md
doc/compiler/gpu-program-ir.md
doc/compiler/passes-and-analyses.md
doc/compiler/physical-parameters.md
doc/compiler/target-lowering.md
AGENTS.md
```

先完整阅读，再动手。`doc/` 是目标规格，当前实现不是标准。如果实现证明规格内部矛盾，停下来说明；不能靠 fallback、默认值或继续保留旧路径绕过去。

参考真实编译器：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

参考它们如何组织 shared physical IR、target-local legalization、verifier 和 terminal lowering，不照搬 TTGIR、TileLang dialect 或 pass 名称。

## 一、前置条件与已知起点

本轮以前一轮已经完成的 canonical KIR 为输入。KIR 必须已经：

- 完整保存作者算法；
- 使用最终 DSL 语义；
- 不再包含 `state_stream`、physical partition、旧 ragged executable ops 等废弃表示；
- 具有 typed control、relations、effects 和 structured operations；
- 通过 canonical verifier。

如果实际代码不满足这些前提，停止并把缺口退回第二轮关闭。不要在GPU IR里顺手修DSL/KIR语义，也不要加入兼容旧KIR的路径。

当前 physical 实现有几个已知结构问题，本轮必须消除：

- `intent_plan.exec_*` 只是把 KIR operation 改名后原样 clone，类型仍以 `AnyType` 为主；
- program mapping、range、transfer、reduction、contract、buffer 等主要存在于旁边的 side records；
- 多数 refinement pass 只改这些 records 或 attributes，没有改写 executable types、operations、regions 和 def-use；
- 三家 `ProgramForms`、materializer 和 handlers 仍重新读取 KIR、result shape、axis role、operation name；
- loops、pointer、mask、validity、workspace、persistent traversal、buffer form、copy/sync 等仍有一部分在生成源码时才被创建。

不能在这些旧结构上继续补字段。要把它们收敛成唯一的 executable GPU program。

本轮执行顺序必须是先竖后横。先选一个能够经过Triton真实运行和数值对照的kernel，只建立它所迫使出来的最小完整shared GPU IR、KIR→GPU construction、shared pass、Triton lowering和terminal source，使下列链第一次真实成立：

```text
DSL → canonical KIR → shared GPU IR → shared passes
    → Triton lowering → Triton source/JIT → GPU numerical run
```

这条纵向链只证明新链路能够执行，绝不替代横向工作。链路跑通后，必须继续按下面各节把GPU IR types/ops/invariants、construction、passes与Triton consumers横向铺到全部KIR families和语料结构；不能把“一个kernel通过”写成本轮完成。

## 二、建立真正的 Shared Executable GPU IR

具体 dialect 和文件如何组织由当前代码决定，但完成后必须存在一份 provider-neutral、可启动、可以脱离 KIR 独立验证和序列化的 GPU block program。

### 顶层与 launch

GPU physical module/kernel 必须显式表示：

- public ABI；
- compiler-private ABI resources；
- target capability object；
- physical parameters；
- program-space extents；
- program coordinates；
- execution-group segments；
- program coordinate 到 logical workset 的映射；
- tail/empty program validity；
- grid rank、grouping、swizzle、grid-stride 或 persistent traversal的实际结构。

Launch extent 使用 typed `LaunchExpr`，只能依赖：

- 常量；
- 已绑定 physical parameters；
- ABI scalar；
- host 可见的 shape/stride metadata；
- 无 effect 的整数和布尔运算。

不得依赖 device load、kernel-body SSA 或字符串公式。

一个 KIR kernel 仍对应一个 target artifact 和一次 launch。Compiler-private workspace可以成为明确的 hidden ABI argument，但不能引入隐藏初始化 kernel、隐藏第二次 launch 或跨-kernel同步。

### Physical types

至少闭合：

- scalar；
- logical index；
- predicate；
- parameterized fragment；
- external view；
- program-private、iteration-private、invocation-workspace buffer；
- typed dependency/token，仅在真实跨 provider dependency semantics 需要时存在。

Fragment 必须明确保存：

- element dtype；
- physical shape expression；
- fragment axes 到 logical coordinates 的 mapping；
- validity relation；
- owning program/workset。

它不能保存 warp/lane/register分布、MMA encoding、shared/TMEM layout。

Buffer 必须明确保存：

- element type和shape；
- allocation instance；
- abstract physical scope；
- ownership；
- initialization或first-write obligation；
- lifetime；
- sharing/visibility；
- workspace ABI requirement。

### Control 与 value graph

GPU IR 必须真正包含：

- scalar `if`；
- ordered `for/while`；
- physical blocking/traversal loops；
- execution-group dispatch；
- loop-carried scalar、fragment、record和buffer state；
- arithmetic、comparison、select、cast、bitcast；
- broadcast、reshape、transpose、join、record；
- scalar↔fragment conversion；
- pure rematerialization产生的真实 def-use。

不得用 `replay=true`、producer ID 列表或 stream record代替真实 executable graph。

### Access 与 effects

每次访问必须有显式：

- resource；
- coordinates；
- coordinate SSA dependencies；
- active validity；
- load fill；
- write collision/effect semantics；
- result relation。

共同 GPU access 至少包含：

- load；
- store/unique scatter；
- gather；
- scatter-reduce；
- atomic load/store/RMW/CAS；
- buffer load/store。

Pointer expression、cuTile tile index、TileLang BufferRegion、descriptor只是后续 provider representation，不能成为共同 access 的唯一表示。

### Coordinate provenance 与 predicate range

GPU construction必须机械带下canonical KIR已经保存的source identity/rank、typed coordinate expression、SSA dependencies、active member set与bounds。Helper call、slice、broadcast、reshape/transpose、tuple/record、integer arithmetic/comparison/select和indexed relation只能组合这些事实；传播不了就得到unknown，不能从shape、名称、相同extent或附近结构猜回。

Blocking形成current physical fragments后，shared pass可以把作者显式写下的coordinate predicate投影到当前physical ranges。只有单调性、bounds和set inclusion被证明时，才能把range分类为all-true、mixed或all-false。

例如对`q in [q_begin,q_end)`与作者predicate`q >= k`：

- `k < q_begin`是all-true；
- 与`[q_begin,q_end)`相交的部分是mixed，保留原predicate；
- `k >= q_end`是all-false。

这必须是generic coordinate/index-set analysis，不是attention、causal或kernel-name matcher。Rewrite真实修改physical loop bounds、access coordinates、active sets和validity SSA，不修改KIR logical source relation。

All-false traversal只有在每个free lane都能逐component证明structured result为identity且没有effect时才能删除。任一component无法证明，必须保留原traversal与predicate。Invalid load不发生memory access并返回显式同dtype fill；invalid write不产生effect。

### Structured operations

Physical reduce、scan、region fold、region scan、contract、scaled contract、sparse contract、histogram必须是当前GPU IR中的真实operations。

它们必须保存完整的：

- physical operands/results；
- physical axes和relations；
- accumulator；
- identity和typed pure combine region；
- direction及inclusive/exclusive；
- dynamic extent；
- KIR保留下来的数值schema；
- physical access和result mapping。

Physical region fold还必须显式包含segment traversal、lockstep source slices、typed summarizer、summary combine、identity/captures与result flow。Physical region scan还必须包含transition composition、incoming-state application、slice emitter、source-aligned output assembly与final-state flow。Segment extent可以是physical parameter，但segment ordinal/count/extent不能成为作者可观察value或ABI。

Identity必须是真正typed neutral element。需要区分空成员时，validity或等价typed state必须保留到能够按当前physical ranges证明消除的位置；terminal leaf不能用finite sentinel或默认值替换它。

未分块实现可以保守，但必须完整可执行，不能留下等待 leaf 解释的 record。

## 三、重写 KIR → GPU construction

Construction 不再 clone KIR function并改 operation 名称。

它应当使用 canonical analyses，完成：

1. 从 KIR effects、def-use、alias、structured axes和independence建立 logical worksets；
2. 将connected computations/effects组成execution groups；
3. 为每个group建立完整、保守但合法的program mapping；
4. 把ordered axes、reduction axes、scan axes、strict recurrence留在program内；
5. 把runtime ragged/member traversal保留为program内部动态subregion；
6. 建立所有scalar/fragment/control/access/buffer/structured-op节点；
7. 建立origin map；
8. 得到第一份完整、可执行、可验证的GPU program。

Construction结束后，后续 pipeline不得再通过 KIR adjacency补执行结构。Origin只能用于诊断和semantic-preservation核验。

完成标志：

- 不再生成 `intent_plan.exec_*` KIR clone；
- 不再需要 cloned `func::FuncOp`解释执行；
- KIR即使不被provider pass解引用，GPU program仍能独立完成lowering；
- construction verifier能验证effect覆盖、ownership、dominance、access、buffer、structured op和launch legality。

## 四、把 shared GPU passes 做成真实 transformation

不要按旧文件名机械搬迁。Pass怎么拆由IR不变量、analysis依赖和真实rewrite关系决定。

至少要横向形成这些能力：

- program mapping refinement；
- execution-group placement；
- automatic blocking与fragment formation；
- ownership extent调整；
- grouping、swizzle、grid-stride或persistent traversal；
- value rematerialization/materialization；
- access realization；
- bufferization、lifetime和workspace realization；
- reduce/scan/contract accumulator realization；
- region fold/scan的segment、summary/state/output realization；
- validity与boundary neutralization；
- coordinate predicate range narrowing与identity-only traversal elimination；
- physical parameter declaration和legality约束。

每个pass必须说明并在代码结构中体现：

- 已物化进GPU IR或其analysis的canonical-derived typed facts，以及current physical facts；
- legality条件；
- 实际改写的types/operations/regions/def-use；
- 保持的算法语义；
- invalidated/recomputed analyses；
- pass后verifier。

只改attribute、range record、`persistent=true`或producer ID列表，不算完成。

同一个pass必须能根据typed facts在不同输入上产生不同合法决定。禁止：

- kernel名称；
- operation数量；
- whole-region模板；
-固定shape特判；
- provider字符串；
-设备型号分支。

如果选择persistent traversal，必须在GPU IR中创建真实loop/mapping；如果外部provider compiler已经可靠完成，则不要重复创建。

## 五、Physical parameters 与 tuner 边界

BM/BN/BK、ownership extent、reduction/scan chunk等参数必须是GPU IR里的typed symbols，实际约束：

- fragment types；
- loops；
- access coordinates；
- validity；
- grid/launch。

`num_warps`、`num_stages`、`num_ctas`等provider参数由Intent声明合法候选和约束，不由共享层用经验阶梯表选择winner。

Pass负责排除能证明非法的候选，例如：

- fragment/static extent不合法；
- grid rank不合法；
- provider surface不支持；
-已知资源上限必然超限。

未知的寄存器分配、机器layout和pipeline成本继续交给外部compiler/tuner。Autotune winner只存在于runtime/tuning artifact，不写回GPU IR。

## 六、收敛 Triton Provider Leaf

共同GPU IR始终是唯一完整executable authority。本轮只闭合Triton，不能为了想象cuTile/TileLang以后需要什么而向shared IR预加字段；真实差异由第四轮拿另外两家验证。

### Triton

普通结构应直接映射：

- program mapping → `tl.program_id`；
- fragments → scalar/blocked tensor values；
- explicit access → pointer/mask/other；
- reduce/scan/contract → Triton原语；
- region fold/scan → 已形成的segment loops、summary/state flow及其中的Triton structured原语；
- structured control → Triton control。

只有真正改变provider program operands、types或legality的descriptor等能力，才建立Triton-local extension。

Descriptor extension若存在，必须显式保存：

- base；
- shape；
- strides；
- block shape；
- padding；
- alignment；
- accesses；
- allocator ABI和lifetime。

不能在materializer中根据KIR relation临时创建。

cuTile和TileLang的provider-local legality、storage/copy/sync/pipeline与native operand forms全部留到第四轮。第三轮只允许阅读它们确认shared abstraction没有显然写死Triton，不实现、不保留兼容路径，也不据此声称另外两家已经覆盖。

## 七、Triton Terminal serialization

Serialization前必须经过provider verifier，证明：

- 每个common operation有唯一lowering；
- 每个local extension完整；
-所有physical/provider parameters已绑定；
- unsupported组合已明确拒绝；
- 不存在未materialized节点。

Serializer只能：

- 发imports、signature、decorator；
- 顺序打印current provider program；
- 转换types、attributes和API spelling；
- 打印已经声明的Config candidates与launch wrapper；
- 为已经显式声明的resource requirement生成机械runtime绑定。

Serializer不得：

- 读取或重新分析KIR；
- 从result shape猜fragment；
- 从role名称选择mapping；
- 从index relation重建pointer/mask；
- 创建loop、workspace、buffer、persistent traversal、copy、sync或pipeline；
- 从KIR或predicate重新推导region segment、causal range、summary identity、incoming state或output assembly；
-增加未声明参数；
-吞异常并切换fallback。

`TargetProgramOp.source`只保留为终端产物容器，不能继续承担补全physical program的职责。

## 八、examples 与接纳范围

完整扫描`examples/kernels/`和Triton registry，把第一条纵向链扩展为Triton对全部KIR families的横向接纳。

所有examples必须：

- 使用最终DSL；
- 生成最终canonical KIR；
- 进入唯一KIR→GPU construction；
- 不依赖旧Plan、旧handler或兼容路径；
- 不为了GPU lowering写program ID、tile、storage或provider分支。

发现仍使用旧DSL的example时，把它视为第一轮遗漏并按最终规格修正；不能在本轮为旧surface增加GPU兼容路径。已经符合最终DSL的文件不为改而改。

横向检查至少覆盖：

- runtime while；
- ragged grouped GEMM；
- paged/ragged attention；
- ordered scan；
- scaled split-K contraction；
- sparse/block-sparse contraction；
- logical workspace与动态规划；
- atomic/scan/scatter routing；
- sparse MLA backward；
- record reduction、多输出和InOut state。

这些结构不是用来添加kernel特判，而是检查GPU IR schema是否完整。

任何example若无法进入shared GPU IR，必须精确归类：

- KIR语义缺失；
- KIR→GPU construction缺失；
- shared GPU legality不成立；
- provider surface不支持；
- hardware resource不支持。

不能统一标成compile failure，也不能发射慢路径冒充支持。

## 节点二：主要结构完成后的强制横向自查

主要代码结构形成后暂停实现扩张，做一次内部审计，不单独产出文档。

逐项检查：

1. 每个canonical KIR family是否都产生完整GPU operations，而不是clone或record；
2. GPU IR在不回读KIR时能否独立verify和serialize；
3. 每个shared pass是否有真实IR前后差异；
4. 每个pass是否至少能由两类不同example的typed facts触发，而非单点形状；
5. 是否仍存在`intent_plan.exec_*`、physical KIR clone或side-record executable authority；
6. provider passes是否仍调用canonical `KernelModel`、扫描KIR graph、读取`intent.result_shapes`或role名称重建结构；
7. materializer是否仍创建loop、pointer、mask、validity、buffer、workspace、persistent traversal、copy、sync或pipeline；
8. region fold/scan是否在GPU IR中具有完整segment、summary/state/output flow，而非等待Triton解释；
9. coordinate provenance是否机械传播，causal等range narrowing是否只依赖typed predicate和current physical ranges；
10. all-false删除是否逐component证明identity且无effect；
11. Triton普通路径是否确实是thin lowering；
12. 是否为了尚未实现的cuTile/TileLang预置未经真实差异逼出的shared字段；
13. physical parameters是否真实进入types/loops/access/launch；
14. tuner是否只选择已声明参数/form；
15. 是否存在kernel名、shape特征、provider字符串或设备型号分支；
16. unsupported是否在最早拥有足够信息的层报出；
17. examples中是否仍有旧DSL或为旧compiler写的绕行结构。

发现问题就在本轮修掉，再重复审计。不能把它们登记成后续清理。

## 节点三：完成后的纯减法收尾

功能闭合后，再做一次收尾审计：

- 删除KIR clone conversion；
- 删除generic `exec_*`；
- 删除只描述外部clone的Axis/Range/Transfer/Contract等旧side-record路径；
- 删除shared与Triton路径重新`analyzeKernel`、`kernel.nodes.lookup`、KIR graph walk的决策路径；
- 删除Triton materializer中的physical binding、ragged metadata、pointer/mask/validity/shape重建；
- 删除Triton terminal no-op handlers和canonical-op直接dispatch；
- 删除Triton scalarized/serial/slow fallback；
- 删除旧Triton provider form字符串、未使用helper和重复verifier；
- 删除shared/Triton旧role-based mapping与固定winner路径；
- 检查没有注释保留、compatibility flag或第二条执行链；
- 检查目录结构真正表达shared IR、shared passes、provider extensions、serializer边界；
- 检查工作区没有缓存、临时IR或生成源码。

厚的provider代码如果只是机械打印已选extension，不因代码量大而删除；有意声明并由tuner实测的等价provider forms也不是冗余。

## 九、唯一验证

只使用一条现有、可手动执行的端到端repro，不建立测试目录、pytest、fixture或额外脚手架。

这条命令在最初纵向链形成时运行一次；横向铺开和收尾完成后必须再次运行，确认重构没有破坏已经成立的唯一链。优先使用：

```bash
./examples/run/baseline-v2.sh triton /tmp/intentdsl-gpu-ir.csv grouped_gemm
```

它应覆盖：

```text
DSL
→ canonical KIR
→ shared GPU construction
→ shared passes
→ Triton legalization
→ terminal source
→ Triton JIT
→ GPU numerical comparison
```

这条命令只证明一条纵向链真实可运行，不作为横向覆盖的替代。其它结构由前述IR/schema/consumer审计负责；cuTile与TileLang由第四轮分别建立真实provider证据。

不更新仓库中的baseline CSV。

## 十、交付

完成后只产出一份报告：

```text
report/gpu-program-ir-triton-reconstruction.md
```

报告写清楚：

- 最终shared GPU IR的组成和不变量；
- KIR→GPU construction如何建立完整initial program；
- 哪些passes产生了真实IR变化；
- Triton leaf最终持有什么；
- 删除了哪些KIR回读、side-record authority和fallback；
- examples接纳与结构覆盖；
- 节点二、节点三发现并修掉的问题；
- 唯一repro命令及数值结果；
- Triton横向接纳中剩余明确unsupported的真实性质；
- 第四轮仍需由cuTile/TileLang真实差异检验的边界。

节点二、节点三不单独建报告。

最后：

1. 将代码与报告整理成一个语义连贯的提交；
2. 确认工作区干净；
3. 不写版本号、迁移指南、CHANGELOG或额外进度文档。
