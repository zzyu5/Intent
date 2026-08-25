# 第一轮：Public DSL 与算法源码重构

这一轮以 `doc/programming-model/` 和 `doc/dsl/` 为最终规格，只重构作者可见的 public DSL、`doc/dsl/examples/` 中的规范示例，并迁移 `examples/kernels/` 中的全部 kernel。Frontend 与 canonical Kernel IR 从第二轮开始，本轮不进入编译器实现。

`doc/` 是权威，当前实现不是。不能为了保留旧代码而修改规格；如果真实代码证据表明规格存在无法消解的矛盾，停止并提出具体问题，不自行修改设计。

参考实现：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

参考它们如何表达 structured control、first-class reduce/scan、typed values、effects 和 verifier，不把其 GPU program id、tile、layout、storage 或 target primitive抬进 Intent DSL。

这一轮的完成标准不是“新 API 可以 import”，而是：

```text
最终 public DSL surface
    → 规范性 doc examples 能完整表达目标算法
    → examples/kernels 全部使用同一套 surface
    → 旧作者构造在源码中彻底消失
```

本轮只确定并迁移作者源码，不为尚未改造的 frontend/KIR 增加 compatibility lowering、旧新 surface 双路径或临时别名。第二轮负责让唯一 canonical KIR 真正接受这套最终 DSL。

## 一、收敛 public DSL

最终 public surface以 `doc/dsl/` 为准。

彻底移除：

- `I.auto(...)`；
- 所有 `I.partition(...)`；
- `I.state_stream(...)`；
- public `ordered`；
- 作者可见 physical atomic `scope=`；
- view constraints中的physical `layout`、alignment、contiguity、vector width；
- ordinary tensor element type中的`i4/u4/fp4`；
- contract中假的`multiply=`与`combine=`参数；
-旧的单函数`I.random(...)`；
-旧的`I.atomic_add`、`I.atomic_cas` public calls。

可以保留为surface helper、但必须具有唯一、可机械展开的逻辑语义：

- `reduce.sum/max`、`any/all`、`arg_reduce.max`；
- `ragged(...)`、`members(...)`；
- ordinary indexing/assignment；
- `sparse_contract_2to4`；
-常用pointwise helper。

新增或闭合：

- runtime `domain(begin,end,step)`与source-derived `axis[begin:end]`；
- generic typed reduce；
- generic typed scan，包含inclusive/exclusive和forward/reverse；
-正式contract签名；
- generic scaled-contract formats；
- generic sparse-contract format schema；
- histogram；
-完整atomic family；
- Philox4x32-10的`I.random.bits`与`I.random.uniform`。

## 二、闭合 domain、subregion 与 index relation 的作者语义

这是本轮最先闭合的共同地基。这里定义作者能写下和观察到什么，不预先规定第二轮的KIR class或ODS名称。

Public surface和规范示例必须无歧义地表达：

- source domain identity；
- source rank；
- subregion begin/end/step；
- source coordinate与local ordinal的区别；
- empty/tail semantics；
-每个result axis对应的source coordinate expression；
- data-derived index的SSA provenance；
- active validity；
- relation composition后的完整坐标关系。

`I.ragged`与`I.members`若继续存在，只能是下列逻辑结构的surface shorthand：

```text
outer domain
member source domain
offsets-derived source subregion
optional indexed mapping
```

它们不能拥有独立于domain、subregion与index relation的第二份语义。具体frontend desugaring和旧KIR节点删除由第二轮完成。

## 三、重建control与state语义

普通`if/for/while`、loop carry、`break/continue`与unordered `parallel`必须形成唯一structured control路径。

不为普通顺序循环保留`ordered`标记。

每一处旧`state_stream`必须逐个按真实语义分类，禁止批量改名：

1. 只需要最终summary且允许重结合：改为reduce；
2. 需要每个logical position的prefix：改为scan；
3. 任意连续source slice先形成region-level summary、再重结合：改为region_fold；
4. slice summary还要作用于incoming state并产生source-aligned输出：改为region_scan；
5. 严格顺序、动态停止、非结合state或ordered effects：改为普通`for/while`与loop carry；
6. page/window/chunk boundary影响读取成员、shape或ABI：显式计算boundary并建立source-derived subregion；
7. boundary只是physical blocking：从作者DSL中删除，由后续physical compiler重新建立。

`region_fold/region_scan`只适用于作者写下的homomorphism语义。不能因为旧代码使用了compiler-selected extent，就假定它满足分段不变性。

如果某处无法从源码、对应source实现和算法语义唯一判断属于哪一类，不自行选择算法，也不在第一处歧义就终止整轮。记录该kernel、源码位置、候选分类及各自依据，暂时跳过该处并继续扫描和迁移其余语料。完成全部93个文件、216个`@intent.kernel`的检查后，一次性提交完整待裁决清单；只有发现最终规格本身互相矛盾时才立即停止。未裁决位置不能被静默计作已迁移，得到用户决定后必须在本轮补完，不能有意留下新旧surface子集。

## 四、闭合structured operations

### Reduce与scan

Generic reduce/scan的public语义提供：

- source与accumulator schema；
- axes；
-逐component identity；
- typed pure combine region；
- explicit runtime captures；
- result schema；
- scan direction与inclusive/exclusive。

`reduce.sum/max`和`arg_reduce.max`必须真正归一到generic reduce，不能保留字符串combine第二路径。

### Region fold 与 region scan

`region_fold`让作者写下：

- 同一source axis上的一组source components；
- 对任意连续非空slice的typed pure `summarize`；
- typed pure `combine`与逐component `identity`；
- 不随slice切分的显式captures；
- 固定summary schema。

`region_scan`在同一summary algebra上再保存`initial_state`、`apply`和`emit`，并定义source-aligned outputs与final state。

所有source components必须具有同一source-axis extent，并按同一组compiler-selected boundaries同步切片。`I.indices(source_axis)`作为source component时，切片后仍表示absolute source coordinates；helper不得观察segment ordinal、segment count、chosen extent或chunk-relative ordinal。

这些helpers都是pure typed code，不得包含external/logical-buffer write、scatter、atomic、RNG或依赖调用次数的行为。退化成element-summary的region写法归一到ordinary reduce/scan，不能保留两条canonical语义。

Identity必须在真实数值语义下逐component中立。需要区分“没有成员”时，summary显式携带bool validity或等价typed state；不能用会在combine中产生NaN或只在假定值域内成立的finite sentinel冒充通用identity。当前FlashAttention规范示例的显式validity保持不变。

### Contract

Contract的public语义定义：

- paired reduction axes；
- batch axes；
- free/result axis order；
- accumulator dtype；
- zero-reduction行为。

Public API删除`multiply/combine`参数。其它semiring写成pointwise加generic reduce。

### Scaled contract

用正式format/schema替换当前FP8+E8M0硬编码路径。至少闭合：

- `e2m1`；
- `e4m3`；
- `e8m0`；
- carrier packing；
- scale group relation；
- reduction/batch axes；
- accumulator与rounding。

普通packed INT4/INT2继续使用`u8/u16/u32` carrier、显式bit/index arithmetic和ordinary contract。

### Sparse contract

建立generic `sparse_contract(format=...)`，至少支持：

- `one_of_two`；
- `two_of_four`；
- compressed ordering；
- typed logical-position metadata；
- metadata legality；
- reduction/batch/result rules。

`I.sparse_contract_2to4`只能是surface shorthand。

### Histogram

新增first-class pure histogram op。现有用buffer加atomic模拟histogram的example迁移到该op；真正需要外部原子更新的算法仍保留atomic。

## 五、闭合memory、atomic与RNG

DSL memory/effect语义必须区分：

- external read/write；
- logical-buffer read/write；
- pure gather；
- arbitrary-index unique store；
- scatter-reduce；
- atomic load/store/RMW/CAS。

Logical buffer允许完整初始化或未初始化。未初始化时，语言要求每个被读取element已有dominant write，不得隐含默认初始化；第二轮由KIR verifier落实这条规则。

Atomic public surface为：

```python
I.atomic.load(...)
I.atomic.store(...)
I.atomic.add(...)
I.atomic.compare_exchange(...)
```

同时提供exchange/max/min/and/or/xor等RMW能力。

要求：

-保留memory order；
- CAS返回`{old_value, success}`；
-不接受public `scope=`；
- physical scope留给后续physical compiler。

RNG删除xorshift/f32-only路径，建立：

```python
I.random.bits(seed, logical_counter)
I.random.uniform(seed, logical_counter, dtype=...)
```

逻辑随机bit stream固定Philox4x32-10，不能读取program/thread/lane identity。

## 六、迁移全部examples

迁移`examples/kernels/**/*.py`全部93个文件、216个`@intent.kernel`，不只迁移baseline registry引用的kernel。

重点族：

- streaming attention、paged attention、MLA、block-sparse attention；
- linear attention、selective scan、Mamba、gated delta；
- quantized、block-scaled、block-sparse contraction；
- ragged/grouped GEMM、MoE、nested ragged；
- split-K/two-pass；
- backward、atomic、compaction、routing；
-所有variants。

具体要求：

-删除全部`I.auto`、`I.partition`与`I.state_stream`；
-可观察parts改成explicit part domain、boundary arithmetic、source subregion、partial tensor和host-visible多kernel编排；
-不可观察blocking从DSL消失；
-ragged改成offsets/subregion/index relation；
-旧atomic调用改成新family；
- histogram使用first-class op；
-删除physical layout/alignment constraints；
- packed INT4/INT2使用carrier与显式decode；
-保留原算法、数值路径、kernel数量、effects和ABI；
-不为迁就当前compiler改变算法；
-不在source、registry或adapter中按kernel名添加语义分支。

`doc/dsl/examples/`是规范性参考，`examples/kernels/`最终必须使用同一套语言，而不是保留“理想DSL”和“实际可运行DSL”两套表面。

## 七、本轮边界：不进入 compiler implementation

本轮不修改frontend lowering、ODS/KIR operation、KIR verifier、Plan/GPU IR、provider pass或terminal materializer。它们由后续轮次按顺序重构。

本轮也不得为了让旧compiler暂时接受新example而：

- 保留旧surface别名；
- 把新构造临时展开回`partition/state_stream`；
- 在example中继续写physical blocking、layout、storage或provider分支；
- 改写作者算法以迁就当前KIR/provider；
- 新增只服务迁移期的flag、fallback或wrapper。

第一轮结束后编译链暂时不能接受全部新源码是允许的；必须把真实frontend/KIR缺口准确交给第二轮，不能用旧语义伪造通过。

## 八、保持接线边界

迁移examples时同步更新必要的：

```text
registry Entry
kernels.* import与symbol
source_runtime
Entry.examples inventory
```

不修改source算法，不改Baseline V2 CSV，不做全量benchmark。

## 节点二：结构改变后的强制横向自查

主要实现完成后，暂停继续加代码，完整静态审计一次。

必须回答：

1. 每项最终语义是否只有一个public surface，shorthand是否能机械解释；
2. 93个kernel是否全部迁移，是否存在只改代表kernel的纵向打点；
3. 每个旧`state_stream`是否按真实算法落到reduce/scan/region_fold/region_scan/ordinary loop，而非机械改名；
4. region source components是否同轴、同extent并按同边界切片；
5. absolute coordinate与local ordinal是否在作者源码中被正确区分；
6. explicit validity/identity是否在online summary、empty source和tail上数值成立；
7. 是否把physical tile、layout、scope、storage或target capability重新泄漏进DSL；
8. 是否有新op只服务一个example，实际应当是普通`@intent.fn` helper；
9. 是否为了旧compiler改写了算法、kernel数量、effects或ABI；
10. 是否用旧surface别名、临时wrapper或默认值掩盖第二轮尚未实现的缺口。

同时汇总所有尚待用户裁决的位置，确认已经检查完整个语料，而不是在第一个问号处停止。

节点二不产出文档，不建立测试脚手架。发现问题直接修；遇到语义分叉则停止向用户提问。

## 节点三：完成后的收尾自查

认为彻底完成后，再进行一次独立收尾：

-全仓搜索作者源码中的旧API与旧写法；
-确认public导出中没有旧surface别名或迁移期helper；
-确认93个kernel文件都使用最终DSL；
-确认理想examples与真实examples没有两套语言；
-确认registry/adapters没有因symbol迁移静默断开；
-检查目录归属是否符合阶段边界；
-检查diff中是否混入无关重构、README批量改写或实验数据；
-确认没有误改frontend、KIR、GPU IR或provider实现。

本轮不建立parser/checker/test脚手架，也不把旧compiler能否执行新源码当作DSL正确性的判据。第二轮只验证全部源码能够形成并通过canonical KIR；第一条真实GPU端到端repro从第三轮shared GPU IR与Triton链形成后开始。

## 最终交付

完成节点三后：

1. 提交全部必要修改并确认工作区干净；
2. 创建一份最终报告，建议路径：

```text
report/dsl-reconstruction.md
```

报告只写：

-最终public DSL发生了什么变化；
-删除了哪些旧作者构造和surface别名；
- 93个文件、216个kernel如何迁移，各类旧构造分别落到哪里；
- 按kernel区分机械迁移与手写语义/位级重写：surface换名、part边界算术等列为机械迁移；packed INT4/INT2 carrier解码与sign extension、histogram first-class改写、atomic family迁移、scaled/sparse format schema迁移列为需要后续数值重点核对的重写；
- summary validity/presence记账出现于多少个kernel、具体位置、形态是否本质相同，以及是否集中在causal/ragged结构；
- 全量扫描后一次性收集的待裁决位置、每处候选分类与依据，以及裁决后的最终落点；
-节点二、节点三发现并修掉了哪些真正问题；
-哪些frontend/KIR缺口必须由第二轮闭合；
-是否存在因规格矛盾而无法完成的源码迁移。

不把节点二、节点三的检查清单或过程日志写成额外文档。

---

这个 prompt 的核心边界是：本轮“完整”只指作者 DSL、规范示例与全部kernel源码完成迁移。Canonical KIR从第二轮开始，GPU physical compiler从第三轮开始。不能为了暂时保住现有provider通过率，让旧作者表面继续活着。
