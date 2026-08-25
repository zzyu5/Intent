# 第五轮：两机全量回归与 Triton 1.05× 性能闭合

本轮验证前四轮完成的：

```text
最终 DSL
→ canonical KIR
→ shared executable GPU IR
→ shared passes
→ provider legalization
→ terminal source
```

目标不是把表跑出来，而是确认这条新链在完整语料上成立，并把 Triton 严格可比项全部追到 source 的 `1.05×` 以内。

## 一、先建立公平比较资格

全量entry以`examples/repro/v2/registry.py`为唯一运行清单，不得只跑前四轮碰过或当前已经通过的examples。当前清单是Triton 54条source-backed entries，cuTile与TileLang各37条；若本轮开始时registry已按正式source inventory发生有依据的变化，使用当时完整清单并在报告中说明，不能静默缩小分母。

运行性能前，逐个 Triton registry entry 核对：

- 算法结构一致；
- 输入、输出、状态更新和数值语义一致；
- dtype、累加精度、近似函数、tie/NaN规则一致；
- kernel数量和调用次数一致；
- 多kernel pipeline的stage边界一致；
- timed closure覆盖同一范围；
- workspace、metadata、输入转换和reset位于相同计时边界；
- 两侧都能在当前设备真实编译、运行并通过数值比较。

不能仅凭输出shape和数值接近就认定算法一致。

如果source是两个Triton kernels加wrapper，Intent example也应写两个`@intent.kernel`并由wrapper按同样顺序调用。不能让compiler自动拆kernel。

对使用`region_fold/region_scan`的entry，还必须逐项核对：

- source与Intent是否具有相同的slice summarizer和summary composition；
- scan是否具有相同的incoming-state application、slice output和final state；
- source members、coordinates、causal/ragged predicate与output assembly是否相同；
- source中算法可见的chunk/page/part ABI是否仍由作者显式表达；
- source只是physical blocking的chunk extent是否没有被误写进Intent DSL。

不能把普通ordered loop、显式chunk ABI或source真实分阶段算法改成compiler-selected region segmentation来取得性能；也不能因为两侧都叫attention/scan就认定算法一致。

如果当前DSL和source算法不同：

- 能对齐就修改example，照source算法结构写；
- 原有算法仍有独立价值时，保留为另一entry；
- 不能因为当前DSL不同就把source降级成“仅供参考”；
- 只有真实算法接口、调用边界或provider能力无法对齐时，才取消严格ratio资格。

不得通过改写upstream source来迁就generated。

## 二、Triton Config必须公平

### Source使用autotune时

双方必须具有相同语义的候选集合，包括：

- BLOCK/tile/chunk参数；
- `num_warps`；
- `num_stages`；
- `num_ctas`；
- `maxnreg`；
- grouping参数；
- provider-form参数；
- autotune key；
- heuristic派生值；
- architecture/capability过滤；
- early prune、perf model和top-k过滤。

参数名称可以不同，但物理角色必须对应。不能把source的`BLOCK_K=64`映射成generated中另一个含义的extent。

双方分别运行真实Triton autotune并选择自己的winner。不得：

- 固定同一个winner；
- generated使用更大的候选空间；
- source关闭autotune而generated打开；
- generated额外搜索`USE_TMA`或`USE_NATIVE_SCALED`，而source没有等价候选；
- 用kernel名称选择经验winner；
- 把更大的搜索预算冒充pass能力。

每个entry都要按当前input shape与device记录source/generated各自实际生效的过滤后candidate集合，而不只记录decorator里声明的原始集合。Architecture/capability filter、heuristic、early prune和invalid candidate必须两侧按同等语义处理。

### Source使用固定config或wrapper heuristic时

把该输入和设备下最终生效的config视为singleton candidate set，generated也使用同语义singleton配置。

不要擅自替source打开一个它原本没有的autotune空间，也不要让generated继续使用宽候选集合。

### Compiler结构与Config的关系

公平Config不要求generated复刻source作者手写的program ID、pointer、mask或具体blocking源码。Canonical KIR固定同一作者算法；shared passes负责把它lower成明确的GPU execution/value/access/structured-operation realization，并选择只属于compiler的physical decisions。

但要区分：

- pass通过typed facts确定的program mapping、blocking、value/access graph：属于compiler能力；
- Triton Config搜索的tile、warp、stage和provider form：属于双方必须对齐的调优预算。

不允许把本该由pass产生的结构藏进一个只对generated开放的Config候选。

候选资格和最终winner记录在本轮报告中，不给CSV增加字段或检查逻辑。

## 三、先修测量本身

当前硬门槛只有5%，测量偏差不能接近这个量级。先检查并修正现有runner中的公平性问题：

- generated与source不能永远固定先后顺序而不做反向复测；
- 双方必须完成JIT/autotune后再进入steady-state benchmark；
- CUDA Graph使用必须一致；
- cache处理必须一致；
- mutable/InOut entry必须在每次测量前恢复相同初态；
- prepare/reset不计时，但两侧规则必须一致；
- 多kernel pipeline必须完整包含在CUDA event范围内；
- 输入、workspace和metadata构造不计入p50；
- 数值失败不允许记录性能。

不要新建测试框架，也不要修改CSV格式。只修现有runner和adapter中真实影响公平性的地方。

接近门槛的entry至少使用独立进程重复测量，排除运行顺序、频率和偶然winner造成的误判。不能用“抖动”解释明显差距。

## 四、两机并行执行

5090与H100并行推进，不能等一台全部完成后再启动另一台。

同一台GPU上provider串行运行，避免三个runner争抢GPU。两个机器按相同阶段推进：

1. Triton全量；
2. cuTile全量；
3. TileLang全量。

每个并发任务使用独立build目录，避免共享`/tmp/intentdsl-build`产生构建冲突。

六张表仍为：

```text
report/baselinev2/triton-5090.csv
report/baselinev2/triton-h100.csv
report/baselinev2/cutile-5090.csv
report/baselinev2/cutile-h100.csv
report/baselinev2/tilelang-5090.csv
report/baselinev2/tilelang-h100.csv
```

不合并、不加版本号、不增加检查逻辑。

CSV字段固定为：

```text
kernel,case,generated_p50_ms,source_p50_ms,ratio,status
```

Generated与source都记录steady-state p50。JIT、autotune、输入/workspace构造不计时；多kernel callable两侧都计完整pipeline，数值失败不记录性能。

现有CSV只是重构前历史观察值。第一次全量后才形成当前状态，不能把旧数字当作本轮结果。

## 五、先处理正确性和回归

优先级固定为：

1. 数值错误；
2. generated编译或运行失败而source成功；
3. 重构前通过、重构后失败；
4. Triton严格ratio超标；
5. cuTile/TileLang性能差距。

每个失败必须定位到真实阶段：

- DSL/example算法；
- canonical KIR；
- KIR→GPU construction；
- shared GPU verifier；
- shared pass；
- provider legalization；
- terminal serialization；
- provider JIT；
- launch；
- numerical comparison；
- benchmark；
- source/adapter/environment。

不能继续使用宽泛`compile_failed`解释不同性质的问题。

Triton generated失败而source通过时，原则上是我们的缺口，必须修。cuTile/TileLang只有在确认目标surface确实没有等价能力后，才允许明确unsupported。

不能通过串行慢路径、缩小输入、减少输出、改变dtype或放宽误差冒充支持。

## 六、逐项关闭 Triton 1.05× 差距

对每个严格可比且`ratio >= 1.05`的entry：

1. 独立复测确认；
2. 并排读取generated和source；
3. 查看双方实际Config集合、裁剪后集合和winner；
4. 比较physical GPU IR；
5. 比较generated Triton source；
6. 必要时查看Triton生成的TTIR/TTGIR，判断下层看见的程序是否已经不同；
7. 只改变一个决定做A/B；
8. 在5090和H100分别确认。

根因按层处理。

### Example算法不一致

修改DSL example，使其写成source的算法、kernel数量和调用结构。不能让compiler替作者完成算法替换。

### Shared GPU pass选择不好

修改shared pass的typed policy，并让决定真实改变GPU IR：

- program mapping；
- blocking；
- ownership；
- fragment；
- value/materialization；
- access；
- validity；
- structured operation；
- buffer/lifetime。

修复必须能解释至少另一类结构为什么也适用。禁止kernel名称和固定shape分支。

### Provider form选择不好

修改Triton legalization/pass，使它正确消费shared GPU IR并选择已有Triton能力。

不能在serializer里增加“遇到这个kernel改发另一段代码”的分支。

### Terminal spelling错误

只有决定已由provider program唯一确定、问题确实只是Triton API拼写时，才修改serializer。

### 外部Triton compiler边界

只有满足以下证据后，才能归因于下层：

- 算法和调用边界一致；
- candidate set一致；
- winner配置可比；
- generated/source Triton结构及关键原语一致；
- 替代合法form做过A/B；
- 差距在独立复测中稳定存在。

不能仅凭“已经用了`tl.dot`”或“看起来差不多”宣布下层质量问题。

## 七、不得用调优掩盖pass缺口

每个性能修复都要回答：

- 改了哪一层当前IR；
- pass前后有什么真实type/op/region/def-use差异；
- legality条件是什么；
- 换另一个kernel或shape是否会产生不同决定；
- generated是否只是获得了更多Config；
- cuTile/TileLang是否会受到shared改动影响。

扩大Config集合只用于恢复双方相同候选预算，不能作为性能修复本身。

## 节点二：主要修复完成后的横向自查

暂停继续追数字，进行内部审计，不单独产出文档。

检查：

- 是否有修复只让一个entry变绿；
- 是否出现kernel名称、固定shape、source路径或provider字符串分支；
- 是否把shared缺口补进了Triton leaf；
- 是否让serializer重新创建loop、access、buffer、workspace或mask；
- 是否修改了作者算法而没有同步source调用边界；
- 是否存在generated/source候选集合不一致；
- 是否通过扩大generated搜索空间获得提升；
- 是否有source fixed-config却让generated autotune；
- 是否有多kernel scope不一致仍参与严格ratio；
- 是否有数值失败仍记录性能；
- shared改动是否在代表性cuTile/TileLang entry上造成退化；
- 每个pass修复是否能由至少两类不同结构解释。

发现问题立即修正，然后重复审计。

## 节点三：最终收尾

完成定向修复后：

1. 再次并行运行两台机器的完整六表；
2. 确认所有历史通过项没有退化；
3. 确认所有严格可比Triton entry在两台机器上均满足`ratio < 1.05`；
4. 对没有严格ratio资格的entry逐项写明真实原因；
5. 删除临时A/B开关、调试输出和实验性candidate；
6. 删除被新pass/form替代的旧路径；
7. 删除serializer fallback和只服务单entry的分支；
8. 检查没有更改输入规模、计时scope或容差；
9. 确认六张CSV格式不变；
10. 确认工作区没有临时IR、日志、缓存或测量文件。

## 八、报告与提交

产出一份报告：

```text
report/full-gpu-regression-and-triton-performance-closure.md
```

报告至少写清楚：

- 两台机器是否并行运行及各自大致耗时；
- 六张全量表的状态分布；
- Triton严格ratio资格如何判断；
- 每个Triton entry的Config候选对齐方式；
- 多少严格项低于1.05；
- 每个剩余非严格项为什么不适用该门槛；
- 每个修复落在哪一层；
- shared修复在其它结构/provider上的验证；
- 数值或编译回归的根因；
- 明确的provider/hardware/source边界；
- 两次强制自查发现并删除了什么。

最后将代码、六张CSV和报告整理成一个语义连贯的提交，并确认工作区干净。
