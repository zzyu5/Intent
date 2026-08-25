# 第四轮：cuTile、TileLang 与横向闭合

这一轮以前三轮已经形成并真实运行的链为起点：

```text
最终 DSL
→ canonical KIR
→ shared executable GPU IR
→ shared GPU passes
→ Triton source/JIT/GPU numerical run
```

本轮用cuTile与TileLang检验shared GPU IR是否真的共同，并闭合两家的provider-local legalization、terminal serialization和全语料接纳。不能把第三轮的Triton通过当作另外两家的证据，也不能预先假定三家需要对称的leaf结构。

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

先完整阅读，再检查第三轮最终shared GPU IR和两家真实provider surface。参考实现仍为：

```text
/home/kingdom/phdworks/ref/triton
/home/kingdom/phdworks/ref/tilelang
```

## 一、先建立第二条真实纵向链

本轮同样必须先竖后横。开始时在cuTile或TileLang中选择一个能暴露真实provider差异、又能完成数值对照的kernel，只实现它迫使出来的最小provider-local operations、legality、serializer和runtime binding，使下列链先真实成立：

```text
same shared GPU IR
→ chosen provider legalization
→ provider source
→ provider JIT/launch
→ GPU numerical comparison
```

选择哪一家由真实代码耦合决定；优先选择能检验access、structured operation和至少一种provider-local form的kernel，而不是纯pointwise。第一条链跑通前，不得先铺设大批想象中的provider extensions。

这条纵向链只证明provider边界能够工作，不代表横向完成。链路成立后，继续闭合另一家，并按后续章节覆盖全部operations与语料；本轮结束前cuTile和TileLang都必须各有真实source/JIT/GPU数值证据。

## 二、Shared GPU IR 的复用边界

共同GPU IR仍是唯一完整executable authority。cuTile与TileLang必须直接消费第三轮已经形成的：

- program mapping、loops与physical parameters；
- fragments、records、loop-carried values与buffers；
- access coordinates、active validity、fills与effects；
- reduce、scan、region fold/scan、contract、scaled/sparse contract与histogram；
- coordinate provenance、predicate ranges与identity proofs；
- lifetime、sharing、dependencies与workspace requirements。

Provider不能回到KIR、origin、result shape、axis role或operation name重新建立这些结构。

如果真实cuTile/TileLang lowering暴露共同IR缺少跨provider成立的事实，可以修改shared IR/pass，但必须：

1. 说明该事实为什么不是target spelling或provider-local legality；
2. 让它由typed KIR/current physical facts产生；
3. 不为某一家在shared层加provider分支；
4. 重新运行第三轮Triton repro，确认既有链没有退化。

如果差异只改变provider operands、types、storage、copy、synchronization、pipeline或native primitive legality，则建立provider-local extension operation/pass，不扩张shared schema。

## 三、闭合 cuTile Provider

cuTile leaf主要完成：

- program coordinates到`ct.bid`；
- fragments与coordinates到tile indices；
- common access到`ct.load/store/gather/scatter`；
- structured ops到native reduce/scan/MMA；
- 已形成的region fold/scan loops与summary/state/output flow到cuTile control和native structured ops；
- grid rank、tile shape、bounds和MMA-scaled legality。

不得从KIR shape、旧state-stream parent、ragged relation、相同extent或role名称重新决定access、tile、range和validity。

Gather/scatter、bounds、collective、row occupancy或等价API forms若会改变cuTile program operands、access shape、tile或primitive，必须先成为cuTile-local form selection；只有form已经唯一确定后，API spelling才留给serializer。

若某种scalarized buffer、serial reduction或gather路径只是慢速替代而非等价native能力，应在cuTile legalization明确拒绝，不能保留为fallback冒充支持。

## 四、闭合 TileLang Provider

TileLang需要由真实surface逼出的provider-local结构，包括：

- storage allocation；
- `T.copy`/async copy；
- BufferRegion；
- synchronization/barrier；
- pipeline；
- native GEMM/reduction所要求的operand form。

这些结构必须先由TileLang-local passes根据共同GPU IR的lifetime、sharing、access、dependency和structured-op facts生成显式extension operations，再由serializer打印。

TileLang surface中的`T.gemm`、`T.reduce_max`等高层拼写用于兑现共同GPU IR中已经存在的contract/reduce；它们不构成在leaf重新选择算法结构的理由。Region fold/scan的segment slicing、summary algebra、incoming state和output assembly必须已在共同GPU IR形成；TileLang-local passes只能补native storage/copy/sync/pipeline与operand form。

不能继续在handlers里临时创建：

- `T.alloc_local/shared`；
- `T.copy`；
- `T.sync_threads`；
- `T.Pipelined`；
- packed INT2 helper；
- contract operand replay；
- storage/copy/sync topology。

某个TileLang native primitive如果只接受更窄的dtype、shape、layout或runtime-lane组合，应在provider verifier/legalization精确拒绝。不能生成慢一两个数量级的scalar/serial路径冒充完整支持。

## 五、Provider verifier 与 terminal serialization

Serialization前必须经过对应provider verifier，证明：

- 每个common operation有唯一lowering；
- 每个provider-local extension完整；
- 所有physical/provider parameters已绑定；
- region fold/scan的segment、summary/state/output flow完整；
- access、validity、storage、copy与synchronization合法；
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
- 从result shape猜fragment或storage；
- 从role名称选择mapping；
- 从index relation重建pointer、tile index、BufferRegion或mask；
- 创建loop、workspace、buffer、persistent traversal、copy、sync或pipeline；
- 从predicate重新推导region segment、causal range、summary identity、incoming state或output assembly；
- 增加未声明参数；
- 吞异常并切换fallback。

`TargetProgramOp.source`只保留为终端产物容器，不能承担补全provider program的职责。

## 六、全语料横向接纳

完整扫描`examples/kernels/`和cuTile/TileLang registry。所有examples必须：

- 使用最终DSL并生成最终canonical KIR；
- 进入第三轮形成的唯一KIR→GPU construction与shared passes；
- 从同一shared GPU IR进入对应provider legalization；
- 不依赖旧Plan、旧handler、KIR回读或兼容路径；
- 不为了某家lowering写program ID、tile、storage或provider分支。

横向检查至少覆盖：

- runtime while；
- ragged grouped GEMM；
- paged/ragged attention；
- ordered scan与region fold/scan；
- scaled split-K contraction；
- sparse/block-sparse contraction；
- logical workspace与动态规划；
- atomic/scan/scatter routing；
- sparse MLA backward；
- record reduction、多输出和InOut state。

这些结构只用于检查shared/provider schemas是否完整，不允许按kernel名称、operation数量、固定shape或whole-region模板添加处理。

任何entry失败必须精确归类：

- canonical KIR语义或第二轮迁移遗漏；
- KIR→GPU construction缺失；
- shared GPU legality不成立；
- provider-local representation/legalization缺失；
- provider surface真实不支持；
- hardware resource不支持；
- provider JIT、launch或数值失败。

不能统一标成compile failure，也不能发射scalarized、serial或其它慢路径冒充支持。

## 节点二：主要结构完成后的强制横向自查

主要代码结构形成后暂停实现扩张，做一次内部审计，不单独产出文档。

逐项检查：

1. cuTile与TileLang是否都直接消费同一份完整shared GPU IR；
2. shared GPU IR在不回读KIR时能否独立verify并进入两家legalization；
3. provider passes是否仍调用canonical`KernelModel`、扫描KIR graph、读取`intent.result_shapes`或role名称重建结构；
4. materializer是否仍创建loop、pointer/tile index、mask、validity、buffer、workspace、copy、sync或pipeline；
5. region fold/scan是否在进入provider前已有完整segment、summary/state/output flow；
6. coordinate provenance和predicate ranges是否由shared IR直接提供，而非两家各推一次；
7. cuTile form与TileLang extensions是否都显式存在、可verify且由真实surface差异逼出；
8. 是否有shared修改只服务某一家；
9. 是否存在kernel名、shape特征、provider字符串或设备型号分支；
10. unsupported是否在最早拥有足够信息的层报出；
11. tuner是否只绑定已经声明的参数/form，不重新选择value flow或physical skeleton；
12. 两家的第一条纵向链扩展到横向语料后是否仍能真实运行。

发现问题就在本轮修掉，再重复审计。不能把它们登记成后续清理。

## 节点三：完成后的纯减法收尾

功能闭合后，再做一次收尾审计：

- 删除剩余KIR clone conversion与generic`exec_*`；
- 删除只描述外部clone的旧side-record executable authority；
- 删除cuTile/TileLang重新`analyzeKernel`、`kernel.nodes.lookup`或KIR graph walk的决策路径；
- 删除materializer中的physical binding、ragged metadata、pointer/tile/mask/validity/shape重建；
- 删除handler临时创建的allocation/copy/sync/pipeline；
- 删除terminal no-op handlers和canonical-op直接dispatch；
- 删除scalarized/serial/slow fallback；
- 删除旧provider form字符串、未使用helper和重复verifier；
- 删除旧role-based mapping与固定winner路径；
- 检查没有注释保留、compatibility flag或第二条执行链；
- 检查目录结构真正表达shared IR、provider extensions、legalization与serializer边界；
- 检查工作区没有缓存、临时IR或生成源码。

厚的provider代码如果只是在机械打印已选extension，不因代码量大而删除；有意声明并由provider tuner实测的等价forms也不是冗余。

## 七、真实运行验证

本轮只使用现有runner，不建立test目录、pytest、fixture或额外脚手架。先用一个provider完成最初纵向链；横向实现完成后，cuTile和TileLang各选一个真实、非纯pointwise entry，执行：

```bash
./examples/run/baseline-v2.sh <cutile-or-tilelang> /tmp/intentdsl-provider.csv <entry>
```

每次都必须覆盖provider source生成、JIT/launch和GPU数值比较。构建成功、打印source或provider verifier通过都不能代替真实运行。

如果本轮修改了shared GPU IR/pass，额外重跑第三轮Triton repro。上述命令只证明纵向链真实可运行，不替代全语料IR/schema/consumer审计。

本轮不更新仓库中的baseline CSV；第五轮统一并行全量重测。

## 八、交付

完成后只产出一份报告：

```text
report/cutile-tilelang-provider-lowering-reconstruction.md
```

报告写清楚：

- 选择哪家建立第一条纵向链及其依据；
- cuTile与TileLang各自新增了哪些真实provider-local operations/passes；
- 哪些shared事实被两家直接复用，哪些真实差异留在local extension；
- 全语料横向接纳和失败分类；
- 删除了哪些KIR回读、side-record authority、materializer重建与fallback；
- 节点二、节点三发现并修掉的问题；
- 两家真实repro命令与数值结果；
- 剩余明确unsupported的真实性质。

节点二、节点三不单独建报告。

最后：

1. 将代码与报告整理成一个语义连贯的提交；
2. 确认工作区干净；
3. 不写版本号、迁移指南、CHANGELOG或额外进度文档。
