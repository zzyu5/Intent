# Analyses、Passes 与 Verification

## 1. 基本纪律

Pass的成立依据是它改变了当前program，并保持可验证不变量；不是文件名、pass数量或旁表字段。

每个transformation必须声明：

- 读取的KIR semantic facts与current physical facts；
- legality/preconditions；
- 改写的types、operations、regions或def-use；
- 保持的logical values、control、effects与ABI；
- invalidated/recomputed analyses；
- pass后verifier。

不允许kernel-name、op-count、whole-region template或provider字符串驱动shared policy。

## 2. Analysis 分层

### 2.1 Canonical analyses

只在immutable KIR上计算：

- shape/domain/subregion identity；
- index relations与alias；
- operation effects与collision semantics；
- control dependence、loop carry与ordered state；
- reduce/scan/region-fold/region-scan/contract schemas；
- logical-buffer lifetime；
- def-use、producer-consumer与logical reuse。

这些analysis以stable KIR IDs为key，不在physical program上重新运行一个假装canonical的KernelModel。

### 2.2 Construction facts

KIR→GPU conversion使用canonical analyses产生：

- logical worksets与execution groups；
- initial program-space mapping；
- KIR origin map；
- initial physical scalar/fragment/value/access/control graph。

Construction完成后，这些结果已经进入GPU IR；后续passes不再把construction facts当作另一份program authority。

### 2.3 Physical analyses

只从current GPU IR计算：

- program coverage与mapping；
- physical dominance、def-use与loop dependence；
- fragment axes、access footprints与validity；
- reuse、rematerialization、buffer lifetime与sharing；
- structured-op operand/accumulator flow；
- target-independent resource estimates；
- provider-form eligibility。

Physical analyses可以通过origin map查询immutable KIR semantics，但不能用KIR adjacency或shape替当前IR补出缺失的execution structure。

Analysis默认在相关IR mutation后失效；只有证明保留的analysis才可缓存。

## 3. Canonical GPU pipeline

### 3.1 Construct executable program

从KIR、canonical analyses与selected GPU capabilities建立[`kir-to-gpu.md`](kir-to-gpu.md)定义的完整initial program。Conversion同时完成types、regions、values、accesses与origin mapping，不产生等待emitter解释的holes。

### 3.2 Refine program mapping

根据worksets、access、reuse、effects与device resources改写：

- program-space rank与linearization；
- execution-group placement；
- 每个physical program instance的logical ownership extent；
- grouping/swizzle；
- grid-stride或persistent traversal eligibility/structure。

若某种persistent form已由provider lower compiler从普通program structure可靠产生，shared pass不复制其loop；它只保留下层需要的legal input。若Intent负责该结构，pass必须创建真实physical loop。

### 3.3 Form blocking 与 fragments

把scalar或较小work ownership重写为parameterized fragments与nested physical loops：

- 扩大一个physical program instance拥有的logical instances；
- 分离free、ordered、reduction与scan axes；
- 创建scalar↔fragment conversions；
- 更新所有users、access coordinates与validity；
- 保持dynamic subregion的source coordinates。

Blocking pass不得修改KIR logical subregions或作者可观察的page/window/chunk boundaries。

### 3.4 Co-realize values、accesses 与 structured operations

这三类不能由三个互不知情的旁表pass完成：fragment shape改变access footprint，access form影响materialization，contract/reduce skeleton又决定accumulator与reuse。

实现可以拆成多个passes并迭代canonicalization，但每一步必须保持完整program：

- replay/rematerialization改变physical def-use；
- bufferization产生真实allocation/lifetime/read/write；
- access realization产生coordinate/validity/fill/effect operands；
- contract/reduce/scan/region-fold/region-scan realization产生真实fragments、loops、carry与accumulator graph；region summarizer中的contract保持显式，不能依赖attention-shaped algebraic recognition恢复；
- boundary neutralization删除或简化实际validity/fill，不只删除padding record。

### 3.5 Target-capability legalization

共同GPU verifier先检查provider-neutral legality。随后selected provider/hardware capability运行local checks与被真实差异逼出的extensions：

- grid rank与static fragment requirements；
- native structured-op/input dtype支持；
- descriptor/tile-index/storage/copy/sync forms；
- compile-time parameter与resource constraints。

不能表达的组合在最早拥有足够信息的层明确拒绝，不生成慢一个数量级的伪支持或等待JIT超时。

### 3.6 Deterministic serialization

Serializer只遍历已legalized current program并发出provider source。它不得调用canonical KernelModel、按result shape反推fragment、解析role字符串、创造workspace/loop/mask/grid或追加search parameters。

## 4. Semantic-preserving rewrites

Physical IR可以与KIR op graph不同，但变化必须属于KIR semantics允许的等价实现。例如：

- 合并pure producer与consumer；
- rematerialize pure value；
- 对reduce/scan/contract采用operation允许的reassociation；
- flatten/permutation paired contract axes；
- 创建blocking loops与fragment accumulators；
- 把exact read→write关系映射成bulk transfer；
- 将program instances group、swizzle或grid-stride遍历。

不合法的变化包括：

- 把ordered recurrence改成reduce；
- 改变KIR combine、dtype、NaN/tie或atomic order；
- 修改logical subregion members；
- 改变kernel数量或引入隐藏跨kernel workspace；
- 把一个算法替换为另一个whole-operator template。

## 5. Verifiers

### 5.1 KIR verifier

验证programming-model与DSL定义的semantic legality；不检查GPU profitability或provider capability。

### 5.2 Construction verifier

验证KIR coverage、origin mapping与第一份完整physical program，规则见[`kir-to-gpu.md`](kir-to-gpu.md)。

### 5.3 Shared GPU verifier

每个shared pass后至少检查：

- program space、execution groups与ownership coverage；
- scalar/fragment shape与parameter expressions；
- structured control、dominance、loop carries与terminators；
- accesses、validity、fill、effects与alias/conflict；
- buffers、initialization/first-write obligation、lifetime与visibility；
- structured-op operands/results/accumulators；
- physical parameters均已声明并有合法domains；
- 无KIR operations、unresolved decision records或unknown executable fields。

### 5.4 Provider verifier

只检查共同program到selected surface的legality及local extension完整性，不重新决定shared structure。

### 5.5 Serializer verifier

Serialization前必须证明所有ops具有唯一provider spelling或已明确unsupported；不允许serializer fallback。

## 6. Side information 的合法范围

下面内容可以独立于executable SSA存在：

- immutable origin/provenance；
- analysis cache；
- diagnostics；
- target capabilities；
- physical parameter declarations与candidate domains；
- benchmark/tuning artifacts。

下面内容不得只存在于side records：

- program ownership与grid mapping；
- loop/range nesting；
- physical value shape/residency-required uses；
- pointer/index/validity；
- accumulator/replay/materialization；
- buffer allocation/lifetime；
- persistent/pipeline control structure；
- provider primitive operands/results。

## 7. Pass quality 的证据

一项compiler capability必须能展示：

1. pass前后真实IR差异；
2. legality与semantic-preservation条件；
3. 换一个kernel/shape后是否产生不同决定；
4. 不依赖触发kernel名称；
5. 受影响程序的数值与性能repro。

只改变attribute字符串或候选名字、却不改变当前program的改动，不构成新的physical compilation capability。
