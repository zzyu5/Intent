# 第 5c 轮：闭合 shared GPU 正确性与 config 分层地基

这一轮从当前工作树继续，完成第五轮最后的 shared correctness 收口，并建立三家 provider
共同消费的 config 分层地基。它不再重复已经完成的 ownership 清理、规格分叉调查或历史
build-mode 调查；性能只收口默认 config 与结构质量，不进入公平候选和最终 1.05× 闭环。

本轮结束时应当成立的是：

- 当前全部 corpus 在 shared construction/verifier 上得到合法、完整、与构建模式无关的结果；
- 54 个 Triton entry 都能生成有界的 terminal source，在两台 GPU 上完成 generated 数值验证，
  并以静态默认 config 进入 source 实际耗时的 1.1× 以内；
- cuTile 与 TileLang 的 74 个 registry entry 都经过 shared/provider-stage 纯编译取证，shared
  缺口与 provider-local 缺口不混在一起；
- shared physical binding 与 provider-local tuning options 已经分层，三家 tuner 接收同一组
  shared binding tuples；
- shared GPU 规格中的十二条完整性不变量都有明确执行归属，没有留在“规格写了、实现没说”。

本轮更新两张 Triton 表，但不做公平候选集对齐、source 侧 autotune 配置匹配或 winner 选择。

---

## 一、开始前阅读与取证方法

从 doc/index.md 进入，完整阅读本轮直接涉及的规格：

    doc/compiler/README.md
    doc/compiler/kir-to-gpu.md
    doc/compiler/gpu-program-ir.md
    doc/compiler/passes-and-analyses.md
    doc/compiler/physical-parameters.md
    doc/compiler/target-lowering.md

同时完整阅读：

    AGENTS.md
    report/current-gpu-compiler-state-and-next-round.md
    report/shared-gpu-compiler-investigation-and-round-recut.md
    report/shared-gpu-reconstruction-completion-audit.md
    report/shared-gpu-analysis-pass-reconstruction.md

历史报告只说明当时观察到什么，不是当前事实。结构判断必须先对照：

    /home/kingdom/phdworks/ref/triton
    /home/kingdom/phdworks/ref/tilelang

ref 对照不是收尾报告里的装饰。每当需要决定 initial physical representation、analysis
exact/unknown、legality、parameter binding、provider boundary 或 verifier 责任时，先找 ref
中最接近的真实实现，说明它如何承载、Intent 当前差在哪里、换一个 kernel 或去掉一项前提后
差别有什么实际后果，再决定实现。

Intent 作者没有写 block shape，automatic blocking 是 Intent 自己的责任。ref 中没有同名 pass
不等于这项责任不存在；应参考的是成熟编译器怎样建立 typed default、组织 analysis、表达
unknown、改写 current IR、验证 legality 和把 provider-local 选择交给下层，而不是照搬模块名或
target surface。

---

## 二、修订时的当前事实：不要把已完成事项重新写成任务

以下事实已经通过 production compile 路径重新取证。实现继续变化后，只重跑受影响的证据；
不要引用更早的历史数字，也不要机械重复已经关闭的调查。

### 2.1 Dense GEMM 的重复 ownership 已经收口

当前 dense GEMM 的 realized Physical Program：

- 只有一个 intent_gpu.delinearize execution mapping，axes 是 contraction M/N；
- parameter 只有 GROUP_SIZE_M、BLOCK_K、BLOCK_N、BLOCK_M；
- 不含旧 FRAGMENT_D* parameter、type/range 引用或死 SSA；
- terminal source 为约 3.5 KB、97 行、一个 triton.Config。

因此历史上的约 362 MB、近 200 万行 dense GEMM source 已经消失，不能继续写成待修问题。
实现依据位于：

    lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp:2551-2570
    lib/Dialect/GPU/Transforms/RealizeContractionBlocking.cpp:2624-2651
    lib/Dialect/GPU/Transforms/Passes.cpp:93-103
    lib/Dialect/GPU/Transforms/Utilities.cpp:2777-2830

通用 pointwise realization 仍会在真正由 pointwise ownership 拥有的维度上建立
FRAGMENT_D*。这不是残留本身；要求是后续 structured refinement 必须精化同一 execution
axes，并删除被替代的 parameter/use，不能并存第二份 authority。

“ownership 必须被精化而不是重复建立”继续作为本轮不变量：

- 保守 pass 可以先形成一份合法 ownership；
- 后续 structured pass 依据 typed relation 精化已有 mapping/parameter；
- replacement 后旧 parameter、physical expression、range/type use、dead SSA 和 config dimension
  必须消失；
- 不能靠 serializer 隐藏、artifact 后处理或宽泛 DCE 掩盖仍被 current IR 使用的旧决定；
- pointwise、reduction、scan、region fold/scan、ordinary/scaled contraction 对同一 axis 只能有
  一份 authoritative physical decision。

不要重新实现 dense ownership，也不要围绕这个已关闭问题重构 family 文件。

### 2.2 构建模式已经一致，但 shared correctness 尚未闭合

完整 examples/kernels/ 为 93 个文件、217 个 @intent.kernel。分别构建 assertion-enabled 与
RelWithDebInfo -DNDEBUG compiler 后，两种构建逐 kernel 结果完全一致：

    180 / 217 passed
    37 / 217 failed

37 个都停在 KIR-to-GPU physical-program construction，诊断也逐 kernel 一致。旧的
assertion/release divergence 已经关闭；新的事实是仍有 37 个 shared construction gap，主要表现
为 broadcast physical-axis projection、binary shape/ownership、cast/reshape relation、dynamic
range coordinate、reduction free-axis coverage、structured control/scan relation 等。

这 37 个属于本轮主线。不得为了得到 217/217 在失败 family 里继续加更窄 matcher；先把每项归到
缺失的 relation/analysis/decision authority，再由统一 authority 关闭。

### 2.3 Triton terminal source 已经基本恢复且规模有界

Triton registry 有 54 个 entry、62 个 generated components。纯编译结果为：

    61 / 62 components 到达 terminal source
    53 / 54 entries 的全部 components 到达 terminal source

唯一未到 terminal source 的是：

    legacy_flash_attention_bias[0]
    → flash_attention_bias_fwd
    → physical_program
    → broadcast physical axis projection is unknown

成功源码最大约 19 KB / 421 行，不再有源码爆炸。当前没有 Triton provider legalization 或
terminal serialization failure。这里仍缺的是上述 shared broadcast relation，不是另一个
Triton leaf 补丁。

### 2.4 三项规格分叉已经闭合

以下三项不再调查或改设计：

- logical buffer element 只支持 ranked tensor 的单一 scalar element type；tuple/record 不进入
  buffer；
- public precondition 只保留 closed typed 的 assume_in_bounds 与 view alias/noalias，不存在
  任意布尔 assume 或 optimization hint；
- scaled_contract 采用 closed positional scale-axis schema，frontend 与 canonical verifier
  精确检查 rank、axis position、group、carrier extent 和 scale shape，不再由 shared pass 从
  rank/shape 反推第二份作者语义。

对应当前规格和实现依据包括：

    doc/dsl/types-numerics-and-effects.md:19
    doc/dsl/types-numerics-and-effects.md:129
    doc/dsl/types-numerics-and-effects.md:232-238
    doc/dsl/core.md:81-89
    doc/dsl/core.md:288-295
    python/intent/frontend/lowering/intrinsics/structured.py:617-654
    lib/Dialect/Intent/IR/IntentDialect.cpp:70-75
    lib/Dialect/Intent/IR/IntentOps.cpp:1182-1223
    lib/Dialect/Intent/IR/IntentOps.cpp:1656-1662
    lib/Dialect/Intent/IR/IntentOps.cpp:1696-1725

### 2.5 当前未提交的两处 correctness 修复尚未收尾

工作区中的两处实现改动分别处理：

- structured free-axis ownership 过去只观察普通 StoreOp，没有把 atomic store/RMW/CAS 与
  scatter-reduce 等写 effect 纳入统一 write-coordinate analysis；
- multi-reduction physicalization 过去用 physical tile extent 计算 logical loop stop，而不是
  使用 lockstep logical range end。

这两处 bug 都来自 f8622d0，不是 05c 新引入的；它们直到全量 generated 数值验证才暴露。
当前修改方向正确，但尚未在冻结最终 binary 上完成 atomic-only write、multi-reduction 数值以及
跨 provider shared probe，因此仍是本轮正式工作。

修复必须落在统一 effect/ownership authority 与 source-range/lockstep authority，不能只在触发
问题的 op 或 kernel 旁补一条判断。

### 2.6 本轮曾自行引入过一次错误拒绝

310449d 把同一遍历下两个不同 derived occurrences 判成 ambiguous，错误拒绝合法 max-pool；
ff62999 通过 typed PhysicalLockstepTraversalFact 修复。

这条事实必须约束本轮：增加 verifier、ambiguity check 或 fail-closed 分支本身不代表架构成熟。
analysis unknown 不能被顺手解释为 program illegal；每一处拒绝都要由当前语义/physical
invariant 和 ref 中同类 legality 行为支撑。

### 2.7 全量取证必须冻结 compiler

以前有一次 54-entry 扫描在同一个 compiler 可执行文件被重新链接时运行，出现 4 项
Permission denied，且整轮混用了不同时间点的 binary。该结果无效。

此后任何 corpus/registry 全量扫描都必须：

1. 完成构建；
2. 将 compiler 复制到本轮独占、不会被重新链接的冻结路径；
3. 整轮只使用该副本；
4. 构建或修改发生后废弃旧结果，重新冻结新的副本再运行；
5. 报告记录源码状态、构建模式、冻结 binary 路径和命令，不使用 hash/checksum。

不得一边 relink 一边运行全量，也不得把不同 binary 的结果合并。

---

## 三、执行顺序：先闭合 authority，再关闭 coverage，再建立 config 分层

这一轮按以下顺序推进：

1. 完成并验证当前 atomic ownership 与 multi-reduction logical-stop 两处修复；
2. 从缺失的 shared relation/analysis/decision authority 出发关闭剩余 37 个 construction gap；
3. 重新确认 assertion 与 -DNDEBUG 构建的完整 corpus 结果逐 kernel 一致；
4. 让 54 个 Triton entry 的全部 62 个 components 到达有界 terminal source；
5. 拆分 provider-neutral shared binding 与 provider-local tuning options，并让三家 target
   invocation 能消费同一组 shared tuples；
6. 建立 physical liveness/interference 分析，并用它对 table 返回的完整候选集做
   resource legality filter；删除可证明超出 shared-memory、register、grid 的候选；
7. 最后执行 74-entry compile-only probe、54-entry Triton generated 数值与默认 config 性能、
   以及十二条不变量归属核查。

第 6 步的输入只有在第 2 步关闭 37 个 construction gap、第 4 步使 terminal source 有界以后才
稳定，因此不能与它们并行；但它是第六、七轮接入真实 provider tuner 以前的必要 compiler
能力，必须在本轮完成。

本轮不实现 cuTile/TileLang 的 provider forms，也不做公平候选集对齐与 source 侧 autotune
配置匹配——那是第六、七、八轮。

Coverage 是 authority 重构的结果，不是与重构并行、反过来定义重构的任务。遇到新的 coverage
失败，先归类到缺哪个 analysis、decision 或 provider-local form；不得在失败点直接加
direct-load、rank、shape、op adjacency 或 kernel-family matcher。

---

## 四、闭合剩余 shared correctness

### 4.1 Write effect 必须进入同一 ownership 分析

ownership 分析不能把普通 store 当成唯一写 effect。它必须从 current GPU IR 的 typed effect 与
coordinates 出发，统一覆盖普通 store、atomic store、atomic RMW、compare-exchange、
scatter-reduce 以及当前 IR 中其它真实写 effect。

要回答并落实：

- authoritative write-coordinate/effect fact 在哪里；
- structured free-axis ownership 怎样消费它；
- direct use 与 derived coordinate relation 怎样区分；
- mutation 后怎样失效/重算；
- ordinary 与 atomic 路径是否读取同一事实；
- analysis unknown 时保留什么合法 program，何时才是 semantic conflict。

不要给 atomic op 单独建立第二份 ownership 推导。

### 4.2 Multi-source/multi-reduction 必须使用 logical relation

physical tile extent 不能代替 logical traversal stop。multi-source/multi-reduction、
region fold/scan 等需要共同遍历的结构，必须消费统一的 lockstep/source-range authority：

- logical start/end/step 来自被保留的 source relation；
- physical extent 只决定 chunk granularity；
- 多 source 必须证明 compatible/lockstep，证不出来返回明确 unknown 或准确失败；
- 不能默取第一 source，也不能用 start + physical_extent 截断 logical domain；
- derived occurrences 共享同一 traversal 不等于它们是同一个 value occurrence。

### 4.3 37 个 construction gap 要按缺失 authority 关闭

当前 37 个失败不是 37 条局部规则。至少从以下共享问题重新归类：

- physical axis projection 与 broadcast/cast/reshape relation；
- pointwise operand/result shape 与 ownership relation；
- dynamic logical range 到 physical tile coordinate 的连接；
- reduction free-axis 的 full-coverage/ownership authority；
- control-flow record、scan carry 与 structured result relation；
- range provenance、validity 和 effect coverage 在 rewrite 后是否仍完整。

每类修复应让多个 family consumer 读取同一结果。family pass 可以保留各自 accumulator/carry 或
mutation pattern，但不能从 source ID、rank、shape、op 邻接重新推一份同名事实。

### 4.4 Unknown 与 illegal 的边界必须从真实 ref 行为判断

不要预先写一条全局“unknown 一律拒绝”或“unknown 一律退化”的规则。对每类分析看 ref 的同类：

- Triton Alias 证不出来会保守为 MayAlias：
  /home/kingdom/phdworks/ref/triton/lib/Analysis/Alias.cpp:55-69；
- Triton BufferIndexAnalysis 证明失败时不匹配依赖该证明的变换：
  /home/kingdom/phdworks/ref/triton/lib/Analysis/BufferIndexAnalysis.cpp:122-139；
- TileLang 对 nested parallel、parallel race 等真实 semantic illegality 直接拒绝：
  /home/kingdom/phdworks/ref/tilelang/tilelang/analysis/nested_loop_checker.py:34-62 和
  parallel_local_index_checker.py:42-57。

Intent 中：

- semantic/type/SSA/closed schema 不成立时给 typed diagnostic；
- optimization fact unknown 时保留一份完整、合法、较保守的 current program；
- provider/hardware legality 在 shared 层无法决定时，通过明确 carrier 交给 provider；
- 不用另一套 shape matcher 猜答案，不把 unknown 解释为第一 source、默认 range 或任意 block 常数；
- mutation 改变 current IR 后，相关 analysis 必须失效或重算。

具体边界必须给双方 file:line 和换一个 kernel 后的实际后果，不能靠上述例子机械套用。

### 4.5 Construction 返回时就必须合法完整

lowerCanonicalKIRToGPU 返回的 initial/current Physical Program 在每个 pass 边界都必须完整：

- runtime logical extent 不进入 compile-time fragment shape；
- program space、coordinates、range、validity、effect 与 fragment granularity 显式连接；
- assertion 开关不改变 type/operation legality 或诊断；
- pass 不能靠 provider 以后补洞；
- verifier 不能以忽略 stale carrier、side record 或未消费 parameter 让程序通过。

不能用慢一两个数量级的串行 fallback 冒充完整，也不能通过放宽 verifier 解决 construction
缺失。

---

## 五、Config 分层：本轮建立三家共享的结构地基

doc/compiler/physical-parameters.md §4、§5 已经规定：

    tuning table（操作类别 × dtype/元素宽度 × target tune profile）
        → 少量完整、相关联的 shared binding tuples
          （例如 BM/BN/BK/segment/ownership）
        → 可证明 legality
        → provider 补自己的 local options
          Triton: num_warps/num_stages/num_ctas
          cuTile / TileLang: 各自真实支持的 local options
        → 各家 tuner 实测选择 winner
        → winner 只进入 runtime artifact/cache，不写回 KIR 或 shared GPU IR

这不是 runtime feedback、cost model 或动态 candidate generator。Tuning table 是静态 target
tuning data；查表不等于推理，winner 也不是 Intent compiler 的创新。

### 5.1 当前偏差

当前实现有三处明确偏差：

1. include/Intent/Dialect/GPU/IR/Program.h:65-78 的 ParameterRole 同时包含 shared 与 provider
   roles；lib/Target/Triton/Transforms/Legalize.cpp:130-170 的默认值只按 parameter role 索引；
   pointwise、reduction 和大 GEMM 可能取得同样的 ownership/reduction 值。若干 family pass也各自
   保存固定 candidate 列表；
2. python/intent/targets/triton.py:10-79 把 shared binding 命名为 TritonParameterBinding，并与
   num_warps/num_stages/num_ctas 一起压进 TritonConfig；cuTile/TileLang 没有可消费同一 shared
   tuple 的对应层；
3. 当前只有少量 typed/domain/provider-form legality，没有基于 current shape、dtype 和已知设备
   资源的统一 pre-tuner filter；非法 tuple 可能只能靠 provider 编译失败被发现。

ref 中已经查实的对应行为是：

- Triton autotuner 接收显式完整 Config 列表，early prune 后逐项编译/benchmark，并由实测结果
  选 winner：
  /home/kingdom/phdworks/ref/triton/python/triton/runtime/autotuner.py:21-35、
  217-251、284-313、328-406；
- TileLang 每个 config 独立 compile，异常和 timeout 只使该 config 失败，成功项再实测：
  /home/kingdom/phdworks/ref/tilelang/tilelang/autotuner/tuner.py:44-51、511-521、
  612-628、715-870；
- TileLang 的 static policy 先产生 tile candidates，再做 typed shape filter：
  /home/kingdom/phdworks/ref/tilelang/tilelang/carver/roller/policy/default.py:72-94。

这些证据支持“完整 tuple + provider-local options + provider tuner”的分层，不支持把 shared
parameter domains 做任意笛卡尔积，也不支持把 provider compile 能决定的全部资源分配复制到
Intent。

### 5.2 本轮要完成的分层

05c 必须完成第 1、2 项的结构地基，因为它是第六、七轮接 cuTile/TileLang provider-local forms
的前提：

- 建立 provider-neutral 的 shared parameter binding/complete tuple 表示，不再使用
  TritonParameterBinding 承载三家共同语义；
- shared tuple 只绑定 current Physical Program 中已经存在的 typed physical parameters；不能从
  kernel 名称、shape 特征或 source graph 重新发明 axes/topology；
- Triton config 由“shared tuple + Triton-local options”组成；cuTile/TileLang target invocation
  也接收同一种 shared tuple，再由各自 provider 层补 local options；
- 三家使用同一 shared tuple identity 与 binding 规则，不要求三家的 local option schema 对称；
- static tuning table 至少按 typed operation category
  （contraction/reduction/pointwise/scan）、element dtype/width 和稳定的 target tune
  profile/capability 索引；
- table 返回少量完整、相关联的 shared tuples，不把每个 parameter domain 做笛卡尔积；
- 普通非 autotune 调用也从同一 table 中选择一份明确默认 tuple，不能另建第二条默认/fallback 路径；
- complete tuple 在 terminal serialization 之前实例化，所有 required shared parameters 必须绑定；
- 删除 artifact 生成后的 repro adapter 过滤等正式路径；adapter 不能在 terminal source 爆炸后再挑
  candidate；
- tuner 只消费 compiler 给出的完整候选并实测 winner；winner 不写回 IR。

这项工作不是为 cuTile/TileLang 实现 provider forms，也不是在 shared pass 里加入 target-specific
机制。它只建立 shared candidate 与 provider-local option 之间的唯一边界。

### 5.3 Resource legality filter 本轮完成，第八轮只剩测量

Filter 在本轮完成，不再推迟。原有排序与第六、七轮矛盾：那两轮要接 cuTile/TileLang 的真实
tuner，接上就必须提供候选；候选未过滤时，tuner 只能靠逐个编译失败发现非法项。这正是已经出现
过的 cuTile 14+14 个 worker_timeout 的形态，也会把资源问题误记成 provider gap。

前置是正式的 physical liveness/interference 分析。当前
lib/Dialect/GPU/Analysis/PhysicalProgram.cpp:1349-1388 只证明 first-write 与 full
initialization，没有 live interval、interference 或 offset。对照
/home/kingdom/phdworks/ref/triton/lib/Analysis/Allocation.cpp:175-201，Intent 当前 IR 已经具备
增量建立该分析所需的基础事实：BufferType 带 scope、instance、owner、lifetime、visibility，
FragmentType 带 owner，PhysicalProgramAnalysis 已有 footprint 与 bufferDataflow；这不是从零建立
另一套 allocator。

必须先有 liveness，再上线 resource filter。没有 liveness 时只能把 kernel 内全部 fragment 的资源
需求相加，必然高估。对默认 tuple 高估只会选到更小但仍合法的 tuple，结果可能慢但正确；对候选集
高估则会静默删除本来合法的 candidate，tuner 永远看不到它，而且没有任何失败信号。因此 filter
不得在 liveness/interference 之前以粗略总和实现。

默认路径同样从 tuning table 返回的完整 tuples 中选择第一份通过该 filter 的 tuple，不再无条件
绑定固定值。全部候选都被证明非法时给出明确 typed diagnostic，不能静默回退到固定值、第一项或
另一条 config path。

边界保持不变：

- Intent 只删除可证明非法的候选，不复制下层完整 resource allocator；
- layout、register allocation、pipeline 等 Intent 无法预知的失败，仍由 provider compile 使对应
  candidate 失效；
- exact/unknown 的边界逐项对照 ref，unknown 不得伪装成 illegal；
- filter 消费已经分层的完整 shared tuple、current Physical Program、dtype/element width、稳定
  target capability 和 provider 已声明的 local form，不反向改变 KIR、program structure 或算法；
- resource 结果用于 candidate legality，不预测 winner，也不写回 canonical KIR/shared GPU IR。

第八轮因此只剩测量：接三家 tuner、公平候选关系、两机全量和 1.05× 性能闭环。本轮结束后不再
留下“到时候才补”的 compiler capability。

---

## 六、三项不能省的最终验证

验证服务于本轮结构，不用旧数字制造门禁。但以下三项是证明 shared correctness 确实闭合的外部
证据，不能省略。

### 6.1 74-entry cuTile/TileLang 纯编译 probe

按执行时当前 registry 枚举 cuTile 37 + TileLang 37 entries，所有 generated components 走：

    frontend
    → canonical KIR
    → KIR-to-GPU construction
    → 全部 shared passes
    → full shared verifier
    → 对应 provider legalization/verification（只用于阶段分类）

本轮要求：

- 74 entries 接收到的 shared program 全部合法、完整、build-mode independent；
- shared representation、ownership、access、buffer、structured flow 缺口必须在 05c 修；
- 后续失败必须准确属于 cuTile/TileLang provider-local form/capability；
- 若 provider 必须从 KIR、shape 或 op 邻接重建 shared fact，仍是 05c 边界缺口；
- 不实现 cuTile/TileLang provider forms，不跑它们的 GPU、timeout 或性能。

需要批量遍历时用一次性脚本调用 production frontend/toolchain API，运行后删除，不恢复长期
inventory runner。

### 6.2 54-entry Triton generated 数值正确性与默认 config 性能

shared、config 分层和 resource legality 完成后，让 54 个 entry 的全部 components 到达有界
terminal source，并在两台机器上并行运行，不让任一台空置。同一次运行同时判定：

- generated artifact 能完成 Triton compile/JIT、launch 和 numerical comparison；
- generated 使用 tuning table 给出的静态默认 tuple，以 source 的实际耗时为分母；两台机器上
  全部 Triton entry 都必须进入 1.1× 以内。

执行链为：

    terminal source
    → Triton compile/JIT
    → launch
    → numerical comparison
    → generated 默认 config 与 source 实际耗时比较

多 kernel entry 按 registry 定义的完整 Python orchestration 执行。54 个 entry 都必须有真实
generated 数值结果、source 实际耗时和 ratio；source 侧 resource/compatibility、adapter 或
measurement gap 不能被记为通过，也不能用 PyTorch composition 新造性能 baseline。

这不是公平候选比较，也不在本轮做公平化：source 运行它自己的实际实现和 autotune winner，
generated 固定使用 tuning table 选出的默认 tuple。这个不对称是有意保留的——在 generated
没有搜索 winner 的情况下仍进入 1.1×，才能说明 Physical Program 结构与 tuning table 默认值
同时成立。

任何一台机器上超过 1.1× 的 entry 都必须逐个归到以下两类之一，并在本轮修完：

- tuning table 质量问题：该 operation category、dtype/element width、target tune profile 对应的
  默认 tuple 明显偏离手写 source 使用的物理量级；修正 typed table policy，不为 entry 开特例；
- 结构问题：generated 与 source 的 physical program 形态不同，例如多余 loop/materialization、
  缺失 native primitive、错误 ownership/traversal 或 provider form；从缺失的
  analysis/decision authority 修 shared/provider 结构，不在 serializer 或 entry 名旁打补丁。

两台机器并行更新：

    report/baselinev2/triton-5090.csv
    report/baselinev2/triton-h100.csv

写入真实运行结果，保持现有表结构。cuTile/TileLang 四张表不动。明显异常的单项独立复测一次
确认，不反复运行取最优。

本轮不做公平候选集对齐、source 侧 autotune config 匹配、generated winner 选择，也不追 1.1×
以内的进一步压缩；这些属于第八轮。

### 6.3 十二条 shared 完整性不变量

用 doc/compiler/gpu-program-ir.md 的十二条完整性不变量审视最终实现。每一条必须落到以下两种
结论之一：

1. 当前阶段由明确 verifier/analysis 检查，给出 current IR carrier 和 file:line；
2. 当前阶段明确不检查，给出 ref 依据、后续 owner 和提前检查会错误拒绝什么合法 program。

这不是要求机械增加十二个 verifier，也不写“X → 1”自评表。没有语料和 ref 依据的能力不投机新增；
但不能把任何一条留在“没说”。本轮直接触及的 program mapping、fragment extent、range/rewrite、
effect coverage、parameter binding 与禁止第二份 side decision 必须真实闭合。

### 6.4 构建模式与冻结执行纪律

完成实现后分别使用 assertion-enabled 与 RelWithDebInfo -DNDEBUG 构建，对完整当前 corpus 逐 kernel
运行同一 production compiler 边界。二者必须：

- 成功/失败 stage 和 typed diagnostic 类别逐 kernel 一致；
- 不出现 assertion、abort 或 build-mode-only acceptance；
- construction 和 whole-program verifier 对 fragment/runtime extent 执行同一不变量。

每次全量都必须使用构建完成后复制出的冻结 compiler；不得在 relink 同一路径时运行。

---

## 七、主要改动完成后的生成式 ref 自查

主要实现完成后先停止补 entry。从本轮实际新增、修改、删除的 authority 和 config carrier 出发，
在 ref/triton 与 ref/tilelang 寻找同类实现。至少回答：

- provider-local grid/mapping 改写怎样保持完整 program carrier；
- predicate/validity 在 blocking rewrite 中怎样保持或证明可丢；
- multi-source traversal 的 logical bound 怎样确定；
- default physical representation 与 analysis unknown 怎样进入保守合法路径；
- complete config tuples、provider-local options、early pruning 和 winner 分别由哪层拥有。

每处给出：

- Intent 与 ref 双方 file:line；
- IR carrier、analysis result、unknown/legality、mutation 与 verifier 责任；
- 具体差别；
- 换另一个真实 kernel、增加一个 source、改变 source graph 或去掉一项 matcher 前提后的实际后果。

不要拿预写清单让实现自证通过；也不要写“与成熟实践一致”“泛化守住了”“emitter 变薄了”这类
无外部证据的结论。

随后沿唯一 executable path 回溯，删除被替代的 duplicate inference、旧 helper、宽松 no-op、
compatibility/fallback、临时 attr/side record、artifact 后过滤和本轮取证产物。不恢复旧 Plan、
厚 materializer 或第二条执行路径。

---

## 八、报告、AUDIT 与提交

产出：

    report/shared-gpu-and-triton-closure.md

报告只写当前事实：

- atomic/write-effect ownership 与 multi-reduction logical stop 最终由哪一 authority 闭合；
- 37 个 shared construction gap 关闭后形成了哪些统一 relation/analysis，而不是增加了哪些 matcher；
- assertion 与 -DNDEBUG 完整 corpus 在冻结 binary 上的逐 kernel 结果；
- 54-entry/62-component Triton terminal-source 规模与所有非通过项的准确 stage；
- 74-entry cuTile/TileLang shared/provider-stage 纯编译分类；
- 两台 GPU 上的 54-entry generated numerical、默认 config 性能、1.1× 内外状态和两张 Triton 表；
- provider-neutral shared tuple、static tuning table 与三家 provider-local options 怎样分层；
- 十二条完整性不变量的真实执行归属；
- 本轮 ref 双向对照的双方 file:line、差别与实际后果；
- 删除的第二份推导、旧路径、artifact 后过滤与临时产物；
- 是否仍有会迫使第六、七、八轮回头修改 shared correctness/config boundary 的问题。

主要实现和三项验证完成后，再执行 report/prompt/AUDIT-5c-state.md 核查最终工作树。AUDIT 不重新
调查已经关闭的历史事项，也不要求为了得到旧数字重复运行。

按语义完整节点提交，不把半条旧/新 execution 或 config path 留在提交边界。最后删除一次性脚本、
日志、cache 和生成源码；整理实现、必要规格修正和报告，确认工作区只保留用户原有的无关改动。
