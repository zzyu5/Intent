# 修订 05c 的 prompt

改的是 `report/prompt/05c-intentdsl-shared-gpu-correctness-closure-prompt.md`。**先调研，再改。
不改实现代码。**

现在这份 prompt 定稿于 `380221b`，此后已经产生 19 个提交和一批未提交改动。它的背景段落
描述的是当时的状态，很多已经不成立；同时讨论中又定下了两件它没写的事。所以这次不是加几句，
是**换掉已完成/已过时的背景，换上当前事实和新任务**。

---

## 一、先把背景重建到当前 HEAD

不要沿用 prompt 里的旧数字，也不要照抄执行报告的自评。逐项在当前代码上取证后重写背景段落：

- 重复 ownership 收敛做到哪了——dense GEMM 的 execution axes、旧 `FRAGMENT_D*` 参数和死 SSA
  是否已从 Physical Program 中彻底删除；
- 断言构建与 `-DNDEBUG` 构建当前各自的 corpus 结果，是否逐 kernel 一致；
- 54 个 Triton entry 当前能生成多少个 terminal source，各自的源码规模（历史上 dense GEMM 曾
  生成约 362 MB / 近 200 万行，规模异常本身就是失败）；
- 那三项规格分叉（tuple/record buffer element、一般调用前置条件、`scaled_contract` 的
  scale-axis relation）最终各自处置成什么；
- 工作区里未提交的 atomic ownership / multi-reduction 改动是什么状态。

已经做完的部分从 prompt 里删掉，不要留着让执行的人再做一遍。

同时把这一轮已经确认的事实写进背景，因为它们改变了后面几轮的判断：

- 05c 期间修掉的两处**数值错误**（atomic write 不参与 structured free-axis ownership、
  multi-reduction 用 physical extent 当 logical stop）都不是本轮引入的，都来自 `f8622d0`，
  是全量数值验证第一次把它们暴露出来的；
- 本轮自己引入过一次回归（`310449d` 把两个不同 derived occurrence 判成 ambiguous，错拒合法
  max-pool，`ff62999` 修复）——**"多加检查"不等于架构成熟**，这一条要写进 prompt 当警示；
- 一次 54-entry 扫描是在重新链接同一个 compiler 可执行文件的同时跑的，4 项 Permission denied、
  整轮混用了不同时间点的 binary。**以后任何全量扫描必须用冻结的 compiler 副本**，这条要写成
  纪律。

---

## 二、新增任务：config 必须分层，并且由 compiler 产出候选送给三家 tuner

这是这次修订的主要新内容。讨论中确认的架构是：

```
tuning table（操作类别 × dtype × device）
    → 少量完整的 shared binding 元组          BM/BN/BK/segment/ownership
    → legality filter                         用当前程序的真实形状计算，删掉非法候选
    → 各 provider 补自己的 local options       Triton: num_warps/num_stages/num_ctas
                                              cuTile / TileLang: 各自对应项
    → 各家 tuner 实测选 winner
    → winner 只存在于 runtime artifact，不写回 IR
```

**这不是新方向，是 `doc/compiler/physical-parameters.md` §4、§5 已经写下的东西**——先去读，
确认下面的判断是否与规格一致，不一致就以规格为准并说明。

当前实现偏离规格的地方有三处，写 prompt 时要分清哪些属于本轮、哪些留给后面：

**1. 默认值是一张只按 role 索引的固定表。**
`lib/Target/Triton/Transforms/Legalize.cpp:130-155` 的 `staticDefault` 只看 parameter role：
OwnershipM/N=64、Reduction=32、Warps=4、Stages=2。候选表同样写死，例如
`RealizeContractionBlocking.cpp:2552` 的 `{64, 128}`。**一个 pointwise 和一个 4096×14336×4096
的 GEMM 拿到同样的值。**

维护一张 tuning table 本身是正当的——LLVM 的 `TargetTransformInfo`、GCC 的 `-mtune` 都是这么
做的，**不要把它改成运行时反馈或 cost model**。问题只是这张表缺索引维度：至少要按操作类别
（contraction / reduction / pointwise / scan）、元素宽度、目标设备索引。**查表不是推理，工作量
和形态都不同，写 prompt 时不要把它描述成要建一个推理引擎。**

**2. legality filter 完全不存在。**候选值有没有超 shared memory、寄存器够不够、grid 合不合法，
从来没算过。**这是整套里唯一必须计算、不能查表的部分**，因为它依赖当前程序的真实形状——同样
的 `BLOCK_M=128, BLOCK_K=64`，fp16 装得下、fp32 装不下。

它的缺席有具体后果：非法候选被送进 tuner，tuner 只能靠编译失败发现。cuTile 那 14+14 个
`worker_timeout` 很可能就是这么来的。规格明写 Intent 负责删除可证明非法的候选，但不复制下层
完整的 resource allocator——边界在这里。

**3. config 的两层被压成了一层。**C++ 侧的 `ParameterOp`、`OwnershipM/N`、Reduction 角色已经是
provider-neutral 的，但 Python 把共享 binding 命名成 `TritonParameterBinding`，再和
`num_warps/num_stages/num_ctas` 一起装进 `TritonConfig`。**shared binding 是三家共用的，local
options 是各家自己的，这两层必须分开**，否则 cuTile/TileLang 无法消费同一组 shared binding。

**次序判断需要你在调研后给出，并写进修订后的 prompt**：第 3 项（分层）是第六、七轮的前提，
不是第八轮；第 2 项（legality filter）是第八轮性能的前提，因为没有它测到的是"哪些候选没编译
失败"而不是性能。**它们要不要放进 05c，取决于 05c 正确性收口还剩多少工作**——把判断和依据
写清楚，不要含糊成"后续处理"。

---

## 三、保留不动的

以下是 05c 现有 prompt 里仍然成立的，不要删：

- ownership 必须被精化而不是重复建立；
- `analysis unknown` 不是 `program illegal`；
- 不建设 shape/device 驱动的动态 candidate generator，不预测 winner，不把候选数量当创新；
- 不写死一条"legality 拒绝 / optimization 退化"的分界规则——**去 `ref/triton` 和
  `ref/tilelang` 看这条线在那里怎么划**；已核实的参照：TileLang 有 reject-only 的 semantic
  checks，Triton 的 `Alias` 证不出来返回 `MayAlias`、`BufferIndexAnalysis` 退保守路径；
- 那三条不能省的验证：74-entry 的 cuTile/TileLang 纯编译 probe、54-entry 数值正确性、
  verifier 12 条不变量逐条处置（已检查 / 明确不检查且说明为什么安全，不能留在"没说"）；
- 边界：不留第二条路径、不加兼容分支/默认值/fallback、不按 kernel 名或形状特征分支、不建
  test 目录/pytest/fixture、不写版本号/CHANGELOG/迁移指南；
- 自查是生成式的 ref 对照，给两边 file:line、差别和实际后果，不写编号清单和"X → 1"自评表。

---

## 四、产出

修订后的 `05c-*.md` 覆盖原文件。同时给一段简短说明：背景改了哪些、依据是什么、新增的 config
分层任务落在本轮还是后续、以及为什么。

不改实现代码。调研可以读、可以跑现有生产路径的编译取证；不跑 benchmark、不更新 baseline CSV。
需要遍历 corpus 就写一次性命令，跑完删掉。
