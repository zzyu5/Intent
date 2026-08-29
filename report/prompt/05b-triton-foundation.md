# 第五轮补完：把 Triton 路径修成第六轮的地基

第五轮建立了 shared analysis authority，也收拢了一部分事实推导；但核查
（`report/shared-gpu-reconstruction-completion-audit.md`）证明它同时留下了一处自己引入的
回归和两处会改变数值语义的缺口。这一轮把这些关掉，让 Triton 路径重新成为可信的地基。

**不做 cuTile、不做 TileLang、不追 `1.05×`。**

---

## 一、根因不是"ProgramGrid 有个 bug"，是 provider transform 破坏了 shared 不变量

现象已经复现：

```
addcmul_broadcast_bf16   shared=OK   triton=provider_program_verification
                                     "physical kernel program mapping/effect coverage is incomplete"
gelu_tanh                shared=OK   triton=同上
ragged_grouped_gemm      shared=OK   triton=OK
```

机制：`lib/Target/Triton/Transforms/ProgramGrid.cpp:126-128` 在替换 coordinate uses 之后
`mapping.erase()`，删掉了唯一的 execution-group carrier；`Legalize.cpp:886-890` 紧接着再跑
一次 `gpu::verifyGPUProgram`，于是几乎所有普通 kernel 当场被拒。

那个 mapping/effect coverage 检查是第五轮 `a246c00` 加强的。**加强 verifier 是对的**——
错的是 provider legalization 没有一种在改写后仍然承载 execution group 的 program 形态。

所以这一轮要修的性质是：**provider transform 必须把一份完整 program 改写成另一份完整
program，不能靠删掉共享层的承载物来达到目标形状。**具体怎么承载（保留 mapping 并让
Triton 消费、还是定义一种 provider-local 的等价 carrier）由 ref 对照和当前 IR 不变量决定；
不要在 verifier 上开例外，也不要把它降级成 warning。

ProgramGrid 只是被抓到的那一个。沿三家 provider 的 legalization/serialization 走一遍，
凡是删除、替换或绕过共享层承载物（mapping、range、validity、effect origin、parameter
binding、access footprint）的地方，都按同一条性质处理。这一轮只修 Triton 那条链上的，
cuTile/TileLang 的记录下来留给第六、七轮。

---

## 二、两处会改变数值语义的 shared 缺口，这一轮修掉

它们能穿过 217/217，因为 verifier 只看重写后的程序自洽，看不出原谓词被丢了。留到第六轮
就会变成"provider 报数值错，查半天发现在 shared"。

**Scaled contraction 丢失原 validity。**`RealizeContractionBlocking.cpp:2160-2172` 的
acceptance predicate 接受三类 validity（无、scalar、可识别 tail predicate），但 `2429-2456`
只重建 row/block/column tail mask，`2497-2510` 建新 load、`2527-2555` 建 store 时都没有
replay 原 scalar/residual validity，也没有与新 tail mask 相与。**接受的语义比能保持的语义
宽**，可能读写作者谓词已经排除的位置。

核查明确指出：不要在 scaled family 里局部补一个 `and`。scalar validity 怎样 broadcast 到
data/scale/result 各自的 fragment、哪些谓词可以合法 replay，应该由已经建立的统一
predicate/replay authority 定义。修在那里，让 scaled 和 ordinary 两条路都消费同一份答案。

**Region fold/scan 的单-master physicalization。**`RealizeRegionFold.cpp:1456-1461` 无条件
用 `plans.front().ranges.front()` 决定 stop；`1637-1662` 同样拿第一个 source 的 extent 当
循环上界。没有检查其余 source 的 start/extent/step 与 master 一致。

canonical verifier 证明过 logical extents lockstep，**但那不等于 family rewrite 之后的
physical source ranges 仍然相同**。要么证明一致（并让这个证明成为 IR 上可验证的事实），
要么在不一致时明确失败，不能默默用第一个。

---

## 三、验证：把编译器对全语料跑一遍，一直跑到 Triton 终端源码

第五轮只跑到 shared verifier，所以这个回归穿过去了。这一轮把同一条生产链往前走一段：

```
frontend → canonical KIR → construction → shared passes → Triton legalization → terminal source
```

用现有的 `intent.compiler.inventory` / `toolchain.run_compiler` 路径，纯编译、不跑 GPU、
不建测试设施。报告里给出：多少 kernel 到达终端源码、其余停在哪个阶段、每个阶段的原因分类。

**不要把这个数字当成能力指标去凑。**217 个 kernel 里有大量结构本来就不在 Triton registry
里，停在真实 provider capability 上是合理结果。要区分的是三类：共享层承载物被破坏、
provider 能力真的不支持、provider physicalization 还没做。第一类这一轮修，后两类记录。

---

## 四、重新建立 Triton 的真实数字

`report/baselinev2/triton-*.csv` 是第五轮之前跑的，在当前 HEAD 上**已经不成立**。修完之后
把 54 个 Triton entry 在两台机器上真实跑一遍，重新写这两张表。

这一轮**不是**追性能：目的是确认第五轮那一万行没有留下退化。判断方式是跟旧表逐项比：

- 原来 pass 现在不 pass 的 —— 这是第五轮的损伤，**这一轮修**；
- ratio 明显变差的（比如从 1.0 附近退到 1.3 以上）—— 同样是损伤，这一轮修；
- 原来就超过 1.05 的那 21 项 —— **不动**，留给第八轮。

跑得快、数据差不多就行，可以并行；不需要为了测量精度串行或反复重跑。明显异常的单项复测
一次确认即可。

---

## 五、边界

不碰 cuTile、TileLang 的 provider 实现和 timeout；不追 `1.05×`；不重新审核 candidate set /
winner / measurement scope（那些属于第八轮）；不改 canonical DSL 或作者算法；不为某个 target
在共享层加机制；不留兼容分支、fallback 或默认值；不建 test 目录、pytest、fixture；不写版本号、
CHANGELOG、迁移指南。

核查报告里其它未闭合项（tuple/record buffer element、scale-axis relation 的作者表面、
一般调用前置条件、multi-contract、runtime free-axis、sparse、mutable buffer、provider 各自的
gap）这一轮不做，但要在报告里说明它们分别属于哪一轮。

---

## 六、自查

主要改动完成后，从这一轮实际动过的东西出发，去 `ref/triton`、`ref/tilelang` 找同一问题的
真实实现：**provider-local 的 grid/mapping 改写在成熟编译器里怎样保持程序完整？谓词/validity
在 blocking 重写中怎样被保持或证明可丢？多 source 的循环上界怎样确定？**

给出两边的 file:line、具体差别、以及换一个 kernel 或去掉一项前提时这个差别有没有实际后果。
找不出具体差异等于没做自查。不要写打勾表，也不要写"X → 1"这种自评。

---

## 七、产出

一份报告，写清楚：

- provider transform 破坏共享承载物这条性质最终怎么解决的、ProgramGrid 之外还扫到了什么；
- 两处语义缺口修在哪一层、为什么不是局部补丁；
- 全语料编译到 Triton 终端源码的结果与阶段分类；
- 两张 Triton 表的新数字，以及跟旧表逐项对比后哪些是第五轮损伤、修了没有；
- ref 双向对照的具体差异与后果；
- 核查报告里剩余各项分别归到哪一轮。

改动、必要的规格修正和报告整理成语义连贯的提交，工作区干净。
