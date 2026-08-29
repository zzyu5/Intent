# 自查：第五轮的 shared GPU 重构到底做完了没有

第五轮报告 `report/shared-gpu-analysis-pass-reconstruction.md` 声称没有遗留的 shared
compiler 设计卡点，全部 217 个 kernel 通过 shared verifier。这一轮**只做核查，不继续推进**。

要防的是一个具体的历史问题：第三、四轮的报告是**诚实的**——它们各自列出了 gap——但第五轮的
第一次全量仍然逼出了约一万行新后端。也就是说，真正伤人的不是被列出来的缺口，是**没人想到
要去看的地方**。所以这一轮不给你检查清单：清单只能查已经想到的东西。

---

## 一、先把项目自己的记录当探针

`report/history/` 下有第二、三、四轮的报告，第三、四轮各自写了明确的 gap 清单，例如
region fold/scan 未物化、multi-axis reduce 未闭合、multi-contract joint realization、
execution-group 联合 physicalization、buffer realization、cuTile/TileLang 各自的
provider gap。`report/full-gpu-regression-and-triton-performance-closure.md` 里还有第五轮
自己列的十类未闭合事项。

逐条追这些历史 gap 的当前状态：**已经关上、变成了别的形态、还是原样存在但这一轮没人再提。**

第三条最危险，也最可能发生——一个缺口在第五轮的报告里消失，不一定是因为它被修了，也可能
是因为这一轮的关注点换了。

同时注意反方向：第五轮报告**一条 gap 都没有列**。前两轮都列了。这要么说明真的关干净了，
要么说明这一轮把边界改窄了、把某些东西移出了"shared 层"的定义范围。用当时的实现证据判断
是哪一种。

---

## 二、编译通过不等于产物可用

现在的 217/217 只走到 shared verifier。它证明每个 kernel 能产出一份通过验证的 GPU
Program，**没有证明那份 program 里有下游真正需要的东西。**

不用做第六、七轮的工作，但可以用不花钱的方式往前探一步：现有三家 provider 的
legalization/serialization 是纯编译动作，不需要 GPU。把 shared program 送进去看哪些能走到
终端源码、哪些在 provider legality 处停下、停在什么原因上。

这不是要修 provider，是要回答：**第五轮产出的 shared program，形状上是不是足以支撑
provider？**如果大量 kernel 在 provider legality 前就散架，那 shared 层的"完整"定义可能
过窄。

旧链曾经覆盖 cuTile 35/37、TileLang 19/37。那些 kernel 现在都在语料里、都通过了 shared
gate。它们产出的 program 与旧链 materializer 当年实际消费的 executable facts 相比，缺什么。

---

## 三、找剩余路径

重构没做完的典型形状不是"报错"，是**代码里还留着为已经不存在的世界服务的东西**：

- 没有调用者的函数、类型、attribute、enum 值；
- 写进去但没人读的字段；
- 只可能在一种输入上成立的分支；
- 同一件事有两种做法，其中一种现在不再被走到；
- 为兼容而留的默认值、fallback、宽松分支；
- 声明了但从未真正被消费的 typed fact。

沿唯一执行链走：frontend → canonical KIR → KIR-to-GPU construction → shared passes →
provider legalization → serialization。每一处出现上述形状，都是没做完的证据。

特别看那四个仍然 2500 行左右的 family 文件（`RealizePointwiseBlocking`、
`RealizeReductionBlocking`、`RealizeContractionBlocking`、`RealizeRegionFold`）。第五轮
把调查报告点名的那几处重复收敛了，但那份点名清单是调查当时能想到的。这四个文件够大，
足以藏住没被点名的第二份推导。

---

## 四、对着 ref 看，双向

不要只问"ref 有什么我们没有"。两个方向都问：

- ref 里的编译器有某个结构而我们没有——为什么我们不需要？如果答案是"我们还没做到"，
  那它就是一处未完成，不是一处设计差异；
- 我们有某个 ref 没有的结构——为什么我们需要？如果答案是"因为我们的某处表达不够"，
  那它是一个变通，不是一项能力。

对照要由 ref 里的真实实现生成问题，不是为已有结论找类比。给出两边的 file:line、具体差别、
以及这个差别在换一个 kernel/换一种结构/去掉一项前提时会不会有实际后果。

**找不出具体差异等于没做。**"与成熟实践一致""泛化守住了"不是结论。

---

## 五、怎么做和怎么交

要跑就跑。现有 compiler CLI、`--stop-after-shared`、inventory 遍历、provider
legalization/serialization 都是生产路径，可以直接用。不建 test 目录、pytest、fixture 或任何
测试基础设施。不跑 GPU、不跑 benchmark、不更新 baseline CSV。

**发现的问题：明确的、局部的、能当场修干净的就修（例如死代码、失效分支、明显的第二条路）。
结构性的写清楚，不要在这一轮动手改架构。**

产出一份报告，写清楚：

- 历史 gap 逐条的当前状态，以及"消失"的那些是被修了还是被移出了范围；
- shared program 送进三家 provider 编译动作后停在哪里、为什么；旧链覆盖过的结构现在缺什么；
- 找到的剩余路径、死代码、单输入分支、第二份推导，以及修掉了哪些；
- ref 双向对照的具体差异和后果；
- 你的判断：shared 重构是真的关上了，还是有一部分只是没人再提。

不要写表格式的自评（"X → 1"这种）。写你试了什么、看到了什么、file:line 在哪。

如果结论是确实关干净了，也要给出支撑它的具体证据，而不是没找到问题就当作没有问题。
