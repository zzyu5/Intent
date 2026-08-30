# 调查：现在到底是什么状况，下一轮该做什么

**这一轮不改实现。**产出是一份调查报告和一份下一轮范围建议。

先读完 `doc/` 全部规格、`AGENTS.md`、`report/triton-foundation-completion.md`、
`report/shared-gpu-reconstruction-completion-audit.md`，以及 `report/history/` 下的历史轮次
报告——**历史报告只作为"当时是这么说的"来读，不作为事实依据。**

---

## 三条纪律，先说，因为这是反复出问题的地方

**一、不许引用历史轮次的结论当既定事实。**第三、四、五轮的报告各自都在自评。它们声称
"横向完成"的地方，后一轮全量都证伪过。任何要拿来当依据的结论，必须在当前 HEAD 上重新
取证。

**二、"target 不支持"这类结论有一个现成的证伪方法。**registry 里每个 entry 都有同语言的
手写 source。**如果某个 entry 在某个 target 上有手写 source 并且能跑，那个 target 就能表达
它——"target 能力边界"在这一格上永远是错的答案。**没有 source 的格子上，这个结论既未经
检验，也不影响任何一张表。用这条把历史报告里所有"provider 能力边界"的说法重新过一遍。

**三、自查不用清单，对着 ref 看。**`ref/triton`、`ref/tilelang` 里同类问题怎么解、我们差在
哪、有没有实际后果，给两边 file:line。找不出具体差异等于没做。

---

## 要调查的事

下面每一条都是**问题**，不是结论。其中标注"未经证实的观察"的，是讨论中提出但没有验证过
的猜测——**验证它、推翻它、或者发现它问错了方向，都是有价值的产出。不要因为它写在这里
就往上凑。**

### 1. 那 119 个到不了 terminal source 的，各是什么性质

`report/triton-foundation-completion.md` 给的分类是：100 fragment 带 runtime dimension、
7 mutable buffer、5 masked gather、3 candidate legality、1 scaled-contract legality、
3 多 source 不 lockstep。

对每一类回答：**是没实现，还是实现了但没被触发，还是根本不该以这种形式出现？**

特别注意有手写 Triton source 的那些格子（`scaled_fp8_splitk_gemm`、
`block_sparse_gqa_decode`、`flaggems_fp8_mqa_logits` 等）。**source 就在仓库里，读它。**
它构造出了什么样的 program，我们的构造停在哪一步、差的是什么。

### 2. shared verifier 检查的东西和 `doc/compiler/` 写的不变量是否一致

*未经证实的观察*：`doc/compiler/physical-parameters.md` §2 说 fragment shape 只能由常量和
physical parameter 组成，但 `VerifyGPUProgram.cpp` 似乎把 fragment extent 和 launch extent
交给同一个 `verifyExpressionSymbols` 验，而那个函数只检查符号有没有声明。如果成立，那么
带 runtime extent 的 fragment 可以合法通过 shared verifier。

不要只查这一条。**把 `doc/compiler/` 里所有写成不变量的陈述，跟 verifier 实际检查的东西
逐条对照**，找出还有哪些"规格写了、verifier 不查"的地方。

顺带回答一个更根本的问题：**verifier 应该保证"这份 IR 自洽"，还是"这是一份完整的 physical
program"？**两者的差别决定了 217/217 这个数字意味着什么。参照 ref 里 verifier 承担什么职责。

### 3. blocking passes 是决策器还是形状匹配器

*未经证实的观察*：四个 `Realize*` pass 里有约 231 处 `return std::nullopt` / `return
failure()`。第三轮报告描述过 contraction blocking 的前置条件（unit-step direct loads、单
reduction pair、零 tail fill、scalarizable accumulator、unique store path、mapping 覆盖整个
program space 且只有一个可独立改写的 contract）。这看起来像一张形状清单。

要查的是：**这些早退点里，哪些是"这个结构需要一个不同的决策"，哪些是"当时只处理了见过的
形状"？**以及**当 pass 决定不处理时，会发生什么——报错，还是让程序原样通过？**

这跟第五轮修掉的 `operator_kind` 可能是同一种问题的另一个位置：那次是"语义用裸整数承载"，
这次可能是"决定用形状匹配做出"。是不是同一种问题，由你判断。

对照 ref：Triton 作者手写 block shape，所以 Triton 编译器不需要这个决策；`ref/tilelang`
的作者也写。**那么在 ref 里，跟"编译器自己选粒度"最接近的是什么？它是怎么组织的？**如果
ref 里没有对应物，说明这是 Intent 特有的责任，那更要说清楚它现在的形态对不对。

### 4. 第五轮的损伤范围

`triton-5090.csv` 现在 20/54，旧表 51/54。31 个从 pass 退到不 pass。

*未经证实的观察*：这 31 个可能就在那 100 个 dynamic fragment 里，即同一个原因。

查清楚：**这 31 个分别停在哪、原因是不是同一个、旧表是在哪个提交上跑的、中间哪次改动让它们
停下来的。**如果是某个 pass 的前置条件在重构中收紧了，指出是哪一条。

### 5. 结构纪律

*已确认的事实*：`find -type d -empty` 给出 43 个空目录，它们是已删架构的完整骨架
（`lib/Dialect/Plan/IR`、`lib/Target/Common/*`、三家 `Target/*/Lowering/*` 共 18 个、
`Target/GPU/{Transforms,Realization}/*`，`include/` 下镜像一份）。`AGENTS.md` 第一条纪律是
"目录结构就是架构"。

八个文件超过 1500 行，最大的 `KIRToGPU.cpp` 4729 行。

调查：**这些大文件里，多少是真实的职责、多少是同一件事的重复形态？**按什么边界拆才是按
职责拆而不是按行数拆？以及：**现在拆合适，还是要等某些结构定下来之后再拆？**给出判断和
理由。

空目录属于纯减法，可以在这一轮直接删掉，但要在报告里说明它们是什么的残留。

### 6. `python/intent/compiler/inventory.py`

这是上一轮为了跑全语料加的 Python 驱动，不参与编译任何东西，性质上是测试装置。它这一轮
还要用来取证。**用完之后删掉**，报告里说明取证时用的命令，以后需要同类遍历时写一次性命令、
跑完即删。

顺带核一遍：**代码里还有没有别的东西是测试性质、却放在产品树里的？**

### 7. H100

外部占用已经解除。但在上面这些问题弄清楚之前跑全量，跑出来的表会立刻作废。

在报告里给出判断：**这一轮要不要跑 H100，什么时候跑才有意义。**

---

## 最后：下一轮该做什么

调查报告的最后要回答这个，而且要基于查到的事实，不是基于上面这些猜测：

- 下一轮的主线是什么，为什么是它；
- 它的完成状态怎么判断——**不要用"多少个通过"当唯一判据**，因为 217/217 已经证明一个数字
  可以在不变量不被执行时毫无意义；
- 哪些事情可以顺手做掉，哪些必须等主线之后；
- 第六轮 cuTile 现在能不能开始，如果不能，缺的是什么。

如果调查过程中发现上面某个问题问错了、或者有更重要的问题没被问到，**直接说，并给出依据**。
这比把七个问题都答完更有价值。

---

## 产出

一份报告。写清楚每个问题的答案、取证方式、以及 file:line。不要写表格式的自评。

除了删空目录之外不改实现代码。取证可以读、可以跑现有生产路径的编译；不跑 GPU、不跑
benchmark、不更新 baseline CSV、不建 test 目录/pytest/fixture。
