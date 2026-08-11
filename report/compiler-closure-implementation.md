# 上一轮收官重构实现报告

## 结论

上一轮不是补 baseline，而是围绕三个问题做了五个代码提交：

1. 把“一个逻辑轴只能带一个物理范围”改成同轴多 purpose、多 level 的 Physical Plan；
2. 用同一模型同时承载卷积的 access footprint 与扫描的两级 traversal；
3. 补齐 helper、负整数语义、signed W4，并清掉没有端到端语义的旧 control/memory 路径；
4. 审计 realizer/emitter 中的 fallback、kernel 分类和重复决定；
5. 用现有 42 个 repro × 3 个 provider 做收官核验。

完成度不是简单的“全部完成”：

| 项目 | 当前状态 | 准确含义 |
|---|---|---|
| 同轴多物理范围 | 完成 | Axis 与 Range 分离，purpose 与 level 可组合 |
| 分块扫描 | 完成 | 同一位置轴的 chunk + scalar step 已投影为真实双层循环 |
| 卷积读覆盖 | 表示完成，物化未固定 | Plan 已记录 halo access range；Triton/cuTile 仍把具体合并交给下层 |
| helper | 完成当前单-kernel合同 | frontend 直接 SSA 内联，不建立设备函数 ABI |
| 负整数 `//`、`%` | 普通 pointwise 路径完成 | 三 target 使用 Python floor 语义；while 条件内仍不支持 |
| signed W4 | 真实 kernel 完成 | packed i32 内以可移植恒等式解码；generic i4 target dtype 未完成 |
| `break`、`continue`、`fence` | 明确不支持 | 删除假 IR 路径，在源码位置诊断 |
| 冗余与 kernel 特判 | 已清理并审计 | 没有 kernel-name realizer/emitter；仍保留一项共享 persistent 策略 |

性能数字与 source baseline 没有在这里复制。固定数据仍在：

- `report/baseline/compiler-closure.md`
- `report/baseline/kernel-performance.csv`

## 一、一个逻辑轴如何承载多个物理范围

### 1. 原模型的问题

逐轴角色重构已经允许一个轴同时是 parallel、ordered、reduction 或 ragged member，但物理 tile 仍挂在 axis 本身。于是同一轴出现第二种范围时只能覆盖第一种：

- 卷积输出轴需要 ownership tile，同时还需要更大的 input access footprint；
- selective scan 的位置轴需要外层 chunk，同时需要 chunk 内有序 scalar step。

这不是两个 kernel 的特例，而是“逻辑轴”和“物理范围”仍被错误地一对一绑定。

### 2. 新 Plan 结构

`intent_plan.axis` 现在只保存角色与 program-space 归属。物理范围成为独立的 `intent_plan.range`：

```text
axis_node
purpose
level
tile
transfer_node? / source_axis?
lower_offset / upper_offset
```

当前 purpose 为：

| purpose | 负责的物理事实 |
|---|---|
| `ownership` | 一个 program 对逻辑轴拥有的写出区域 |
| `traversal` | ordered 轴如何推进 |
| `reduction` | 收缩轴的物理范围 |
| `lane` | 片内稠密 lane 范围 |
| `access` | 某次读取相对 ownership 多出来的覆盖范围 |

`AxisBinding` 持有 `SmallVector<RangeBinding>`，emitter 通过 `(purpose, level)` 精确读取。`fff89f9` 进一步删除了初版“找不到当前 role 就 fallback 到其他 range”的行为：program/ragged 只读 ownership，stream 只读 traversal，reduction 只读 reduction，lane 只读 lane。

这一步很关键。否则 schema 虽然允许多个 range，实际消费仍会偷偷回到“选一个主范围”。

### 3. verifier 承担的约束

Plan verifier 现在检查：

- range purpose 必须合法并与轴角色一致；
- access 必须绑定 transfer、source axis 和一个已有的 parallel ownership；
- 非 traversal range 只能是 level 0；
- traversal level 必须连续，当前 level 1 必须是 scalar `one`；
- emitter 不能靠另一个 purpose 的值补缺失决定。

因此多 range 不是一组随便附加的 metadata，而是有消费合同的物理决定。

## 二、卷积：完成的是 footprint 表示，不是强制 halo buffer

### 1. access range 如何从算法中得到

共享 `KernelFacts` 沿 `view_load` 的索引 SSA 恢复受限 affine expression。只有满足以下条件才形成 access fact：

- 索引中恰好有一个已经 partition 的 ownership 轴；
- ownership 轴系数为 1；
- 其余参与轴有静态 extent；
- 可以算出相对 ownership 的有限 lower/upper offset。

因此当前卷积得到：

```text
Conv1D: output ownership + access[-2, +2]
Conv2D: H ownership + access[-1, +1]
        W ownership + access[-1, +1]
```

realizer 把它们写成绑定具体 transfer/source-axis 的 `intent_plan.range purpose="access"`。整个过程只看索引 SSA、ownership 和静态小域，没有检查 convolution op 或 kernel 名字。

### 2. 三个 target 实际做了什么

- Triton、cuTile 继续发射精确索引张量和边界，把重复地址的合并、缓存与实际搬运交给下层；当前 emitter 没有直接读取 access range 去新建唯一 halo buffer。
- TileLang 会读取一个 transfer 的 access ranges。单轴 footprint 可以继续投影，因此 Conv1D 通过；两个 access ranges 的联合 footprint 当前不能成为一个 parallel fragment，因此 Conv2D 明确 N/S。

本轮曾在 Triton 中显式物化 Conv1D 唯一 halo hull。数值正确，但 p50 从约 `0.0082 ms` 变为约 `0.0102 ms`。这条路径把下层可自行处理的 load 形态提前固定，并造成约 24% 退化，所以没有保留。

准确结论是：

- “写范围”和“读覆盖”已经成为共享 Physical Plan 中两份独立事实；
- Conv1D/Conv2D 的正确索引与边界在 Triton/cuTile 实际运行；
- 编译器尚未建立一个跨 target、能稳定获益的“唯一 halo tile + 片上物化”机制；
- source 中的 causal Conv1D 与当前 same-padding Conv1D 算法不同，不能接成伪 baseline。

如果把“卷积完成”定义成已经显式消除 patch 中全部重复地址，那么这一项没有完成；如果定义成物理模型已经能表达独立读覆盖，则已完成。

## 三、扫描：同一轴的 level 0 / level 1 已真实兑现

selective scan 的位置轴现在同时带：

```text
traversal level 0 = chunk
traversal level 1 = one
```

shared realizer 在 ordered 轴上先选择外层 traversal；当该轴同时是 serial loop domain 时，再附加 level 1 的 scalar step。

三个 emitter 都显式读取两级 range：

- level 0 形成外层 chunk loop；
- level 1 形成 chunk 内逐位置 loop；
- leave/indent 逻辑也按是否存在 level 1 退出两层，而不是只把 range 当注释；
- 缺少 level 0、或 level 1 不是 `one` 时直接报不支持。

因此这部分不是“Plan 能表示但 emitter 不消费”。三个 provider 都运行了真实的两级循环，并与逐位置 reference 一致。

source 中的 Mamba chunk scan 是多状态、带块间作者编排的另一种算法。它仍是结构参考，不是当前 selective scan 的一对一性能 baseline。

## 四、辅助函数如何闭环

### 1. 选择 frontend inline，而不是建立第二套函数 ABI

原先的问题是 frontend 可以识别 `@intent.fn`，但若生成 canonical call，就还需要同时定义：

- helper function body 如何进入唯一 module；
- helper 的 facts 与 Plan 如何独立建立；
- 三个 emitter 如何生成设备函数 ABI；
- constexpr、shape、region 与多返回值如何跨 call 边界。

对于当前“一个 source kernel 对应一个 target entry”的语言合同，这会平白增加一套表示层。最终选择是在 frontend lowering 中直接内联 helper：

1. 调用点只接受 positional SSA 参数；
2. `FrontendCompiler` 用 `(helper definition, argument types)` 检测递归；
3. `FunctionLowerer` 保存 caller 的 definition、source、environment 与 loop stack；
4. 在 caller 当前 MLIR block 中 lowering helper body；
5. 显式 `return` 的一个或多个 SSA 值回填调用点；
6. 恢复 caller 状态，后续 realizer/emitter 只看到普通 canonical ops。

结果是 canonical Kernel MLIR 中没有 `intent.call`、helper symbol 或另一份函数计划。

### 2. 当前 helper 合同

已支持：

- 一个或多个 SSA 返回值；
- 在普通表达式中复用；
- constexpr/type/source-location 继续由 frontend 管理；
- 四个 attention 算法共用在线归一化状态更新 helper。

明确不支持：

- keyword 参数；
- 递归；
- helper 内 structured region 中的 early return；
- 跨 kernel 调用或独立设备函数。

这些限制都在源码位置报错，不会落到 emitter 才失败。

## 五、负整数整除与取模

### 1. 当前跨目标合同

普通 pointwise `//` 与 `%` 采用 Python floor 语义：

```text
q = floor(a / b)
r = a - q * b
```

三个 target 的机械投影都先转成 i64，构造绝对值除法得到 truncating quotient，再根据 operand/remainder 符号修正 quotient 与 remainder。它不依赖三门 surface 原生 `%` 对负数的差异。

### 2. 本轮真正新增了什么

这套多语句 lowering 在本轮提交前已经存在。本轮没有重新造一套除法机制，而是修改真实索引 kernel，让它们必须经过负数合同：

- `shifted_row_copy` 使用 `((row + 1) % -M) % M`，制造负 divisor；
- `grouped_query_head_add` 使用 `(query_head - HQ) // group + HK`，制造负 dividend。

两个 kernel 在三个 provider 上均数值通过。这证明当前普通 pointwise/address 路径没有继续依赖“样本全是非负”的偶然性。

仍有一个明确边界：floor divide/remainder 需要多条 target 语句，因此当前 scalar `while` condition 的内联路径会直接拒绝它们。上一轮没有补这条组合，也没有独立穷举所有正负 operand 组合。

## 六、signed W4 如何避免目标右移差异

实际 W4 kernel 的 ABI 仍是：

```text
packed_weight: i32[K / 8, N]
```

本轮把原 unsigned nibble 解释改为 signed two's-complement 解码：

```text
nibble = (packed >> shift) & 15
sign = -((nibble & 8) << 1)
unpacked = nibble + sign
```

该恒等式只对已经 mask 到 `[0, 15]` 的非负 nibble 做左移、按位与和加法，不要求任何 target 对负数右移给出相同结果。repro 真实生成 `[-8, 7]` 权重，pack 前做 `& 15`，三 provider 的 W4A16 contraction 均数值通过。

这里要区分两件事：

- packed i32 中的 signed W4 权重 kernel 已支持；
- generic `I.i4` tensor/view 尚未进入三个 GPU emitter 的 dtype mapping。

因此不能把本轮结果外推为“语言已支持任意 i4 外部张量 ABI”。

## 七、终止类与内存序构造的处理

### 1. `break` / `continue`

检查现有真实 kernel 后，没有算法需要 portable non-structured loop exit。保留 canonical op 只会形成“前端看似支持、Plan/emitter 不支持”的假语言能力。

本轮删除了：

- canonical `intent.break` / `intent.continue`；
- 对应 operation enum、terminator/effect/signature；
- Facts/Plan/emitter 可能误认为它们可达的旧接口。

Python AST 遇到两者时直接给出带源码位置的诊断。当前 portable loop 合同只有结构化 condition、yield 与 return。

### 2. `fence`

三个 target 对 scope、ordering、visibility 没有共同语义，不能把某一个 surface 的 fence 字段搬进 Kernel IR，也不能降成 no-op。

因此 canonical `intent.fence` 被删除；`I.fence` 只保留作者可见的诊断入口，调用即说明“支持的 tile languages 之间没有 portable semantics”。这是一条有意保留的错误路径，不是未删除的 lowering。

## 八、冗余路径与“不是编译器思想”的审计

### 1. 实际删除的冗余

- 旧的单 tile axis 字段与“主范围”假设；
- 多 range 初版按其他 purpose fallback 的 `primaryRange` 行为；
- canonical call/break/continue/fence 及对应死 schema；
- helper 内联完成后仍传递 callable type signature 的旧参数；
- reduce、scan、scatter-reduce 等 intrinsic 对同一 callable 信息的重复表达。

`getTile()` / `getTileRole()` convenience API 仍存在，但它们内部只调用当前 active role 的精确 `roleRange()`，不再保存或推导第二份范围事实。

### 2. 没有发现的错误形态

当前没有发现：

- 按 kernel 名字选择 realizer 或 emitter；
- 为 Conv、scan、attention、MoE 单独增加 whole-kernel matcher；
- target 通过 entry 名决定算法结构；
- Python typed Kernel IR 或 Python target emitter；
- target 从 tensor shape 重建 ownership/access/traversal 来覆盖 Plan。

entry 名只用于 MLIR symbol 与最终 target function name。

### 3. 仍然保留且需要知道的边界

**shared persistent 策略仍是一个 whole-program 判断。**当某个 contraction 拥有至少三个 program axes、至少两个 tiled 且不含 ragged ownership 时，shared realizer 选择 persistent traversal，并把 program axes 折叠到同一 worker。它不按 kernel 名或 provider 分支，选择的是改变源码循环结构的 program mapping；当前 batched contraction 性能依赖它，所以本轮没有删除。但它仍是一项基于结构计数的策略，而不是从单轴局部事实独立得到。

**target emitter 仍然较厚。**三个 leaf 中还有 staged workspace、ragged、row mapping、allocation、GEMM primitive、runtime launch 等 target-specific control path。审计没有发现它们重新决定算法或 ownership，但它们也不是简单的静态字段打印器；当前边界更准确地说是“共享决定 + target-specific mechanical projection”。

**TileLang 对 access range 的消费主要表现为能力检查。**它能接受一个 affine access range，遇到两个就明确拒绝；没有在 leaf 中重新推导联合 footprint。Triton/cuTile 则尚未使用 Plan access range 物化独立 halo。

## 九、代码提交边界

| Commit | 内容 |
|---|---|
| `2141e56` | Axis/Range 分离；access facts；扫描 level 0/1；三 target range 投影 |
| `8292ed3` | helper frontend inline；四 attention helper 化；删除 call/break/continue/fence 假路径 |
| `8672bac` | 真实负 divisor/dividend 索引；signed W4 portable 解码与输入 |
| `fff89f9` | range 消费按 active role 精确化；加强 Plan 不变量；删除 fallback |
| `1d8fcec` | 删除 helper 内联后遗留的 callable signature 参数 |

## 十、验证与当前边界

本轮最后一次全量执行为 42 个公开 repro × 3 个 provider：

| 状态 | 数量 |
|---|---:|
| 数值 PASS | 123 |
| 明确 N/S | 2 |
| 下层运行时 FAIL | 1 |

与本轮直接相关的结果：

- Conv1D：Triton、cuTile、TileLang PASS；
- Conv2D：Triton、cuTile PASS，TileLang 对双 access-range footprint 明确 N/S；
- selective scan：三个 provider 的真实双层 traversal PASS；
- attention/helper：四个 attention 算法保持运行，dense attention 三 provider PASS；
- negative div/mod：两个真实索引 kernel 三 provider PASS；
- signed W4A16：三个 provider PASS。

另外两项不是本轮抽象的假成功：

- TileLang CAS 因下层无等价 primitive，明确 N/S；
- TileLang LayerNorm backward 的四个候选均在下层 autotuner benchmark/module load 阶段触发 CUDA launch failure，仍是 FAIL，没有用 target 特判掩盖。

本轮没有建立新的 test 目录或验证脚手架；全量核验仍通过既有单一 repro 入口完成。
