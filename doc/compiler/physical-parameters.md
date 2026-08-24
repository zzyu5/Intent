# Physical Parameters 与 Tuning

## 1. 参数不是旁表中的名字

Physical parameter是在target compile time绑定、直接参与当前GPU program的typed symbol。例如BM/BN/BK必须同时约束fragment types、program-space extent、loops、access coordinates与validity，而不是只出现在search declaration或generated variable name中。

每个parameter declaration至少定义：

- stable symbol；
- semantic role，例如logical ownership extent、reduction chunk、scan chunk、group size或provider option；
- integer/bool/enum kind；
- finite candidate domain或可验证constraints；
- 它直接影响的types、operations与launch fields；
- target/provider legality requirements。

Physical parameter不进入KIR，也不成为作者DSL参数。

## 2. Parameter expressions

Fragment shapes与compile-time loop steps可以使用由常量和physical parameters组成的typed integer expressions。Runtime logical extents仍是SSA values；二者不能用字符串混合。

典型关系包括：

```text
grid_m = ceil_div(M, BM)
m_coord = program_m * BM + range(0, BM)
m_valid = m_coord < M
fragment_type = fragment<f16, [BM, BK]>
```

这里M是runtime logical extent，BM/BK是compile-time physical parameters。Grid和validity显式连接两类values。

## 3. Structural decisions 与 parameter decisions

Shared passes先确定一份physical program structure，例如program mapping、loop nesting、value/access graph与structured-op skeleton。Parameters绑定该程序中的granularity和provider compile options。

不建立任意physical-program Cartesian product或图级structural autotuner。若一种选择会改变program structure且下层无法由参数自行形成，它由typed compiler pass作出并写入IR；若下层已把该选择作为compile-time config并能实测winner，Intent只提供合法parameter domain。

`num_warps`、`num_stages`、`num_ctas`虽然会驱动Triton TTGIR结构变化，仍属于Triton provider parameters：Intent不写设备无关阶梯表选winner，只声明与当前program兼容的候选值和可证明constraints。

## 4. Candidate 与 instantiation

一个candidate是所有required physical/provider parameters的concrete binding。Candidate instantiation必须得到一份完整、合法的provider program；不存在“候选只改字符串、程序仍靠emitter解释”的状态。

Legality分两级：

1. Intent验证可证明约束，例如positive/static extent、shape关系、operation capability、grid限制、已知资源上限和provider surface requirements；
2. provider compiler验证其拥有的layout、register、shared-memory、instruction与pipeline约束。

Intent删除确定非法的候选，但不复制下层完整resource allocator。下层编译失败若来自无法预知的机器资源，可以作为该candidate无效；不能反向改变KIR或静默使用另一算法。

## 5. Provider autotuner

Provider autotuner对剩余candidates分别编译、benchmark并选择winner。Winner是runtime/tuning artifact，不写回canonical KIR或shared GPU IR，也不变成设备型号分支。

Triton路径应生成一份参数角色明确的kernel与真实`Config`集合，让Triton autotuner选择BM/BN/BK、num_warps、num_stages、num_ctas及被允许的provider forms。cuTile/TileLang使用各自实际支持的tuning入口；没有下层tuner时，provider runtime可以对同一已声明candidate set实测选择。

## 6. Provider form candidates

Pointer/descriptor、cuTile gather spelling等local form只有满足下列条件时才进入candidate surface：

- 两种形式实现同一shared GPU access/operation；
- 每种形式都能在serialization前独立legalize；
- 下层或provider runtime能真实编译和测量；
- 选择不会改变KIR algorithm、kernel数量或计时scope。

若form改变operands、types或control，它必须先成为current program中的provider-local extension/compile-time branch，不能只在serializer里用字符串条件切换。

## 7. Cache 与 specialization identity

Compiled-candidate identity至少包含：

- canonical kernel specialization与public ABI；
- runtime shape/dtype specialization key；
- selected provider与hardware target；
- concrete physical/provider parameter bindings；
- compiler/provider code identity与relevant options。

Autotune winner cache另外以runtime tuning key索引候选timings。Compiled artifact cache与winner cache是不同层次，不能把winner写进IR冒充编译决定。

## 8. Baseline 对照边界

比较generated与手写provider source时，双方必须暴露语义角色相同的candidate集合并各自由同一provider tuner选择winner。固定同一个winner没有意义；不同候选预算也不能归因于compiler program quality。

Tuning只选择参数和已声明local forms，不能掩盖缺失的program mapping、access graph或structured realization。若generated依赖更大的search space才弥补结构缺口，该问题仍属于compiler IR/passes。
