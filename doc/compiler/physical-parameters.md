# Physical Parameters 与 Tuning

## 1. 参数不是旁表中的名字

Physical parameter是在target compile time绑定、直接参与当前GPU program的typed symbol。例如BM/BN/BK必须同时约束fragment types、program-space extent、loops、access coordinates与validity，而不是只出现在search declaration或generated variable name中。

每个parameter declaration至少定义：

- stable symbol；
- semantic role，例如logical ownership extent、reduction chunk、scan/region segment、group size或provider option；
- integer/bool/enum kind；
- finite candidate domain或可验证constraints；
- 它直接影响的types、operations与launch fields；
- target/provider legality requirements。

Physical parameter不进入KIR，也不成为作者DSL参数。

## 编译期候选数据输入

`intent.compile(..., tuning_config=path)`与`compile_shared_gpu(..., tuning_config=path)`接受有限JSON profile覆盖；CLI对应`--tuning-config <path>`。不传时读取随compiler分发的默认表。表与其shared/provider职责相邻，编译器在配置物化前读取一次；调表不需要重编译C++，但必须重新编译kernel artifact。launch与autotune不再读取该文件。

覆盖文件的顶层命名空间为`shared`、`triton`、`cutile`、`tilelang`，每个命名空间包含已有family到候选行数组的映射。例如：

```json
{
  "shared": {"contraction_narrow": [[128, 128, 32, 1, 128, 1, 8]]},
  "triton": {"contraction": [[4, 2, 1], [8, 3, 1]]},
  "cutile": {"occupancy_loop": [[1], [2]]},
  "tilelang": {"stages": [[2], [3]]}
}
```

显式提供的family整组替换默认行；未提供的family继续使用默认数据。不存在按kernel名称、source或registry选择profile的规则，也不能在JSON中写条件或算法。文件不可读、未知字段/命名空间/family、错误列数/类型、非正整数、重复行或空候选表直接诊断。

Shared行的列依次为`ownership_m, ownership_n, reduction, reduction_outer, scan, traversal_workers, traversal_group`。它们是按role消费的相关粒度偏好，不是对任意shape强制生效的parameter binding：既有投影选择typed domain内不超过请求值的最大值，无此值时选择domain最小值；程序已固定的维度保持固定。相同规则用于默认表和覆盖表。IR graph classification、profile correlation、候选预算与role绑定仍由shared transformations决定，最终tuple明确保存实际值，不能把profile原值冒充最终binding。

Triton行依次为`warps, stages, ctas`；cuTile与TileLang的各family使用单列实际provider值。Provider值不投影到另一个值：先过滤可证明非法的值/组合，再物化typed domain与完整config。cuTile occupancy的可接受范围独立于选用的搜索集合；TileLang按当前GEMM分区选择普通或小线程family，不因该family全部非法而另读默认候选。若当前程序没有合法候选则编译失败，不换算法、不退回默认表。外部provider compiler继续负责其独有的机器资源约束。

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

调优试跑不属于作者的一次可观察invocation。Runtime依据external view的读写方向和实际allocation alias关系维护trial state；需要原始内容的`InOut`、读写alias和atomic状态在每次试跑前保持同一初始内容，不能只在候选之间恢复。Trial参数保留shape、dtype、strides、offset与alias关系，不把每个view独立clone成互不alias的输入。Winner只对调用者参数执行一次；调优失败也不把试跑效果留在调用者状态中。

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
