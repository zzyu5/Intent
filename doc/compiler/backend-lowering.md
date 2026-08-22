# 后端 lowering

Backend lowering 先在同一 MLIR module 中运行 shared Physical Program construction/refinement，再运行 provider-local ProgramForms 与 materialization，得到 provider-legal Program。Terminal translator 只验证并序列化该 program 为 Triton、cuTile 或 TileLang source。Python 只负责调用 compiler、加载产物和运行 entry。

## 共享层与 target 叶子

各 target surface 共享：

- Kernel IR 的 operation/region/def-use 遍历；
- ABI、domain、ragged relation、state stream 与 contraction 分析；
- ownership、traversal、tile role、storage、boundary 和合法搜索轴的 machine 决策；
- operation handler registry 与 unsupported-op 诊断机制。

每个 provider 叶子提供：

- provider-local ProgramForms/refinement：选择有 shared obligation 来源的 target form；
- provider materialization：把已选 form 兑现为 provider-legal Program；
- capability/verifier：拒绝无法表达或不合法的 form；
- terminal translation：只序列化 provider-legal Program；
- compile/run 接线以及对下层 tuner 的显式委托。

接入第四门 GPU tile surface 不应增加 kernel 分析路径，也不应修改共享 emitter traversal；只新增 capability/spelling、leaf handlers 与 runtime adapter。

## 逐轴物理投影

Machine realization 不保存 row/tiled/ragged 之类的 kernel 类别。它逐轴记录可组合的角色，并按用途、层级记录 ownership、lane、ordered traversal、reduction 与 access footprint 等 range。同一逻辑轴可以同时承担 parallel、ordered、reduction、ragged member 或 packed lane 中的合法组合。

| machine concept | Triton | TileLang | cuTile |
|---|---|---|---|
| parallel ownership range | `program_id` 与 grid projection | `T.Kernel` block projection | `ct.bid` 与 grid projection |
| packed scalar lane range | `tl.arange` | `T.Parallel` lane | `ct.arange` |
| ordered traversal range | state-carried loop | `T.Pipelined` / serial loop | state-carried loop |
| access footprint / logical validity | pointer mask 或收紧范围 | guarded copy / range predicate | checked load、gather 或 scatter |
| persistent program choice | grid-stride program loop | persistent block loop | persistent block loop |

一条 realization 可以让不规则 membership、ordered stream、head mapping 与尾块同时作用；provider 不得把组合重新压回 kernel 类别字符串。Region argument、row-vector logical extent 与 stream/ragged relation 都由 shared Physical Program 显式绑定；provider-local passes 可以在这些 obligation 下选择 target-specific access/storage/primitive form，但不能另选 shared ownership、流终点或遍历。

## Operation 对应关系

| Intent / Plan | Triton | TileLang | cuTile |
|---|---|---|---|
| logical view/index | pointer + masked load/store | buffer region + `T.copy` | array/tile load/store |
| `contract` role | `tl.dot` | `T.gemm` | tile MMA/matmul |
| fixed `reduce/scan` role | `tl.max` / `tl.sum` / `tl.cumsum` | fixed target reduction/scan | tile reduction/scan |
| generic `reduce/scan` combiner | typed target helper | target-native generic combiner，或 capability rejection | typed target helper/lambda |
| logical validity | mask / tightened loop | predicate / range | boundary handling |
| Plan storage | compiler-local representation | shared/fragment/local | tile/register storage |
| Plan stream | explicit state-carried loop | pipelined state-carried loop | explicit state-carried loop |

使用 target 的高性能内层 primitive，不等于把数学语义交给 target。Reduce/scan closure、contract reduction axes、operand dtype、accumulator dtype 与 result role 先由 Kernel IR/Plan 固定；leaf emitter 只选择对应 spelling，并让下层完成 collective tree、layout、指令和 machine code generation。Generic reduce/scan 不意味着 arbitrary contract semiring；`contract` 只承接正式声明的 multiply/add 与 dtype capability。

## Single-launch 投影

Shared Physical Program 给出一次 launch 内的 ownership、traversal、range、storage 与 access obligations。公共 analysis index 从 Kernel IR def-use 和 memory effects 派生 value lifetime、terminal 与 visibility；provider-local passes 兑现对应 forms，terminal translator 不重选 workspace owner 或重建 shared physical decisions。任何 provider 都不能在投影期间新增 target kernel、跨-launch workspace 或 host synchronization。

## 诊断边界

Translator 开始前验证 Kernel IR、machine plan 与 target projection。以下情况直接以关联的 source location 报错：

- Plan binding 缺失或引用错误 logical node；
- 逐轴 role/range 与 operation binding 的组合不合法；
- target capability 不包含某个 machine concept；
- 某个 canonical operation 没有注册 target handler；
- target spelling 无法保持 dtype、boundary、effect 或 state semantics。

禁止旁路 Python Plan、根据 kernel 名称套模板、整-kernel matcher，以及 emitter 从 tensor shape 重新猜 physical structure。

诊断必须区分三种性质：

1. **Target capability subset**：目标 API 或程序模型没有等价机械投影，在 emission 前以源码位置拒绝。
2. **Lower compiler cost/failure**：目标源码已经合法生成，但下层 compiler、设备或 toolchain 给出诊断。这是运行状态，不得冒充 Core 或 target-language 不支持。
3. **Intent implementation gap**：Kernel IR 能表达、target 也有能力，但缺少 Plan binding、handler 或机械投影。这一类必须给出 implementation-gap 诊断，不能登记成 target subset。

设备或下层 toolchain 的一次编译结果不能修改 Core 语义。只有能由正式 target/device capability 表达的限制才进入能力检查。

## 与下层系统的关系

Intent 决定依赖算法结构信息的部分：合法 ownership、遍历、tile 关系、片上复用边界、logical validity 与数值角色。Target compiler 负责其程序模型能自行推断的 layout、寄存器分配、指令选择、低层流水线和候选评测。下层能力增强时，surface leaf 应变薄；共享 Kernel IR 与 machine decision space 不因 target surface 的实现变化而改变。
