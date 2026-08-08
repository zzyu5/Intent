# 前端、Kernel IR 与首个 Realization 的重新审计

## 结论修正

上一版报告把“当前 `OpCode` 枚举均能由示例产生”称为“完整前端”，并把一个手写 vector-add Triton emitter 称为后端闭环。这两个结论均不成立。

当前真实状态是：

```text
Python DSL
  -> Python Kernel IR
  -> Intent MLIR 文本导出

Python Kernel IR
  -> Python PhysicalPlan dataclass
  -> 手写 pointwise Triton emitter
```

`intent.compile()` 没有消费 Intent MLIR。Intent MLIR 是旁路展示产物；Physical Plan 也不是 MLIR。现有 emitter 只匹配一维 f32 loop，并固定 tile、warp 与 pipeline 参数。因此它只能证明最小 plumbing，不能证明 MLIR backend、完整 frontend 或 realizer 设计成立。

## Intent MLIR 与 TianchenIR 的关系

当前 Intent dialect 不是 TianchenIR 的复制改名。两者共享 `intent` dialect 名、MLIR TableGen 组织方式以及少量常见 mnemonic，但 IR schema 不同：

- 当前设计以 domain、region、partition、parallel、ordered、state stream、ragged 与 logical view 为核心；
- TianchenIR 以具体 tensor operations 和既有 IntentIR JSON lowering 为核心；
- 当前项目额外定义了 logical domain/region/record/view 等类型；
- 同名 reduce、scan、gather、transpose 等 operation 的参数和语义并不相同。

但是，“重新定义”不等于“已经成熟”。当前 ODS 中大量 operation 仍以 `AnyType` 和未声明属性表示，C++ 侧没有 operation semantic verifier。Python verifier 保存了更丰富的语义，而 MLIR dialect 尚未独立守住这些结构边界。

## 50/50 coverage 实际证明什么

现有 coverage 只证明：当前枚举中的每个 opcode 至少被某个选定示例产生一次，生成的 Python Kernel IR 通过 Python verifier，打印出的 MLIR 文本能被 parser 接受。

它不证明：

- 每个构造的所有 dtype、shape、axis、constexpr 和 control-flow schema；
- symbolic shape 的跨参数一致性；
- stride、alignment、alias/noalias 的完整 ABI 语义；
- record reduce/scan、复杂 helper、多结果与嵌套 helper；
- dynamic domain、ragged 边界、gather invalid/fill 与 scatter conflict；
- ordering、scope、effect dependency 与 RNG identity；
- 文档中全部算法能够从 Kernel IR realization 到后端；
- MLIR 自己能够验证或供 backend 直接消费。

因此不能用一个封闭 opcode 集合的存在性覆盖，证明语言已经覆盖所有算法逻辑。

## 本轮“完整 frontend / Kernel IR”的完成标准

本轮完整性限定为文档定义的 Core，而不是“所有可能算法名称”。每个 Core 构造必须同时具备：

```text
Python DSL surface
  -> AST lowering
  -> typed Kernel IR node/type
  -> Python semantic verifier
  -> 稳定 Intent MLIR schema
  -> MLIR structural/backend-boundary verifier
```

Core 包括：

- kernel/helper ABI、view kind、dtype、symbolic shape、runtime scalar 与 constexpr；
- domain/region/product/partition、parallel/ordered/state stream；
- structured if/for/while 与多 SSA carry；
- tensor expression、broadcast、reshape、transpose、mask 与 cast；
- reduce、scan、contract 与 record state；
- index relation、gather/scatter、ragged membership；
- logical buffer、atomic、fence、effects 与 logical RNG identity。

`sort/topk/group_by` 等不是 opaque Core operation。若需要它们，应由明确算法实现使用 Core 组合，而不是让 realizer 偷换 source algorithm。算法库是否齐全与 Core frontend 是否闭合分开验收。

## 正式编译数据流

本轮采用：

```text
Python DSL
  -> Kernel IR
  -> Intent Kernel MLIR
  -> Realizer
  -> Physical Plan MLIR
  -> MLIR parse + verify
  -> Triton source translator
  -> Triton JIT
  -> Runtime
```

目标不是 Triton MLIR。Backend 的正式输出是可读、可独立运行的 Triton Python source。

关键边界是：translator 必须读取已经被 MLIR parser/verifier 接受的 Kernel IR + Physical Plan，不能绕回 Python Kernel IR 或按 kernel 名称套模板。Python 可以负责 DSL frontend、编译入口和 runtime wrapper，但不能在 MLIR 之外保留另一份决定后端语义的 plan。

## Physical Plan MLIR 的本轮边界

Realizer、搜索、cost model 与 tile policy 是项目最复杂的部分，本轮不宣称完成。只为一个真实 kernel 建立最小但通用的 Plan MLIR schema，至少表达：

- logical extent 与 physical tile；
- worker ownership 与 traversal；
- storage、layout 与 access mode；
- primitive binding；
- boundary/tail；
- pipeline；
- launch grid、program count、warps 与 stages；
- 对 Kernel IR operation/value 的稳定引用。

Plan verifier 必须证明这些引用属于当前 entry，并保持 source ABI、logical workset、tensor-flow、state、index relation 与 effects。未实现的 realization 明确失败，不提供 Python fallback。

## 首个真实 kernel：Triton fused stable softmax

首个闭环固定使用：

```text
source/triton/triton/normalization/softmax/02-fused-softmax.py
```

它与 `doc/kernels/softmax.md` 使用同一 source algorithm：

```text
full-row max
  -> subtract
  -> exp
  -> full-row sum
  -> divide
```

Realizer 不得把它替换成 online softmax。首个 Plan 需要表达原版实现中的：

- row 到 persistent/grid-stride program 的 ownership；
- `next_power_of_2(n_cols)` column tile；
- `col < n_cols` masked tail；
- max 与 sum reduction primitive；
- numerator 的 register-local tensor flow；
- `num_warps`、`num_stages` 与 program count；
- global contiguous input/output layout。

其中算法依赖、reduction identity、row/column logical domain 从 Kernel IR 推导；tile、ownership、warps、stages 与 launch 属于 policy/search 空位。本轮可以使用与原版相同的确定性 policy 走通 demo，但必须把这些选择显式保存在 Plan MLIR，而不是写死在 emitter 中。

## 对照与性能口径

原版和自动生成版必须：

- 使用同一个 upstream Triton kernel 文件作为 baseline，不修改其 kernel source；
- 使用相同 GPU、dtype、shape、stream 和算法；
- 在计时前完成 JIT 与 warmup；
- 将数值比较放在计时外；
- 使用重复测量并报告稳定统计量，而不是一次 CUDA event；
- 明确计时是否包含 output allocation，并保证两边边界一致。

主 workload 采用现有 runtime 的模型级形状 `(8192, 8192)`。由于 DSL canonical contract 当前是 f32，正式横向比较使用双方均为 f32；原版 kernel 本身支持该 dtype。若保留原版 `softmax(x)` wrapper，则自动生成版也用相同的 output-allocation wrapper；若比较纯 kernel latency，则双方都预分配 output 并直接 launch kernel，不能混用两种边界。

## 本轮成功条件

唯一手动 repro 最终必须同时证明：

1. 文档 Core 的 frontend/Kernel IR/Intent MLIR schema 闭合；
2. stable-softmax DSL 生成的 Kernel IR 保留与原版一致的算法顺序；
3. Physical Plan 作为 MLIR 被解析并验证；
4. Triton source 由 Kernel IR + Plan MLIR 自动生成，而非按名称或旁路 Python plan；
5. 自动生成版与原版在同一模型级输入上数值一致；
6. 输出原版与生成版的性能结果，并展示生成的 Triton source 与 Plan MLIR。

在这些条件满足前，不再使用“完整前端里程碑”或“后端闭环完成”的表述。
