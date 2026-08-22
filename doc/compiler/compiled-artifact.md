# 编译产物与运行边界

## Compile result

```python
compiled = intent.compile(
    kernel,
    target=intent.TritonTarget(device=0),
    compiler=intent_compile,
)

print(compiled.source)
print(compiled.ir)
```

编译产物至少包含：

- 一个对调用方可见、对应一次 launch 的 target kernel entry；
- 包含 Kernel IR 与 Physical Plan 的组合 MLIR；
- 可读、可导出的 target source；
- 第一次真实 launch 后由 Triton JIT 产生的 backend 或 lower-level IR。

Launch policy 由组合 MLIR 中的 Plan 和生成源码共同保存，artifact 不维护第二份 Python launch/Plan 数据模型。

以 Triton target 为例，调用形式为：

```python
compiled = intent.compile(
    stable_softmax,
    target=intent.TritonTarget(device=0),
    compiler="/path/to/intent-compile",
)
compiled(input, output)
```

`source` 与包含 Kernel IR/Physical Plan 的 MLIR 在 compile 返回时即可读取。`intent-compile` 在同一进程内完成 realization、组合 MLIR 验证与 target emission，并分别输出组合 MLIR 和目标源码。Triton 的 TTIR、TTGIR、LLVM IR、PTX 等 backend IR 由第一次真实 launch 触发 JIT 后写入同一个 artifact；artifact 不用第二套编译路径伪造这些结果。

## Generated code 是正式输出

生成代码应：

- 保留 source variable 与 region 的可识别命名；
- 明确显示 Intent 已决定的 tile、grid、ownership 与 storage；surface 要求显式拼写的 layout 或 pipeline 参数也保留在源码中，下层自行推断的部分不伪造出来；
- 使用结构化 helper，不生成难以阅读的一次性字符串；
- 可以脱离 Intent 继续编译、调试和人工修改；
- 成为检查生成实现与极端 kernel 人工接管的直接接口。

Profiling、cost breakdown 或 `plan.explain()` 可以作为 compiler tooling，但不是 source language semantics。

## Runtime invocation

调用 Intent kernel 时，用户不提供 `[grid]`。Runtime 根据 compiled artifact 的 entry 与 Physical Program，在当前 device/stream 提交一次 target kernel launch。

底层 `compiled(input, output)` launch 不负责：

- framework graph partition；
- 用户输出或跨 source-kernel workspace 分配；
- 多 source-kernel 调用顺序；
- 自动融合或拆分 kernels；
- source variant 的 library policy。

这些仍属于普通 Python wrapper。`compiled.run(input)` 可以作为分配 `empty_like` 输出的 convenience wrapper，但不改变底层 launch 边界。

## Single logical callable invariant

一个 source kernel invocation 对应一个 target kernel invocation。Compiler-private scratch 与 intermediate value 只能服务于这一次 launch；dependency、lifetime 和 visibility 从 Kernel IR def-use 与已选 physical realization 派生。Artifact 不提供自动 kernel fission、跨-launch workspace 或跨 source-callable fusion 入口。
