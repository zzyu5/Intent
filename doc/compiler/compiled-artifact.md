# 编译产物与运行边界

## Compile result

```python
compiled = intent.compile(kernel, target=device)

print(compiled.source)
print(compiled.ir)
```

编译产物至少包含：

- 一个 callable target kernel entry；
- 对应 launch configuration；
- 可读、可导出的 target source；
- 对应 backend 或 lower-level IR。

## Generated code 是正式输出

生成代码应：

- 保留 source variable 与 region 的可识别命名；
- 明确显示 tile、grid、ownership、storage、layout 与 pipeline；
- 使用结构化 helper，不生成难以阅读的一次性字符串；
- 可以脱离 Intent 继续编译、调试和人工修改；
- 成为性能理解与极端 kernel 人工接管的直接接口。

Profiling、cost breakdown 或 `plan.explain()` 可以作为 compiler tooling，但不是 source language semantics。

## Runtime invocation

调用 Intent kernel 时，用户不提供 `[grid]`。Runtime 根据 compiled artifact 的 entry 与 launch configuration，在当前 device/stream 提交一次 invocation。

Runtime 不负责：

- framework graph partition；
- 输出或 workspace 分配；
- 多-kernel 调用顺序；
- 自动融合或拆分 kernels；
- source variant 的 library policy。

这些仍属于普通 Python wrapper。

## Single-kernel invariant

一个 source kernel invocation 对应一个 target entry invocation。Compiler-private scratch、completion counter 或 target 内部多级 physical implementation 可以存在，但不得暴露为额外 runtime dispatch。
