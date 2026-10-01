# Intent kernel 示例

`examples/kernels/` 只保存作者编写的目标无关算法，按 activation、contraction、normalization、streaming 等算法职责分类。编译器保持这些程序的类型、shape、数值和 effects；target 在编译调用中选择。

安装说明见 [README](../README.md)。普通 host 示例直接复用 `kernels/` 中的定义：

| 入口 | 展示的完整调用 |
|---|---|
| `python examples/softmax.py --target triton` | 编译一次、传入 PyTorch tensor、分配声明的输出、定位编译产物 |
| `python examples/softmax.py --target cutile` | 在独立 cuTile 环境中复用同一算法定义 |
| `python examples/softmax.py --target triton --torch-compile` | 先普通调用同一算子完成 JIT/调优，再通过 opaque custom op 进入 `torch.compile(fullgraph=True)` |
| `python examples/softmax_forward_backward.py --target triton` | 作者显式编译 forward/backward，传递中间结果并决定调用顺序 |

Forward/backward 示例使用既有 backward 定义的固定 shape，不注册 autograd，也不由 compiler 隐式创建第二次调用。PyTorch adapter 当前支持 GPU 的只读 `In`/scalar 输入和新分配的 `Out`；fake 实现来自同一 ABI，不调用 provider、不读取 tensor 数据。`InOut`/返回 alias 暂不支持；backward 需作者通过返回的 `CustomOpDef.register_autograd` 注册。示例输出不是 benchmark 或数值验证结论。每个脚本把 host 执行放在 `main` 下，因此也能作为模块加载。

只编译而不运行可用：

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
```

JSON 包含原定义位置、真实编译阶段、source/IR/metadata 和日志路径。编译失败时从 `diagnostic` 与 `files` 继续查看；`intent doctor --target triton` 检查依赖和目标解析。完整算法保留在普通示例中，manual MCP 只提供通用语言规则。

执行、reference baseline、生产 registry、实验结果和 pass 对照在 [experiments/](../experiments/README.md)：

- [GPU：Triton / cuTile，保留 TileLang](../experiments/gpu/README.md)
- [CPU：Mojo / Weft](../experiments/cpu/README.md)
- [MLU：DSA / BANG C](../experiments/mlu/README.md)
- [Agent TritonBench](../experiments/agent_tritonbench/README.md)

新增算法示例放在 `kernels/`，相应运行适配与实验数据放在所属实验组。不要在 examples 下重新建立 runner 或 baseline 副本。
