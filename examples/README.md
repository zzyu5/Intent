# Intent kernel 示例

`examples/kernels/` 只保存作者编写的目标无关算法，按 activation、contraction、normalization、streaming 等算法职责分类。编译器保持这些程序的类型、shape、数值和 effects；target 在编译调用中选择。

安装说明见 [README](../README.md)。普通 host 示例直接复用 `kernels/` 中的定义：

| 入口 | 展示的完整调用 |
|---|---|
| `python examples/softmax.py --target triton` | 编译一次、传入 PyTorch tensor、分配声明的输出、定位编译产物 |
| `python examples/softmax.py --target cutile` | 在独立 cuTile 环境中复用同一算法定义 |
| `python examples/softmax.py --target triton --prepared` | 显式提供输出，准备一次调用，再分别执行 `launch()` 与 `result()` |
| `python examples/softmax.py --target triton --inspect-native` | 执行后查看当前配置、候选状态与 SDK 实际提供的原生资源 |
| `python examples/softmax.py --target triton --torch-compile` | 先普通调用同一算子完成 JIT/调优，再通过 opaque custom op 进入 `torch.compile(fullgraph=True)` |
| `python examples/softmax_forward_backward.py --target triton` | 作者注册已有 backward，保存 forward 输出，用 `Tensor.backward(upstream)` 取得输入梯度 |
| `python examples/softmax_forward_backward.py --target triton --torch-compile` | 同一 forward/backward 注册进入 PyTorch 图编译；两份 kernel 在捕获前完成首次 JIT/调优 |
| `python examples/softmax_forward_backward.py --target mojo --torch-compile` | 同一作者 forward/backward 在 Mojo CPU runtime 上接入 PyTorch 图编译与 autograd |

`artifact.run(...)` 省略声明的 `Out`，返回新分配的输出；显式调用 `artifact(...)` 保留全部 runtime 参数的声明顺序。`artifact.prepare(..., outputs=(...))` 接受相同输入和指定的 `Out`，准备过程不执行 kernel。返回的调用对象拥有这次参数绑定：`launch()` 执行，`result()` 取得输出容器而不执行或同步。零输出返回 `None`，单输出返回该值，多输出按声明顺序返回 tuple；`InOut` 始终由调用方传入。

`--inspect-native` 读取 `artifact.observation`，也可与 `--prepared` 一起使用。资源带有来源、阶段和单位；SDK 未提供的值明确显示原因。快照不是 correctness 或 occupancy 结论，也不会为了观察额外执行 kernel。首次真实调用前该属性为 `None`。

Forward/backward 示例复用既有的 `4096 × 4097`、f32 backward 定义。作者分别编译两个 kernels，把它们注册为 PyTorch custom ops，再通过 `register_autograd` 提供梯度公式：`setup_context` 保存 probabilities，backward 回调调用已注册的 backward op。普通 Python wrapper 决定这两个 kernels 的关系，compiler 不推导梯度或隐藏增加调用。这里只演示一阶梯度；更高阶梯度需要作者另行提供相应公式。

PyTorch adapter 支持 GPU 和 Mojo CPU runtime 的 `In`/scalar 输入、新分配的 `Out`，以及声明为 mutation 的 `InOut`。`InOut` 原位更新，不额外返回；没有 `Out` 时返回 `None`。每个 `InOut` 不得与其他输入共享 Torch storage，返回 alias 仍不支持。Runtime 通过 `infer_outputs` 提供 fake 输出，复用普通调用的公共 shape、dtype 和 stride 合同，不调用 provider、不读取 tensor 数据。`register_autograd` 仅用于 functional custom op，PyTorch 不接受 mutable custom op 的该注册；Weft 和 BANG C 的原生 buffer 接口不冒充 Torch tensor 接口。

两个脚本都接受 `--target triton`、`--target cutile` 和 `--target mojo`，输入 tensor 跟随 artifact 的实际 CPU/CUDA 设备。Mojo 使用公开 `MojoTarget()` 的默认配置；SDK 可通过 `INTENT_MOJO` 指定。换 target 不改变示例算法、shape 或 dtype。示例输出不是 benchmark 或数值验证结论。每个脚本把 host 执行放在 `main` 下，因此也能作为模块加载。

只编译而不运行可用：

```bash
intent compile examples/kernels/normalization/softmax.py:stable_softmax_f16 --target triton --json
```

JSON 包含原定义位置、真实编译阶段、source/IR/metadata 和日志路径。编译失败时从 `diagnostic` 与 `files` 继续查看；`intent doctor --target triton` 检查依赖和目标解析。完整算法保留在普通示例中，manual MCP 只提供通用语言规则。

执行、reference baseline、生产 registry、实验结果和 pass 对照在 [experiments/](../experiments/README.md)：

- [GPU：Triton / cuTile](../experiments/gpu/README.md)
- [CPU：Mojo / Weft](../experiments/cpu/README.md)
- [MLU：DSA / BANG C](../experiments/mlu/README.md)
- [Agent TritonBench](../experiments/agent_tritonbench/README.md)

新增算法示例放在 `kernels/`，相应运行适配与实验数据放在所属实验组。不要在 examples 下重新建立 runner 或 baseline 副本。
