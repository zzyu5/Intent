# 实验

四组实验分别保存执行入口、baseline、结果及与本组有关的分析：

| 目录 | 范围 | 入口 |
|---|---|---|
| [gpu](gpu/README.md) | Triton、cuTile；保留 TileLang | `python -m experiments.gpu <provider>` |
| [cpu](cpu/README.md) | Mojo、Weft/RVV/IME | `python -m experiments.cpu <provider>` |
| [mlu](mlu/README.md) | DSA → BANG C / MLU | `python -m experiments.mlu bangc` |
| [agent_tritonbench](agent_tritonbench/README.md) | Intent 与 agent Triton 的单次生成实验 | `python -m experiments.agent_tritonbench` |

`_common/` 复用原生产 runner 的准备、数值比较、计时和 source 加载实现；各组的 `registry.py` 与 `providers/` 保存具体算子接线。没有另建实验调度框架。`examples/kernels/` 仅保存作者算法。

## 构建与调用

从仓库根目录执行，Python 使用所选后端的既有环境：

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/examples"
cmake -S . -B /tmp/intentdsl-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build /tmp/intentdsl-build --target intent-compile --parallel 8
```

Weft 还需要 CPU 目录说明中的两个 CMake 路径。也可用 `./experiments/run.sh <provider> <output.csv> [kernel ...]` 完成构建和运行；`INTENT_PYTHON`、`INTENT_BUILD_ROOT`、`INTENT_BUILD_JOBS`、`INTENT_TUNING_CONFIG` 可以显式指定环境与已有配置。

准备/编译允许并发，原 runner 在同一批次中串行放行性能计时；不要同时启动争用同一设备的多个独立性能任务。`--kernel` 选择现有 registry 项，不修改输入规模、算法、dtype、reference 或容差。

## 结果规则

- 常规算子 CSV 字段为 `kernel,case,generated_p50_ms,source_p50_ms,ratio,status,note`；`ratio=generated/source`，越小越快。无 reference 的运行和没有同设备 source 时间的 MLU 结果不能补造比值。
- 已迁入的历史表保留原值；它们不代表本次合并后全量重测。`integration-*.csv` 仅记录目录整理后的单点验证。Agent 实验保留自己的结果 schema 和原始首次成绩。
- 后续全量、单点及 pass 效果实验均写入所属组的 `results/`。不同机器或前后对照使用不同文件；操作、参数和解释放在同组目录，需要正式分析时放入该组 `reports/`。
- 单点输出建议单独命名；runner 会按 kernel 更新指定 CSV 中的对应行，避免误把单点结果当新全量。
- 编译、首次 JIT 和 tuning 不计为算子时间。多 kernel callable 按原合同计完整 GPU/native 调用；CUDA Graph 与 CUDA event 的 host 开销范围不同，具体以原 entry 合同为准。
- CSV 是运行观察，不定义 DSL、compiler policy 或性能通过阈值。比较差距时先看算法与实际物理结构，不改题面、输入或容差来获得较好数字。
