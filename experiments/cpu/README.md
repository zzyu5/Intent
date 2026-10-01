# CPU 算子实验

本组包含 Mojo CPU 与 Weft/RVV/IME。CPU 分支的 lowering、输入复用、布局、私有存储和输出转发实现已合入 main。

- [registry.py](registry.py)：Mojo 199 项、Weft 6 项。
- `providers/{mojo,weft}/`：运行适配、输入、参考与部署 profile。
- `baselines/{mojo,pytorch,weft}/`：原 source/runtime，未改算法、输入或容差。
- [mojo-x86.csv](results/mojo-x86.csv)：原 199 行观察表，187 pass、11 run_only、1 numerical_failed。
- [weft-rvv.csv](results/weft-rvv.csv)：原 6 行 pass。不同 deployment 的 ISA/IME 条件以行与 profile 为准。
- `reports/`：原 CPU 分析资料；其中早期数量是历史快照，当前条目与状态以 registry/CSV 为准。

这些历史数据不是合并后新的全量验收。Mojo 的 FP8 数值失败仍保留，不放宽容差；run_only 不等于已经对照验证。

## Mojo 单点

在已安装 Intent 和 CPU 运行依赖的环境运行，明确指定 Mojo 可执行文件：

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/examples"
export INTENT_MOJO="$HOME/.venvs/intentdsl-mojo/bin/mojo"
python -m experiments.cpu mojo \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --kernel relu --jobs 1 --worker-timeout 600 \
  --output experiments/cpu/results/mojo-single.csv
```

原 runner 保持 8 个 CPU workers 的预算，并设置空闲线程等待策略。去掉 `--kernel` 执行现有 199 项，不扩大矩阵。

## Weft 单点

Intent 与 Weft 使用匹配的 LLVM/MLIR。先配置外部 source/build，再构建同一份 main 编译器：

```bash
export INTENT_WEFT_SOURCE_DIR="$PWD/../TianchenRV-intent-ime"
export INTENT_WEFT_BINARY_DIR="$HOME/.cache/intentdsl/weft-ime-build"
export INTENT_WEFT_COMPILER="$INTENT_WEFT_BINARY_DIR/tools/weft-compile/weft-compile"
cmake -S . -B /tmp/intentdsl-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DINTENT_WEFT_SOURCE_DIR="$INTENT_WEFT_SOURCE_DIR" \
  -DINTENT_WEFT_BINARY_DIR="$INTENT_WEFT_BINARY_DIR"
cmake --build /tmp/intentdsl-build --target intent-compile --parallel 8

export PYTHONPATH="$PWD:$PWD/python:$PWD/examples:$INTENT_WEFT_SOURCE_DIR/python"
python -m experiments.cpu weft \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --kernel i8_gemv_bias --jobs 1 --worker-timeout 600 \
  --output experiments/cpu/results/weft-single.csv
```

`i8_gemv_bias` 使用 [ime.json](providers/weft/ime.json) 的 k1/CPU3/VLEN256/`spacemit-ime1` 条件；普通 RVV 项使用 [rvv.json](providers/weft/rvv.json)。`INTENT_WEFT_PROFILE` 可显式选择现有部署配置，不能把另一台机器的工具链路径直接套用。远端需要 SSH、profile 中的 clang 与运行库；本机生成 artifact，远端编译和执行。

目录整理后的单点结果保存在 `results/integration-mojo.csv`、`results/integration-weft.csv`。

2026-09-22 两个单点均 pass：Mojo `relu` 完成原数值比较；Weft `i8_gemv_bias` 在 k1 上实际选择 `weft.matrix_i8_i32`，产物报告使用 `spacemit-ime1`。
