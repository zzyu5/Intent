# MLU / DSA 算子实验

本组使用 KIR → DSA → BANG C → CNCC/CNRT。DSA dialect、lowering、serializer 和 runtime 保留在编译器目录，实验适配与记录放在这里。

- [registry.py](registry.py)：13 个原生产 case，含显式多 kernel 序列。
- `providers/bangc/`：输入准备、编译、远端运行和数值比较。
- [bangc-mlu370.csv](results/bangc-mlu370.csv)：原完整 13 项结果，全部 pass。
- baseline 数值参考继续复用 GPU/CPU 组的既有 source，不复制成另一份算法。当前没有同机手写 MLU baseline 时间，`source_p50_ms`、`ratio` 留空；generated 时间是 CNRT notifier 的设备时间，不含传输与 CNCC 编译。

## 环境与部署

使用已经配置的 MLU SSH 环境，设置 `INTENT_BANGC_HOST`、`INTENT_BANGC_ROOT`、`INTENT_BANGC_PYTHON`、`INTENT_BANGC_DEVICE`。远端要求 MLU370/CNToolkit、CNCC、CNRT，Python 中可导入项目 runtime。已验证环境为 CNCC 4.7.2、CNRT 6.7.0。

更新远端 runtime 时，从仓库根目录执行：

```bash
tar --exclude=__pycache__ --exclude='*.pyc' -C python -czf /tmp/intentdsl-bangc-runtime.tar.gz intent
scp /tmp/intentdsl-bangc-runtime.tar.gz "$INTENT_BANGC_HOST:/tmp/"
ssh "$INTENT_BANGC_HOST" "mkdir -p '$INTENT_BANGC_ROOT/python' && tar -xzf /tmp/intentdsl-bangc-runtime.tar.gz -C '$INTENT_BANGC_ROOT/python'"
```

账号、凭据和部署配置保留在现有外部环境，不放进实验结果。

## 单点运行

本机使用能运行原 cuTile/Triton reference 的环境；包含 Mojo reference 的项还需 `INTENT_MOJO`。以下从仓库根目录执行：

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/examples"
export INTENT_MOJO="$HOME/.venvs/intentdsl-mojo/bin/mojo"
python -m experiments.mlu bangc \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --kernel relu --jobs 1 --worker-timeout 600 \
  --output experiments/mlu/results/bangc-single.csv
```

去掉 `--kernel` 运行原 13 项。执行前确认目标 device 可用；目录整理后的单点单独保存在 `results/integration-bangc.csv`，不冒充完整重测或性能达标。

2026-09-22 使用合并后的 main 编译器与已同步的远端 runtime，MLU370 上的 `relu` 单点数值比较与 CNRT 执行均 pass。
