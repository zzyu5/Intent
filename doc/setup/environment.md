# 环境、依赖与构建

本文给出从干净机器建立 Intent Kernel DSL 开发环境的唯一推荐方式。虚拟环境必须放在仓库外；仓库只保存直接依赖版本，不保存 `.venv`、编译缓存、JIT cache 或 CUDA 工具链副本。

## 1. 为什么是三个 Python 环境

Triton、cuTile、TileLang 是同一个 GPU Physical Plan 的三个 surface，但它们的 Python 包、Torch/Triton 版本和 CUDA 编译工具链并不一致。把三者装进同一个环境会让一个 provider 的升级替换另一个 provider 的依赖，并使性能变化无法归因。

推荐结构是：

```text
仓库外的环境根目录/
├── triton/    Triton generated code 与 Triton upstream baseline
├── cutile/    cuda.tile、TileGym 与 cuTile upstream baseline
└── tilelang/  TileLang generated code 与 TileLang upstream baseline
```

`environment/triton.txt`、`environment/cutile.txt`、`environment/tilelang.txt` 是三个环境的直接 Python 依赖权威来源。Torch 的 CUDA wheel 单独安装，因为不同机器可能需要不同的官方 wheel index；不要把本机环境路径写进脚本或复制到仓库。

## 2. 当前实际使用的版本

### 2.1 C++/MLIR 构建工具链

| 组件 | 当前使用 | 项目合同 |
|---|---:|---|
| Python | 3.10.12 | 三个 provider 环境都使用 Python 3.10 |
| LLVM | 20.1.8 | major 20；提供 LLVM CMake package 与 headers/libraries |
| MLIR | 20.1.8 | major 20；提供 MLIR CMake package、TableGen 与 libraries |
| Clang | 20.1.8 | C++17 host compiler |
| CMake | 3.22.1 | 项目最低要求 3.20 |
| Ninja | 1.10.1 | 推荐 generator |

项目的 Python pipeline 调用 C++ `intent-compile`；它不依赖 PyPI 上名为 `mlir` 的包，也不要求 MLIR Python binding。真正的 build dependency 是 LLVM/MLIR 20 的 CMake package。

### 2.2 Provider 环境

| 环境 | RTX 5090 当前实测 | H100 当前实测 | 固定的 provider 版本 |
|---|---|---|---|
| Triton | Torch 2.10.0+cu130，Triton 3.6.0 | Torch 2.10.0，`torch.version.cuda=12.8`，Triton 3.6.0 | Triton 3.6.0，NumPy 1.26.4 |
| cuTile | Torch 2.13.0+cu130，Triton 3.7.1 | Torch 2.13.0，`torch.version.cuda=13.0`，Triton 3.7.1 | `cuda-tile` 1.5.0，TileGym 1.4.0，CUDA Toolkit Python meta-package 13.3.1 |
| TileLang | Torch 2.10.0+cu130，Triton 3.6.0 | Torch 2.10.0，`torch.version.cuda=12.8`，Triton 3.6.0 | TileLang 0.1.13，Apache TVM FFI 0.1.12；wheel 内含 TVM 0.25.dev0 |

cuTile 当前 pip 工具链中，`nvidia-cuda-tileiras` 为 13.3.36，`nvidia-cuda-nvcc` 为 13.3.73。它们由 `cuda-toolkit[tileiras,nvvm,nvcc]==13.3.1` 的依赖合同解析，不需要逐个写进项目代码。

当前两台机器都使用 r580 系列驱动；cuTile 1.5.0 的官方要求也是 NVIDIA driver r580 或更新版本。Driver 报告的“CUDA Version”、Torch wheel 自带的 CUDA runtime、cuTile pip 工具链和 TileLang 使用的 NVCC 是四件不同的事，不能把它们压成一个“项目 CUDA 版本”。

## 3. 安装系统依赖

Ubuntu/Debian 示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10 python3.10-venv \
  cmake ninja-build \
  clang-20 llvm-20 llvm-20-dev llvm-20-tools \
  libmlir-20 libmlir-20-dev mlir-20-tools
```

如果发行版仓库没有 LLVM/MLIR 20，应按照 [apt.llvm.org](https://apt.llvm.org/) 配置 LLVM 官方 Debian/Ubuntu 仓库，或者从同一 LLVM 20 构建中安装 LLVM 与 MLIR。不要混用不同 major 的 LLVM headers、MLIR libraries 和 `mlir-tblgen`。

安装后确认：

```bash
/usr/lib/llvm-20/bin/llvm-config --version
/usr/lib/llvm-20/bin/mlir-opt --version
/usr/lib/llvm-20/bin/mlir-tblgen --version
cmake --version
ninja --version
```

当前项目默认的 CMake package 位置是发行版常见布局：

```text
/usr/lib/llvm-20/lib/cmake/llvm
/usr/lib/llvm-20/lib/cmake/mlir
```

若安装在其他位置，通过 `INTENT_LLVM_DIR` 和 `INTENT_MLIR_DIR` 显式传入，不要创建仓库内软链接。

## 4. 创建隔离的 Python 环境

下面的路径只是占位符，必须替换成仓库外的绝对路径。`venv` 默认不继承 system site-packages；不要启用 `--system-site-packages`，否则某个环境可能静默借用用户级 Torch、Triton 或 TileLang。

```bash
INTENT_ENV_ROOT=/absolute/path/outside/repository/intentdsl-envs
mkdir -p "${INTENT_ENV_ROOT}"

python3.10 -m venv "${INTENT_ENV_ROOT}/triton"
python3.10 -m venv "${INTENT_ENV_ROOT}/cutile"
python3.10 -m venv "${INTENT_ENV_ROOT}/tilelang"

for provider in triton cutile tilelang; do
  PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/${provider}/bin/python" \
    -m pip install --upgrade pip setuptools wheel
done
```

### 4.1 选择 Torch CUDA wheel

当前 RTX 5090 与 cuTile profile 使用 CUDA 13.0 wheel：

```bash
PYTORCH_INDEX=https://download.pytorch.org/whl/cu130

PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/triton/bin/python" \
  -m pip install --index-url "${PYTORCH_INDEX}" 'torch==2.10.0'
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/tilelang/bin/python" \
  -m pip install --index-url "${PYTORCH_INDEX}" 'torch==2.10.0'
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/cutile/bin/python" \
  -m pip install --index-url "${PYTORCH_INDEX}" 'torch==2.13.0'
```

H100 的当前 Triton/TileLang 固定表来自 CUDA 12.8 Torch wheel；需要复现实测环境时，把前两条的 index 改成 `https://download.pytorch.org/whl/cu128`。cuTile 当前两台机器都使用 CUDA 13.0 Torch wheel与独立的 13.3 TileIR 编译工具链。

Torch wheel 的选择必须与设备、driver 和下层 provider 支持范围一致。`torch.version.cuda` 是实际确认值；不要用 `nvidia-smi` 顶部显示的最高 CUDA 版本替代它。

### 4.2 安装 provider 依赖

从项目根目录执行：

```bash
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/triton/bin/python" \
  -m pip install -r environment/triton.txt
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/cutile/bin/python" \
  -m pip install -r environment/cutile.txt
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/tilelang/bin/python" \
  -m pip install -r environment/tilelang.txt
```

cuTile 的 distribution 名是 `cuda-tile`，Python import 是：

```python
import cuda.tile as ct
```

不存在项目所依赖的 `import cutile` 包。TileLang 则使用 `import tilelang`；它的 TVM Python module 随 TileLang wheel 提供，不应另外安装一个不匹配的 `apache-tvm` wheel。

### 4.3 TileLang 的 CUDA compiler

TileLang 除 Torch runtime 外，还需要能识别目标 GPU 架构的 CUDA compiler。可选两种方式：

1. 安装 host CUDA Toolkit，并把其 `bin` 加入运行环境；
2. 按 [TileLang 官方安装文档](https://tilelang.com/get_started/Installation.html) 使用 pip-provided CUDA toolchain，例如额外安装 `tilelang[nvcc]==0.1.13`。

使用 host toolkit 时，应显式选择正确版本：

```bash
CUDAToolkit_ROOT=/absolute/path/to/cuda-toolkit
PATH="${CUDAToolkit_ROOT}/bin:${PATH}" \
  PYTHONNOUSERSITE=1 \
  "${INTENT_ENV_ROOT}/tilelang/bin/python" -c \
  'import tilelang; print(tilelang.__version__)'
```

当前 H100 主机的默认 `/usr/bin/nvcc` 是 11.5，不能编译 `sm_90a`；固定表运行时显式选择了可识别该架构的 CUDA 12.2。RTX 5090 的 TileLang 路径使用 CUDA 12.8。这个选择属于运行环境，不允许转化成按设备型号修改 Kernel IR 或 Physical Plan 的分支。

cuTile 不依赖这个 host NVCC：`environment/cutile.txt` 已通过 CUDA Toolkit Python meta-package安装其 TileIR、NVCC 和 NVVM 组件。

## 5. 核对安装结果

```bash
PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/triton/bin/python" -c \
  'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)'

PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/cutile/bin/python" -c \
  'import torch, triton, cuda.tile as ct; print(torch.__version__, torch.version.cuda, triton.__version__, ct.__version__)'

PYTHONNOUSERSITE=1 "${INTENT_ENV_ROOT}/tilelang/bin/python" -c \
  'import torch, triton, tilelang; print(torch.__version__, torch.version.cuda, triton.__version__, tilelang.__version__)'
```

如果输出的包路径落到用户级 `site-packages`，环境没有真正隔离。应重建 venv，而不是继续补 `PYTHONPATH` 或在仓库内复制 wheel 内容。

## 6. 构建 `intent-compile`

```bash
INTENT_BUILD_ROOT=/tmp/intentdsl-build
INTENT_MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
INTENT_LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm

cmake \
  -S . \
  -B "${INTENT_BUILD_ROOT}" \
  -G Ninja \
  -DMLIR_DIR="${INTENT_MLIR_DIR}" \
  -DLLVM_DIR="${INTENT_LLVM_DIR}"
cmake --build "${INTENT_BUILD_ROOT}" --target intent-compile
```

Build 目录位于 `/tmp` 或其他仓库外目录。顶层 CMake 使用 C++17，并链接 Intent dialect、GPU realization、三个 emission library 以及 MLIR parser/IR/support。

## 7. 运行一条真实 repro

`examples/run/repro.sh` 是唯一公共验证入口。外部环境必须用 `INTENT_PYTHON` 显式传入，不依赖维护者机器上的默认路径：

```bash
INTENT_BUILD_ROOT=/tmp/intentdsl-build \
INTENT_MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
INTENT_LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
INTENT_PYTHON="${INTENT_ENV_ROOT}/triton/bin/python" \
PYTHONNOUSERSITE=1 \
./examples/run/repro.sh triton softmax

INTENT_PYTHON="${INTENT_ENV_ROOT}/cutile/bin/python" \
PYTHONNOUSERSITE=1 \
./examples/run/repro.sh cutile softmax

INTENT_PYTHON="${INTENT_ENV_ROOT}/tilelang/bin/python" \
PYTHONNOUSERSITE=1 \
./examples/run/repro.sh tilelang softmax
```

每条命令都会重新确认 `intent-compile` 可构建，然后执行完整的 DSL→canonical Kernel MLIR→Physical Plan→target source→下层 JIT→真实 GPU 数值对照。不要用“能 import provider”替代这条闭环。

## 8. 版本升级纪律

升级某个 provider 时只改对应的 `environment/*.txt` 和这份支持矩阵，不把目标 API 变化塞入共享 Plan。升级后先跑一条真实 repro；若只是 API 名或参数形式变化，修改该 target 的 `Emission/Syntax/Spelling`；若 provider 明确缺少能力，capability check 在 emission 前拒绝；只有真实算法无法用现有 Core 表达且也不能委托给下层时，才讨论修改语言或 Kernel IR。

官方安装入口：

- [Triton installation](https://triton-lang.org/main/getting-started/installation.html)
- [cuTile Python quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)
- [TileLang installation](https://tilelang.com/get_started/Installation.html)
