# cuTile、TileLang provider lowering 与横向接纳

## 结论

本轮从第三轮已经可运行的 shared GPU Program 出发，建立了 cuTile 和
TileLang 两条真实 provider 链。两家现在都经过显式 provider IR、provider
legalization/verifier 和 terminal serializer，再进入各自 JIT/launch；没有恢复
KIR 回读、旧 Plan、generic `exec_*` 或第二条 executable path。

真实运行证明两条链已经成立，但横向扫描同时暴露出 shared executable program
仍未覆盖的结构。本轮没有把这些结构降成 scalar/serial 慢路径，也没有把我们的
实现缺口写成 target capability 边界。尤其是 2:4 sparse contraction：TileLang
上游有 `T.gemm_sp`，当前缺的是 shared sparse physicalization 和对应的
TileLang-local four-buffer form。

## 第一条纵向链

第一条链选择 cuTile `grouped_gemm`。它同时覆盖 runtime ragged row range、二维
load/store、blocked contraction 和 native `ct.mma`，比纯 pointwise 更能检验
shared/provider 边界。链路为：

```text
same shared GPU Program
→ cuTile formNativeTiles
→ cuTile verifier
→ cuTile terminal serialization
→ cuTile JIT/launch/numerical comparison
```

随后 TileLang 以 dense GEMM 建立第二条链，并用 FP8 GEMM 进一步检验
provider-native operand storage order、transpose 和 dtype-changing copy-out。

## Provider program

### cuTile

cuTile-local IR 只保存真实 cuTile surface 所需的 form：

- `tile_load` / `tile_store`；
- `scalar_load` / `scalar_store`；
- `gather_load` / `scatter_store`；
- `atomic_rmw` 与 `extract_scalar`；
- `mma` / `scaled_mma`；
- native `reduce` / `scan`。

`formNativeTiles` 直接消费 shared `Load/Store/Gather/AtomicRMW/Contract/
ScaledContract/Reduce/Scan`。连续 unit-step access 形成 tile load/store；需要逐
coordinate bounds 的 access 形成显式 gather/scatter；MMA、scaled MMA、reduce
和 scan 都在 legalization 中检查完整 dtype、axis identity、owner、shape 与
combine legality。Serializer 只打印这些 operations 和仍然显式存在的 common
value/control operations。

本轮把 coordinate-domain adoption 从 serializer 前移到 legalization：插入的
`gpu.broadcast` 必须保留 logical axis identity，source extent 只能与目标相等或
为 unit extent。Serializer 不再因为 physical shape 恰好相同而跳过 axis
permutation。Scalar load 使用 bounds-checked `ct.gather`，而不是无检查的慢速
伪支持。

### TileLang

TileLang-local IR 保存了 TileLang surface 真正要求的结构：

- `launch_config`；
- `alloc`、`clear`、`fill`；
- `copy_in`、`copy_out`、`cast_copy_out`；
- `parallel` / `yield`；
- `buffer_load/store` 与 `view_load/store`；
- native `reduce`、`scan`、`gemm`；
- `pipeline`。

Provider pipeline 依次形成 launch configuration、bufferization/native operand
form、GEMM pipeline，并在 serialization 前验证：顶层 launch config 唯一、没有
剩余 fragment SSA、loop carry 已物化、每个 operation 都属于封闭 TileLang
surface。

Bufferization 显式产生 shared/fragment allocation、copy 和 `T.Parallel`
materialization。它不是 serializer 中的临时 allocation，也不是遇到 unsupported
primitive 后的 generic fallback；无法由已支持的 pure producer 或 native
structured op 物化时直接诊断。

FP8 GEMM 暴露了两个真实的 provider form 缺口：

1. external RHS 的 physical storage 是 `[N,K]`，不能按 logical contract 的
   `[K,N]` 重新解释；bufferizer 现在从 typed coordinate provenance 计算
   buffer-axis 到 fragment-axis mapping，并在 `tilelang.gemm` 上显式记录
   `transpose_lhs/rhs`；
2. `f32` MMA accumulator 写入 FP8 output 时不能先建第二个逐元素 FP8 fragment；
   `cast_copy_out` 保留 accumulator layout，并机械投影为 TileLang 原生
   dtype-changing `T.copy`。

修正后生成代码的 RHS shared buffer 为 `[N,K]`，`T.gemm(...,
transpose_B=True)`，并直接 `T.copy(f32_accumulator, fp8_output)`。数值失败由此
从 `max_abs=496` 变为通过。

## Shared GPU Program 的补正

cuTile/TileLang 横向 lowering 迫使 shared 层补齐了以下跨 provider facts；这些
修改不含 provider 分支：

- static view axis 可以没有 runtime dimension ID；dynamic axis 必须有唯一的
  positive dimension identity；
- static/dynamic `dim` 与 domain extent 被物化为 typed physical expressions；
- advanced index、range、broadcast、reduce result/identity 和 contract operand
  保留 current fragment axis map；
- reduction/contract blocking 对 fragment shape、accumulator、tail validity 和
  loop-carried result 做真实 SSA 改写；
- canonical `intent.result_nodes` 是新建 physical value axis identity 的唯一来源，
  identity 使用独立 namespace，避免与 ABI/dimension source identity 碰撞；
- grid verifier 要求每个 grid axis 恰好有一个 `program_id`，不能以未消费的
  launch dimension 重复执行 effect；
- singleton `delinearize` 在后续 blocking pass 消费前不会被普通 dead-value
  cleanup 删除。

这些补正后重跑 Triton，证明 shared 修改没有破坏第三轮链。

## 横向接纳结果

本轮对 37 个 cuTile registry entries、37 个 TileLang registry entries以及
`examples/kernels/` 的相关 structured forms 做了 schema/consumer 扫描；真实
运行仅使用既有 runner 的定向 repro，没有把旧 baseline CSV 当成当前结果。

| 结构 | 当前事实与归类 |
| --- | --- |
| ragged grouped GEMM | cuTile `grouped_gemm` 已 source/JIT/numerical 通过；TileLang 可进入 native GEMM/tuning，但大候选编译成本尚未形成一条本轮完成的数值证据。 |
| scaled contraction | cuTile `block_scaled_gemm` 已通过；TileLang 普通 FP8 GEMM 已通过。`deepgemm_fp8_2xacc` 仍在 shared scaled-contract construction 拒绝非当前 rank/group schema，属于 shared construction gap。 |
| reduce/scan | 两家都有显式 native reduce/scan form。cuTile `recurrent_gated_delta` 的 structured producer graph 尚未被 shared reduction blocking 物化；TileLang `mamba_chunk_scan` 仍被多个 runtime-shaped contraction 的 joint blocking 阻塞。 |
| region fold/scan | shared verifier会拒绝尚未形成 segment traversal、source slice、carry 与 output assembly 的 `RegionFold/RegionScan`。这是 shared executable-program gap，不是两家 serializer gap。 |
| paged/ragged attention | `grouped_flash_decode`、`varlen_gqa_prefill` 等仍停在 shared 多 contraction/range joint blocking；provider 没有从 KIR 名称或 role 重建它们。 |
| sparse 2:4 contraction | `sparse_2to4_gemm` 已到达 shared `SparseContractOp`，但缺 logical-K ↔ compressed/metadata/slot 的 physical relation、blocked metadata representation 和 TileLang-local `gemm_sp`。TileLang upstream 确有 `T.gemm_sp`，因此这是 Intent shared+provider implementation gap。 |
| block-sparse contraction | 当前 shared contraction pass不物化 `SparseContractOp`，TileLang 也没有 four-buffer sparse native op；没有用 dense GEMM 或逐元素 fallback 冒充。 |
| atomic/scatter routing | cuTile 有显式 gather/scatter/atomic forms，`moe_alignment` 已通过。TileLang upstream 有 atomic APIs，但当前 TileLang IR 尚未闭合 atomic/scatter form；这是 provider implementation gap，不是 target unsupported。 |
| runtime while | shared KIR→GPU 可保存 `scf.while`，两家当前封闭 provider surface 尚未接收；registry 没有 runtime-while entry，因此只登记为未验证的 provider coverage gap。 |
| logical workspace / dynamic programming | 当前 registry 没有能独立证明 generic workspace path 的通过项；相关 attention/scan entries先被 shared structured blocking 阻塞，不能据此归因 workspace 或 target。 |
| record、多输出、InOut state | shared record与多结果 reduce/scan schema存在；当前没有覆盖全部组合的 provider numerical evidence，不能把 schema 存在写成横向运行通过。 |

另外，TileLang `sparse_mla_backward` 的 upstream source 在当前设备请求
231424 bytes dynamic shared memory；这是 source/hardware resource 事实，不用它
掩盖 Intent 侧尚未闭合的 sparse/provider paths。

## 节点二与节点三审计

主要结构完成后做了两轮独立审计，并据此修掉：

- region block argument 的 null defining-op 解引用；
- shape 相同但 axis identity 不同的 cuTile broadcast 早退；
- cuTile coordinate broadcast 在 serializer 中重建 axis relation；
- CuTile tile validity/fill 与 MMA operands 缺少完整 physical-domain/axis-owner
  校验；
- TileLang launch config 可以嵌套而 serializer 假定顶层存在；
- TileLang FP8 cast 通过不同 fragment layout 逐元素搬运；
- TileLang GEMM external storage order被 logical operand order覆盖；
- physical grid rank没有完整 program-coordinate coverage。

最终源码检查没有发现 provider 对 canonical KIR、`KernelModel`、
`kernel.nodes`、`intent.result_shapes`、kernel 名称、role 名称或设备型号的读取；
没有旧 `exec_*`、旧 ProgramForms/materializer、compatibility flag、fallback 或
第二条 executable chain。cuTile/TileLang serializer 只消费 current provider
program。仓库 baseline CSV 未修改。

## 真实验证

执行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 ./examples/run/baseline-v2.sh cutile \
  /tmp/intentdsl-cutile-r4-final.csv grouped_gemm
PYTHONDONTWRITEBYTECODE=1 ./examples/run/baseline-v2.sh tilelang \
  /tmp/intentdsl-tilelang-r4-dense-final.csv dense_gemm
PYTHONDONTWRITEBYTECODE=1 ./examples/run/baseline-v2.sh tilelang \
  /tmp/intentdsl-tilelang-r4-fp8-transpose.csv fp8_gemm
PYTHONDONTWRITEBYTECODE=1 ./examples/run/baseline-v2.sh triton \
  /tmp/intentdsl-triton-r4-source-axis.csv dense_gemm
```

观察结果：

| provider / entry | generated | source | ratio | result |
| --- | ---: | ---: | ---: | --- |
| cuTile `grouped_gemm` | 0.750560 ms | 0.635552 ms | 1.180958 | pass |
| cuTile `block_scaled_gemm` | 1.072688 ms | 7.105024 ms | 0.150976 | pass |
| cuTile `moe_alignment` | 0.061808 ms | 0.079072 ms | 0.781667 | pass |
| TileLang `rms_norm` | 0.175648 ms | 0.180224 ms | 0.974609 | pass |
| TileLang `dense_gemm` | 2.853072 ms | 2.333696 ms | 1.222555 | pass |
| TileLang `fp8_gemm` | 1.081600 ms | 1.132544 ms | 0.955018 | pass |
| Triton `dense_gemm` | 2.035120 ms | 2.083840 ms | 0.976620 | pass |

本轮闭合的是两家真实 provider chain 与其已实现 native forms，不是全部横向
structured forms。剩余项的边界已经定位在 shared construction、shared
structured blocking 或明确的 provider-local representation；没有一项以
serializer 猜测、scalar/serial fallback 或笼统 `compile_failed` 隐藏。
