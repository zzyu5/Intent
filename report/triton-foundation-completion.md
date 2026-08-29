# 第五轮补完：Triton 地基修复与真实状态

## 结论

本轮关闭了核查报告点名的三条具体缺口：

- Triton `ProgramGrid` 不再删除 shared `DelinearizeOp` 及其 execution-group
  segment carrier；provider grid 改写后，完整 shared program 仍能被同一个 verifier
  解释；
- ordinary 与 scaled contraction 统一通过一份 shared validity retarget/replay
  实现，原 tail 与作者 scalar residual 不再由 family rewrite 静默丢弃；
- region fold/scan 在采用第一份 source range 作为循环 master 前，先证明所有
  physical source ranges 的 `start/extent/step` lockstep；证明失败时明确拒绝。

这些修复是真实的，但“当前 Triton 路径已经足以作为第六轮可信地基”这一更强结论
没有成立。最终生产链 probe 为：

```text
93 files / 217 kernels
terminal_source                 98
provider_program_verification 116
physical_program                3
```

第五轮核查时 Triton 只有 `5/217` 到达终端源码；本轮消除了全部
`physical kernel program mapping/effect coverage is incomplete`，因此恢复了 `93`
个 specialization。但是当前 54-entry Triton registry 在 5090 上只有 `20` 个 entry
完成真实数值与计时；旧表中 `31` 个 pass entry 现在停在 provider physicalization
或 candidate-contract gate。它们不能写成已经修复，也不能用旧数字冒充当前结果。

H100 侧当前有外部 vLLM 服务持续占用约 `74888 MiB / 81559 MiB`，GPU utilization
曾为 `100%`。没有终止或干扰该进程，因此 H100 54-entry benchmark 没有执行，旧
`triton-h100.csv` 没有被伪装成当前表。本轮已经在独立临时目录构建好当前 compiler，
所以这里剩下的是外部 GPU 占用，而不是仓库或编译环境准备未完成。

## ProgramGrid：从删 carrier 改为保持完整 program

旧实现位于 `lib/Target/Triton/Transforms/ProgramGrid.cpp:119-128`：它创建最多三个
`ProgramIdOp`，直接替换 `DelinearizeOp` results，然后 erase mapping。execution group、
segment offset/length 和 coordinate roles 都附着在被删除的 mapping 上；紧接着
`Legalize.cpp:888-890` 再跑 full shared verifier，因此普通 pointwise kernel 当场丢失
mapping/effect coverage。

当前实现保留 `DelinearizeOp`。`ProgramGrid.cpp:108-124` 只确定 Triton grid extents、
grid order 和 provider program ids；`126-141` 按原 coordinate order、使用 mapping 的
current runtime extents 重建 row-major linear coordinate，再把它接回原 mapping 的
linear operand。mapping 的 results、execution group、segment bounds、roles、effect
coverage 和 consumers 都没有被重建或迁移到旁表。

这不是 verifier 例外。`Legalize.cpp:888-890` 的 full verifier 原样保留。最终 IR 中
同时存在三个 provider `program_id` 和仍带 typed segment attributes 的 shared
`delinearize`；serializer 已经机械消费两者。

定向事实如下：

```text
addcmul_broadcast_bf16  terminal_source
gelu_tanh               terminal_source
ragged_grouped_gemm     terminal_source
```

在 GPU 被外部服务占用前，真实数值/运行验证得到：

```text
dense_gemm    generated 2.059824 ms / source 2.083496 ms / 0.988638x
grouped_gemm  generated 4.742088 ms / source 4.611808 ms / 1.028249x
```

`ProgramGrid` 之外，Triton provider transform 中没有再找到删除
`DelinearizeOp`、program mapping 或 execution-segment attributes 的第二处路径。
`Legalize.cpp:321-435` 的 masked-gather rewrite会建立 safe coordinate、重建 gather、
保留 origin 后再替换旧 result；`527-550` 的 scatter-add rewrite把完整 resource、
coordinates、value、validity和memory order搬入 atomic RMW。两者之后仍在
`891-894` 运行 full shared verifier。当前未闭合的 gather 诊断是 form 覆盖不足，
不是另一处偷偷删除 carrier。

## Validity：统一 replay authority

`materializeRetargetedValidity` 位于
`lib/Dialect/GPU/Transforms/Utilities.cpp:616-682`。它接受：

- rewrite 前的 validity；
- 该 access 的原 logical tail ranges；
- rewrite 后新产生的 physical tail；
- target fragment schema。

它只做三类可证明动作：识别并剥离由原 range-end comparisons 构成的旧 tail；保留
scalar `i1` residual；把 residual broadcast 后与新 physical tail 相与。若 residual
仍依赖旧 fragment shape、又不能被证明为旧 tail，它直接失败，不从 op 邻接、shape
或 kernel 名称猜测含义。

`intent_gpu.physical_tail` 是 compiler-created comparison 的 provenance 标记，而不是
第二份 validity。`PhysicalProgramAnalysis::isTailPredicate` 仍要求 comparison 的 source
range 与 expected logical range一致、终点 scalar expression一致；仅有标记不能把任意
predicate认成 tail。pointwise blocking 和 region slice construction在创建 physical
tail时写这个标记，后续 blocking才能区分“要被新的 tail替换的 compiler predicate”和
“必须保留的作者 residual”。

ordinary contraction 的 lhs/rhs loads 与 result stores，以及 scaled contraction 的
data/scale四个loads与result stores，现已全部调用同一个 helper。此前 scaled path在
接受 scalar validity后只发 row/block/column masks 的语义洞已经不存在。定向编译还抓到并
修掉一个实现生命周期问题：`scf::ForOp` builder callback执行时新 block尚未挂到父 op，
helper不能从该临时 insertion block向上找 kernel；现在从原 range authority取得所属
kernel。修复后 `ragged_grouped_gemm` 不再 compiler crash，scaled contraction也会到达
真实 `tl.dot_scaled` provider legality，而不是在 shared rewrite里崩溃或丢 predicate。

## Region fold/scan：physical lockstep 不再靠第一份 source 假定

`lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp:81-98` 新增的检查只读取 current GPU IR。
它把第一份 source 的第一条 range作为候选 master，然后要求每个 source plan 的每条 root
在 `start/extent/step` 三个 current SSA expressions上等价。fold 和 scan 都在读取
`plans.front()` 构造 loop之前调用该检查。

这条检查没有把 logical extent equality当成 physical equality，也没有用 source name或
rank代替证明。最终 217-kernel probe 中三个 historical FA variants 因 physical sources
不 lockstep 而停在 `physical_program`：

```text
flash_attention_full_causal_stream_fwd
flash_attention_inline_fwd
flash_attention_select_fwd
```

这是 fail-closed 的 correctness结果，不是把错误路径恢复成 pass。它同时说明第五轮的
`217/217 shared` 结论不能继续沿用：那三个程序过去通过，是因为 pass默默采用了第一个
source的 traversal，并非因为多 source physicalization正确。

## 全语料到 Triton terminal source

最终 probe 使用生产 `lower_to_mlir` 和 `toolchain.run_compiler`，逐个处理 93 个文件中的
217 个 specialization，命令没有创建测试目录或额外 runner：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:examples:. \
  /home/kingdom/.venvs/intentdsl-mlir20/bin/python <stdin inventory driver>
```

最终耗时 `86.301 s`。失败按实际首条诊断归类：

- `3`：shared multi-source physical ranges不 lockstep，明确停在 physical program；
- `100`：current fragment仍带 runtime dimension，Triton `tl.arange` 需要的
  compile-time physicalization尚未形成；
- `5`：masked gather没有落成 safe-index Triton form；
- `7`：mutable `gpu.buffer`没有 Triton-local realization；
- `3`：全部 candidate违反 typed fragment legality；
- `1`：scaled-contract不满足当前 `tl.dot_scaled` rank/format/group/axis legality。

后五类合计 `116`，全部停在 provider verifier；没有 terminal serializer failure。
这里没有再出现 shared carrier被 provider transform破坏的失败。`100` 个 dynamic fragment、
`5` 个 gather和`7` 个 mutable buffer是 provider physicalization尚未完成，不是 Triton
语言已证实没有能力；candidate与scaled-contract两类还需分别结合第八轮公平候选和真实
provider primitive约束处理。

## 5090 表与旧表对比

`report/baselinev2/triton-5090.csv` 已写入本轮真实运行结果。最终状态分布为：

```text
pass                                  20
provider_program_verification_failed 25
candidate_contract_failed             7
source_device_resource_gap             1
source_compatibility_gap               1
```

20 个仍 pass 的 entry 中，没有一项从旧表约 `1.0x` 退到 `1.3x` 以上。代表性对比如下：

```text
dense_gemm                  0.987929 -> 0.988638
grouped_gemm                1.027504 -> 1.028249
flash_cross_entropy         0.992943 -> 0.999349
embedding_lookup            0.999862 -> 0.999816
padded_rope_cache_update    0.793948 -> 0.755491
qkv_projection_pipeline     0.943132 -> 0.920019
index_select                0.995763 -> 1.002899
rope_qk                     0.541513 -> 0.536792
causal_conv1d_update        1.075916 -> 1.057292
mamba3_siso_step            1.497076 -> 1.409174
moe_expert_projection       1.008372 -> 1.007923
moe_splitk_expert_projection 2.679961 -> 2.797796
```

`mamba3_siso_step` 和 `moe_splitk_expert_projection` 原来就明显超过 `1.05x`；本轮没有为性能
改 DSL、candidate或leaf。后者从 `2.68x` 到 `2.80x`，但不是从正常区间塌到 `1.3x`，也
没有证据指向本轮三条 structural修复，按本轮边界保留给第八轮。

旧表的 `31` 个 pass entry现在不再 pass：其中 `24` 个停在 provider physicalization，
`7` 个停在 candidate-contract gate。另一个 `modern_flash_attention_forward` 旧表本来就是
source resource gap，当前更早停在 provider physicalization。完整逐项状态在 CSV 中保留，
没有沿用旧 performance number。

运行期间本机出现外部 root `sgl-omni` 服务反复启动并占用接近全部 5090显存。第一次完整
run中因此产生的 worker/adapter OOM 已在 GPU释放窗口逐项复测；能复测的结果已合并。最终
`fused_softmax` 已确认不是 OOM，而是同一 dynamic-fragment provider gap。最后一次 grid
runtime recheck再次被该服务抢占，因而 CSV 中已有数值来自本轮完整 run；最终 runtime-extent
linearization之后的 production compile probe已重跑并保持同样 `98/116/3` 分布。

## H100 状态

H100 未更新，旧 `report/baselinev2/triton-h100.csv` 仍是历史表，不能视为当前 HEAD结果。

实时证据为：

```text
PID 2173100  VLLM::EngineCore  74888 MiB
GPU memory                    74897 / 81559 MiB
```

远端现有项目工作树有 39 项修改，未触碰。当前本地工作树已复制到独立目录
`/tmp/intentdsl-round5-foundation-P0KJYP`，并在独立 build目录
`/tmp/intentdsl-round5-foundation-P0KJYP-build` 使用远端 LLVM `20.1.8` 完成
`intent-compile` 构建。H100 baseline仍被持续运行的外部服务阻塞；在不终止或干扰它的约束下，
没有可诚实产生当前 H100数字的路径。

## 与 ref 的具体对照

Triton `ref/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp:685-707` 的
`replaceForOpWithNewSignature` 建新 loop时复用原 lower/upper/step，在 body splice后复制
全部 attributes。它说明结构变换的正常形态是把旧 carrier 的完整语义搬到新 carrier，
而不是替换 users后删除唯一承载物。Intent这次选择更窄：不创造第二种 provider carrier，
直接保留 shared `DelinearizeOp`，只改 linear program-id producer。换一个带execution group
或segment offset的kernel时，这一差别直接决定full verifier能否继续建立coverage。

TileLang `ref/tilelang/src/transform/materialize_kernel_launch.cc:63-96` 将launch loop转为
thread extent或serial loop时保留原 min、extent、loop var、annotations和step；它同样没有
把body需要的execution coordinate换成无语义连接的裸值。

Triton `PipelineExpander.cpp:315-375` 为dynamic pipeline stage显式构造 `iv < ub` predicate，
每个clone都经过predicate function；外部可见的loop-carried result还用`select`保持旧值。
`SoftwarePipeliner.cpp:146-163` 只在mask predicate被证明为constant false时删除mask，并明确
说明更宽删除会错误地允许speculative async copy写shared memory。对应到Intent，本轮不能因
新blocking已有tail就丢弃旧validity；统一helper只替换被证明为旧tail的部分，scalar residual
继续进入新access。

多source方面，TileLang `copy_op.py:23-50` 对两个buffers先要求structural-equal shape，再将
extent交给`utils/language.py:468-511`的pairwise legalization；它不会默认用第一份source
extent。Intent region fold/scan的语义要求比copy更强，因此本轮采用更保守的规则：current
physical `start/extent/step`不全等就拒绝。若换一个source extent或step，旧代码静默沿master
运行，新代码在创建loop前停止；这是可观察的correctness差别。

## 剩余事项归属

核查报告中的剩余项不能因为本轮报告结束而消失：

- tuple/record buffer element、scale-axis relation作者表面、一般calling preconditions属于
  canonical DSL/KIR设计与实现轮；它们不是第六/七轮provider拼写问题；
- multi-contract joint realization与runtime free-axis属于shared GPU physicalization，必须在
  依赖它们的provider覆盖前闭合；
- mutable buffer的logical lifetime/ownership属于shared authority，Triton/cuTile/TileLang的
  storage realization分别属于各provider轮；当前Triton也尚未完成；
- sparse contract已有shared semantic op，但compressed value/metadata relation与三家provider
  consumer尚未闭合；不能归为某门语言天然unsupported；
- cuTile mapping/tile-index/provider legality留给第六轮；
- TileLang bufferization/storage/copy/sync/pipeline legality留给第七轮；
- candidate set、winner、measurement scope以及已有 `>1.05x` 性能项留给第八轮。

因此，第六轮开始前仍有一项明确阻塞：Triton registry的dynamic fragment、safe gather和
mutable buffer physicalization尚未恢复旧能力。本轮没有用更窄family规则临时补绿，也没有
把这项事实藏进“provider capability”。
