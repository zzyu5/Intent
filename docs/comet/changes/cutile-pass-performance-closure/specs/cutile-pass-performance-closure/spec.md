# cuTile Pass-Based Performance Closure

## 目标与范围

本能力通过现有 Intent compiler、作者 kernel 和 cuTile provider 的通用实现，解决当前 cuTile 性能缺口。它要求有效的 IR 优化与逐项性能结果同时成立，而不是只完成调参、改名或可执行路径闭合。下表固定当前 37 个 entry 的 case；名称只标识运行范围，不参与 compiler policy。

| Entry | 固定 case | 达标设备 |
| --- | --- | --- |
| official_fmha | B4-QH32-KVH8-S4096-D128-fp16-causal | 两台 |
| block_scaled_gemm | M4096-N14336-K4096-fp8-block32 | RTX 5090D |
| dense_gemm | M4096-N14336-K4096-fp16 | 两台 |
| moe_expert_projection | T4096-E8-top2-4096x14336-bf16 | 两台 |
| layer_norm | 8192x4096-bf16 | 两台 |
| silu_and_mul | 4096x28672-to14336-bf16 | 两台 |
| attention_backward | B2-QH8-KVH2-S1024-D64-fp16-causal | 两台 |
| dense_attention_forward | B2-QH32-KVH8-S4096-D128-bf16-causal | 两台 |
| grouped_flash_decode | B8-QH32-KVH8-S8192-D128-bf16 | 两台 |
| swiglu | 8192x14336-bf16 | 两台 |
| splitk_attention_reduce | B8-H32-S8192-splits16-D128 | 两台 |
| mla_prefill | B1-QH128-KVH1-S2048-D128-R64-fp16 | 两台 |
| batched_gemm | B32-M512-N512-K1024-bf16 | 两台 |
| tilegym_dense_gemm | M8192-N11008-K4096-bf16 | 两台 |
| grouped_gemm | rows256-512-1024-2048-K4096-N4096-bf16 | 两台 |
| moe_alignment | T4096-top2-E64 | 两台 |
| mhc_gemm_rms_scale | T2048-H4096-streams4-bf16 | 两台 |
| chunked_softmax | 8192x32768-bf16 | 两台 |
| rope_qk | B2-S4096-QH32-KVH8-D128-bf16 | 两台 |
| gelu | 8192x4096-fp16-tanh | 两台 |
| geglu | 4096x28672-fp16-tanh | 两台 |
| relu | 8192x4096-fp16 | 两台 |
| dropout | 8192x4096-fp16-p0.1 | 两台 |
| attention_sink_prefill | B1-S4096-QH32-KVH8-D128-bf16 | 两台 |
| attention_sink_decode | B32-S8192-QH32-KVH8-D128-bf16 | 两台 |
| mhc_apply_residual | T2048-H4096-streams4-bf16 | 两台 |
| gemma_prefill | B2-S4096-QH32-KVH8-D128-window1024-cap50-bf16 | 两台 |
| gemma_decode | B32-S8192-QH32-KVH8-D128-window1024-cap50-bf16 | 两台 |
| mhc_sinkhorn | T8192-streams4-fp32 | 两台 |
| absorbed_mla_decode | B8-H64-S8192-C512-R64-fp16 | 两台 |
| splitk_mla_decode | B8-H64-S8192-C512-R64-split512-fp16 | 两台 |
| sparse_mla_prefill | S2048-SKV4096-H64-topk512-D128-R64-bf16 | 两台 |
| sliding_window_attention | B2-S4096-QH32-KVH8-D128-window1024-fp16 | 两台 |
| rms_norm | 8192x4096-bf16 | 两台 |
| recurrent_gated_delta | B2-S2048-H8-K128-V128-bf16 | 两台 |
| chunk_gated_delta | B2-S2048-H8-K128-V128-C64-bf16 | 两台 |
| nvfp4_quantize | 8192x4096-block16-bf16 | RTX 5090D |

“两台”指现有 RTX 5090D 与 H100，不直接比较两台机器的绝对时间。硬目标包含 37 个 RTX 5090D 条目和 35 个 H100 条目，共 72 项；不随 CSV 中哪个条目当前超标而改变集合。

两个既定 H100 限制为：`block_scaled_gemm` 所需 E8M0 scaled MMA 不受 SM90 支持；`nvfp4_quantize` 的现有 source 使用 SM90 不支持的原生 FP4 转换/打包。两项在 H100 表中保留真实状态和说明，不参与性能比值门槛，也不在本 change 新增其硬件支持。其余条目的失败、超时、空值或超标不能成为新的豁免。

## 通用 compiler 优化

以 current Physical Program 的 typed semantics、def-use、坐标与访问关系、effects、alias、reuse/lifetime、physical facts 和能力为依据组织优化。Program mapping、ownership、fragment 分块、循环携带状态、矩阵依赖与合流、共同复用和物化关系由相应 shared analyses/passes 处理；provider-local form 与合法性留在对应 provider pass；线程/warp 布局、机器流水、寄存器分配等下层事实由已有 provider compiler 负责。

Compiler 可以作出合法性与收益选择，但必须落实为当前 IR 中的 types、operations、regions、def-use、访问或循环改写。不能按 kernel/source 名、whole-operator template、op 数量、CSV winner 或 serializer 字符串选择另一个执行实现。需要 local form 候选时，先形成当前程序中完整、合法、可编译的表示，不在 serializer 临时构造算法、workspace 或控制流。

复用现有 analysis、typed carrier 和 transformation groups；只有当前程序确实缺少必要执行事实时才补充其表达。不要求每个问题新建一个 pass，也不把所有 provider 优化机械上移。实际修复覆盖所有适用位置和已有 Triton/cuTile/TileLang 消费者；其他 target 不承担本 change 的全表性能门槛。

对照环境中的 `ref/triton` 或 `ref/tilelang` 真实 compiler 实现，说明双方 file:line、前提、IR 改写与实际后果。作者例子可用于理解算法或存储组织，但不能独自证明 compiler pass 的机制。已知差异只是归因线索，不能把未经证明的假设固定为方案。

## 作者表达与数值边界

作者 kernel 可以显式改善不合理表达和中间精度，不冻结已有作者写法，也不让 serializer 隐式改变精度、dtype、NaN 或累加语义。本 change 增加已获用户确认的逐操作近似数学与 FTZ：`I.fdiv`、`I.exp2` 与 `I.tanh` 的显式近似选择，以及近似除法/exp2 的 FTZ 选择；支持范围与闭合数值契约由 `doc/dsl/` 定义。外部 ABI、原容差和性能门槛不变，除以下已确认的局部融合外，其它语言变化仍需单独澄清。

普通零初值 contraction 的结果只有一个同 dtype 加法 consumer 时，允许将另一 operand `C`接入contraction accumulator，实现 `contract(A, B) + C` 或 `C + contract(A, B)` 的局部融合。融合后的舍入、特殊值与空 reduction 规则由 DSL 数值规格定义；shared pass从current def-use、零初值、结果坐标关系与dominance判定并直接改写IR，所有provider消费同一结果。不得跨数值cast或多个consumer，不把该许可扩展为非零初值的重复重结合、全局fast-math、输入精度下降、FTZ或kernel-name/tuning策略。

近似与 FTZ 必须成为 canonical unary/binary operation 的 typed attributes，并随 construction、cloning、blocking、summary 改写、bufferization 和 provider lowering 保留；它们不进入 shape、kernel ABI、候选配置或 kernel-name policy。不同数值属性的运算不能被当作同一纯值合并；近似选择不授权改变相邻普通运算、累加 dtype、数据依赖或 source order。Provider 对不支持的 dtype/primitive 组合明确拒绝，不能静默改回另一种数值语义。

Generated/source 使用相同算法、固定 case 和外部 dtype。舍入位置、近似数学、FTZ 和中间精度的细微差异允许注明后比较，不要求逐操作或 bitwise 一致；原有 entry 容差不放宽，真实 NaN/Inf 或容差外错误必须修复。若确有算法变化，保留已有 Triton 使用或已对齐的算法；没有 Triton 使用的 cuTile 专用算法可向 baseline 对齐，但不得偷换完整 callable 的功能或计时范围。

## 有效调优与 JIT

Passes 形成合法的参数化程序，有限候选绑定 shared granularity 与 provider 编译选项，下层编译将候选落实为具体程序，provider tuner 按实测选择并复用 winner。参数可以驱动相应 passes 改变具体分块、布局与流水；缺少 transformation 时，增加候选不能替代实现。

保留 provider JIT/autotuner、适用的 compiled-candidate cache 和 winner cache，不固定历史 winner、不取消必要 specialization，不将所有程序无依据地收缩为统一参数。删去可证明非法、重复或无效的组合；有限 profile 是候选数据，不是按算子名选择的策略库，也不是任意 IR 图结构的笛卡尔积搜索。

双方可独立调优，不要求相同候选或相同 winner。不得通过限制 source 的合理调优、增加其额外工作或改变比较口径制造比值优势。准备和编译在资源预算内并发；provider 调优本身的 GPU 测量以及最终计时都避免被其他 GPU 工作污染，不把计时隔离扩大成所有 worker 从准备到结束全程串行。

## GPU 计时与性能结果

每项硬目标为 `generated_p50_ms / source_p50_ms <= 1.05`。两侧测量的是完成既定 callable 的 GPU 执行延迟，不是 CPU launch API 调用耗时，也不是编译、JIT、候选搜索、graph capture 或单独的准备成本。既有多-kernel 算子计完整 GPU pipeline，不能只挑其中最快的 kernel；辅助输出、布局转换与 workspace 的计时范围如实说明。

CUDA Graph 是组织重复执行的方式，CUDA event 是设备时间测量工具，两者并不互斥。沿用现有 entry 的 Graph/Event 适用条件，双方保持一致，不为提高比值单边换计时方法。NVFP4 当前采用 CUDA Graph 重放并以 CUDA event 计时，输出 buffer 在计时前准备；对应 adapter 注释与项目表应明确表述为“CUDA Graph 重放下的算子 GPU 执行延迟，不含编译、调优与输出分配”，不将其含混写成“一次 kernel launch”。

结果持续写入 `report/baselinev2/cutile-5090.csv` 与 `report/baselinev2/cutile-h100.csv`。保留全部 entry、case、两侧时间、ratio 和真实状态；CSV 中 `pass` 表示该次比较完成，不自动等于达到 1.05。每项完成即更新该项，不等待全量结束，不增加候选耗时或运行历史审计列。

## 验收与交付

验收仅使用 brief 的 A1—A3：通用 compiler 实现、有效有限调优、全部固定性能结果。任何一个硬目标失败或超标，都不能以平均值、其他条目更快、局部收益或代码提交替代完成。

只通过现有 `examples/run/baseline-v2.sh` production 入口执行取得性能所需的 emit、编译、预热、GPU 运行与同次原容差检查。不建立另一套测试、临时脚本、fixture、pytest、性能框架或输入矩阵；现有代码与 reference 对照直接服务于实现，不另建调查报告。

完整覆盖可以逐项完成并续跑。未变化且仍适用的已有结果可复用；受 compiler program、provider form 或计时路径改动影响的条目更新其实际结果，不把旧值当作新程序已达标的证据。环境资源不足或外部负载干扰时保留未完成状态；需要外部协调的测量障碍不自动阻断其他已授权、无此依赖的实现工作。

本 change 使用独立 worktree，接续旧 `compiler-cross-target-optimization` 的已提交工作与剩余问题；旧 change 不因本任务建立而通过或归档。保留已有文档权威和目录边界，只提交必要实现、项目性能表与 Comet 正式产物。局部提交和一次 benchmark 结束不是整个 change 的终点；直到上述结果完成或出现明确的外部/用户决策阻塞才交回。
