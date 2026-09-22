# CPU 实现需求、选优与真实算子覆盖阶段报告

本阶段围绕两个实际缺口推进：CPU implementation 的输入与资源需求表达过窄，以及合法实现只按注册顺序选择。随后把这些机制用于更多原始 examples 生产算子，并用原有 runner 做数值与性能核对。

## 已完成的编译器工作

`Implementation` 现在表达 operand 的元素类型、panel 轴与大小、对齐、输入复用组，以及输入的 storage 和 logical 起点。需求满足需要真实 typed SSA、相同 panel 几何、对齐、作用域和稳定性证明；跨消费者复用只在同一可证明的 block 和读快照范围内发生。分组 supply 的 copy、allocation 和最后消费者释放由 CPU lowering 形成，Mojo 隐藏 packing 不再替代显式需求。

`ImplementationRegistry` 会收集每个结构计算的所有适用且合法实现，先构造有限的相关候选组合，再在 bind 阶段重新验证 legality 和参数。首个实现非法时会继续尝试后续实现；RVV、Mojo、Weft 和 IME 候选可以在同一个候选组中竞争。当前仍是有限相关 portfolio，不是所有实现参数的笛卡尔积，也没有把运行时全局最优搜索假称为已经完成。

需求已经接入几类真实结构：转换输入的显式 widening supply、只读/动态 stride 描述、scaled contraction、稀疏 contraction 与独立 output row task、一般 row-major reshape、窄浮点原子 CAS、局部近似 unary 与 FTZ、向量循环中结构不变 supply 提升，以及 Mojo 候选的独立原生编译单元。所有这些决定先进入 CPU IR 或 pass；serializer 和 target emitter 只拼写已经决定的程序。

关键实现提交包括：`f6104eac`（typed supply 与复用）、`71725957`（合法候选与 RVV/IME 竞争）、`f80c9b1c`（scaled/strided ABI）、`a52a669b`（稀疏与 reshape）、`292e50d4`（不变 supply 提升）、`d7c8dd77`（窄浮点 CAS）、`af15a10d`（Mojo 候选分单元编译）和 `2816e4a5`（生产 adapter 扩展）。

## 原始生产覆盖

当前 `examples/kernels` 静态统计为 97 个 Python 文件、228 个 `@intent.kernel` 入口。Mojo registry 有 124 个 case，Weft registry 有 6 个 case；去重后覆盖 143 个 kernel 入口、95 个文件。Mojo 的 123 个 case 通过原有数值检查，1 个 FP8 GEMM 保留 `numerical_failed`；Weft 6 个 case 全部通过。静态覆盖不等于所有 228 个入口都已运行，仍有 85 个入口未覆盖。

完全没有现成原始 production runner 的文件是 `examples/kernels/contraction/vector.py` 与 `examples/kernels/loss/fused_linear_cross_entropy.py`。本阶段没有为它们另造输入矩阵、fixture 或独立测试，因此没有把它们计入覆盖。TritonBench 的 166 个任务也没有在本阶段伪称已经移植或跑通。

这些 case 覆盖了 dense/batched/transposed contraction、ragged/jagged、paged/block sparse attention、linear attention、Mamba、MoE、cache update、scan、histogram、atomic reduction、FP8/NVFP4/MXFP8、Q4/Q8、sparse 2:4、reshape、RoPE、normalization、backward 和多个 variants。每项仍沿用原始 shape、dtype、数据分布、算法和容差；没有通过放宽容差或改写作者 reference 来换取 pass。

## 数值与性能证据

当前生产 CSV 的 Weft 六项均为 pass。最新 i8 GEMV 复查为 generated `500.272931 ms`、source `956.789898 ms`、ratio `0.522866`；i8 GEMM 的既有记录为 `15963.757691 / 31323.657884 ms`、ratio `0.509639`。Q4 RVV 为 `5.926775 / 5.198772 ms`、ratio `1.140034`；Q4 task 的合法候选选择结果为 `14.670117 / 14.331436 ms`、ratio `1.023632`，实际 winner 是 RVV 候选，不能写成 IME 已经追平 reference。

Mojo 的代表性原始生产结果包括：Gemma decode `7.035576 / 8.576771 ms`、paged GQA `30.703387 / 129.090549 ms`、block-sparse GQA `7.365403 / 20.229510 ms`、sparse MLA backward `14308.964499 / 14120.285425 ms`、dense F32 `1.640211 / 2.211787 ms`、LayerNorm backward `64.143932 / 112.274746 ms`、Mamba chunk `10.357069 / 41.963454 ms`、attention backward `37.418803 / 53.170623 ms`。这些结果证明了生成、编译、运行和容差检查已经贯通，但 CPU reference 多数是 PyTorch eager 或 NumPy，不能统一当作强 native library baseline。

仍有清楚的性能边界：paged MLA 约为 source 的 8.41 倍，sparse 2:4 约 10.42 倍，QKV projection 约 2.20 倍，Flash BF16 attention 约 4.21 倍。sparse 2:4 的生成结构仍是 `M -> compressed-K -> N`，在每个 K 步重复读取 lhs/metadata 并写回整个 N；成熟 matmul 结构则把 accumulator 初始化、整个 K 累积和最终 store 分开。这个差异解释了当前慢项，不能用“已经有 RVV load/FMA”代替结构优化结论。

FP8 GEMM 保留失败状态。原始 production case 的首个失败元素为 generated `-16`、reference `-18`。从同一次运行已经存在的 f32 中间值观察到，最快候选为 `-17`，reference 为 `-17.00000762939453125`；FP8 最近偶数舍入跨过 `-17` 中点，造成输出差 2，超过原容差 `0.5 + 0.05 * 18 = 1.4`。这定位了首个失败的放大机制，但没有证明整个算子可以改 pass，因此没有选慢候选、修改算法或放宽容差。

此前记录的 i8 退化也没有被源码证据归因到 compiler 结构。旧缓存与当前缓存的 `host.c`、`kernels.c` 完全相同，`cpu.mlir` 只差函数级 matrix capability 属性；候选参数、tile、task 调用、目标 flags 和测量入口相同。旧 CSV 的 GEMV/GEMM 为约 `211.679 / 7095.383 ms`，当前复查为约 `500.273 / 15963.758 ms`。历史 `kernel.so` 和 winner 细节不可得，因此当前只能把它报告为原生产物或远端运行状态的未归因差异，不对 CPU pass 做无证据改写。

## 是否已经具备编译器形态

可以支持这一表述：作者算法中的区域、遍历、访问、状态、ownership 和数值约束进入 typed IR；CPU pass 根据 def-use、坐标、effects、lifetime 和 target capability 形成 task/block、supply、reuse、materialization 和实现绑定；Mojo/Weft implementation 再展开为可验证的目标程序，最后由 native runtime 执行并按原 reference 检查。区域结构驱动执行构造、implementation 需求驱动计算分组，是论文中最有说服力的主线。Weft IME 的 fragment 解码、拼接和寄存器保留应作为下层 physical compiler 如何兑现这些结构的证据。

当前边界也必须同时写清：任意布局、任意符号 stride、跨控制流的全局资源规划、所有实现参数的全局搜索、所有 TritonBench 任务和全部 228 个 kernel 入口都没有完成。性能 pass、数值 pass、生成代码、编译成功和达到 reference 都是不同结论，不能合并成“CPU 已全面高性能”。

本报告对应分支 `cpu-weft-ime`。本阶段最后一次原始 i8 复查记录提交为 `a7f0bf9d`；本阶段没有把未证实的 i8 归因或 FP8 失败改成 pass，也没有把这一轮新提交自动合并到 main。
