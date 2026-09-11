---
generated_from_state_version: 5
---

# 验证

## 当前结果

- 结果: **验收通过，需要你确认**
- 验证情况: **已完成检查，但需要你确认验证结果**
- 目标周期: 1
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-11T08:00:39.960Z
- 摘要: 新的独立只读 Verifier /root/cpu_structural_comet_verifier 判定 A1–A4 全部通过；主代理核对关键原文及最终日志，补全 UniformValues 证据为 lib/Analysis 路径。未发现本轮必要的语义、alias/lifetime、provider lowering 或性能证据缺陷；剩余慢项和保守路径如实保留。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：CPU region transformation 根据当前坐标范围和完整的 typed summary/identity、数值及 effects 证据形成有效遍历与谓词/状态特化；可证明无贡献的区间真实改变循环边界及其专属访问，而非只记录 possibleBegin/End。无法证明的区间保留原计算；空 source、首次 combine、绝对坐标、tail、scan 输出和最终状态不被省略。不得将 mask-false 或一个浮点零操作数单独当作删除整段的依据。 | 独立 Verifier 的直接证据：lib/Dialect/CPU/Analysis/RegionPredicates.cpp:267–327 仅接受非 scan、typed 坐标比较及完整 effects/identity 证明；lib/Dialect/CPU/Transforms/RealizeRegions.cpp:227–301 将已证区间写入真实循环边界与访问，未知证明保留原遍历。lib/Analysis/UniformValues.cpp:265–280 的浮点 contraction 要求两个操作数已知，不能以 0*unknown 删除区域；scan emit、首次 combine 和最终状态保持。 |
| A2 | passed | brief.md | A2：当前 region-scan 的重复准备/状态物化得到实际改写：在同一 slice 的来源稳定、访问一致且生命周期合法时复用输入供应，在旧状态不再被读取时消除不必要的 next→state 全量复制或重复初始化。生成的 CPU/provider IR 能指出减少的具体工作及 owner/lifetime，所有 summarize/emit/apply 依赖和完整 final state 保持；不以全局字符串缓存代替 IR 复用。 | 独立 Verifier 的直接证据：lib/Dialect/CPU/Transforms/FuseStructuredComputations.cpp:283–350 限定 private alloc/alloca 目标，并检查唯一完整 writer、映射、alias、dominance 和中间 effects；lib/Target/Weft/Transforms/Legalize.cpp:441–453,590–645,944–954 只复用作用域内的只读供应，并为完整 private 更新保留类型、extent、owner。Build 正常生产 IR 记录两个 linear 候选 memref.copy 12→4，caller 输出与必要 snapshot 保留。 |
| A3 | passed | brief.md | A3：所选 Weft f32 实现能承接合法的有界二维计算块，CPU/provider transformation 不再无条件为每个输出元素建立 1×1 allocation/fill/contraction。块、输入供应、累加与尾部由同一 binding 和当前 IR 连接，展开结果进入正式 Weft lowering；Mojo 仍按自身需求形成局部微块，不强迫两个 provider 使用相同表示，也不引入整算子模板。 | lib/Dialect/CPU/Transforms/BlockContractions.cpp:63–140 消费 staticParallelExtent；lib/Target/Weft/Transforms/Implementations.cpp:47–67 以同一 binding 的 panel 形成不超过 4 的二维主块，K=1 保留真实标量尾部；Legalize.cpp:819–899 进入正式 Weft IR。TianchenRV MaterializeRISCVPrograms.cpp:472–610 使用 hasCompleteStreamAxis 的当前 extent 证明，PlanRISCVMemory.cpp:4138–4144 消费旧 implementation。对照 /home/kingdom/phdworks/ref/triton/python/tutorials/06-fused-attention.py:54–80，本轮是 typed 当前程序的条件证明而非 source stage matcher；对照 ref/tilelang/src/transform/storage_rewrite.cc:301–325,951–1000，本轮采用更窄的 private forwarding；ref/tilelang/tilelang/tileop/gemm/__init__.py:121–139 与 src/transform/lower_tile_op.cc:1134–1156 的局部实现、布局和 lowering 连接在本轮体现为 binding 驱动的当前 IR 展开，不引入 emitter 算法或整算子模板。 |
| A4 | passed | brief.md | A4：通过既有性能入口获得受影响 CPU 算子的 generated/source ms、G/S 和同次原容差结果，及时写回项目 CSV；对 Weft linear attention 给出明确结构归因和优化前后 generated 时间，实际改善不能由 source 时间漂移替代。保持同算法、shape、dtype、线程预算和完整 native 调用范围；没有真实性能收益不能把结构改写宣传成性能收敛，未运行项不宣称通过。 | Runtime 最终 Mojo 日志与项目 CSV：causal attention 0.278868/0.279233 ms，linear 0.083348/0.154081 ms；Weft 最终 causal 3.354785/109.410910 ms，均为现有 native benchmark 的同次原容差 pass。Weft linear 同条件 Build 前后 generated 3.437264→1.260196 ms，source 1.901288→1.906588 ms，G/S 0.660969，证明收益不靠 source 漂移。既有 CSV 保留 generated/source/ratio/status、线程及完整 native 调用口径；RMSNorm 1.455781 慢项未隐藏，未受影响 Q4_K 不宣称新测。 |

## 检查

| 检查 | 命令 | 工作目录 | 状态 | 退出码 | 耗时 |
| --- | --- | --- | --- | ---: | ---: |
| Mojo 两条现有区域程序的最终 native 性能 | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/python:/home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/examples INTENT_MOJO=/home/kingdom/.venvs/intentdsl-mojo/bin/mojo /home/kingdom/.venvs/intentdsl-mlir20/bin/python -m repro.v2.runner mojo --compiler /tmp/intentdsl-cpu-structural-build.bs2cdg/tools/intent-compile/intent-compile --output /home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/report/baselinev2/mojo-x86.csv --kernel causal_attention_f32 --kernel causal_linear_attention_f32 --jobs 2 --worker-timeout 300 | . | passed | 0 | 73011 ms |
| Weft 现有 causal attention 的最终 native 性能 | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/python:/home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/examples:/home/kingdom/phdworks/TianchenRV/python INTENT_WEFT_COMPILER=/tmp/weft-cpu-structural-build.Btb6Nm/tools/weft-compile/weft-compile /home/kingdom/.venvs/intentdsl-mlir20/bin/python -m repro.v2.runner weft --compiler /tmp/intentdsl-cpu-structural-build.bs2cdg/tools/intent-compile/intent-compile --output /home/kingdom/phdworks/intentdsl/.worktrees/cpu-structural-optimization/report/baselinev2/weft-rvv.csv --kernel causal_attention_f32 --jobs 1 --worker-timeout 300 | . | passed | 0 | 32300 ms |

## 阻塞项

- **user**: The generic Skill bridge cannot prove an independent Verifier execution; user confirmation is required before Archive. — next: `await-user`

## 风险与跳过的工作

- RMSNorm 仍为 5.019436/3.447934 ms、G/S 1.455781。原容差通过，但新旧 Mojo 源码无差异；运行、调优或外部编译层原因未证明，不能宣称全 CPU 性能收敛。
- causal mask-false 且 V 未知的路径仍保守保留；IME/AMX、新量化格式与其它非目标未覆盖。
- 未执行额外测试或扩大 benchmark 矩阵。依据是现有生产证据、当前源码与最终 Runtime 性能检查，不是所有语义输入的穷举证明。

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | pass | — | 新的独立只读 Verifier /root/cpu_structural_comet_verifier 判定 A1–A4 全部通过；主代理核对关键原文及最终日志，补全 UniformValues 证据为 lib/Analysis 路径。未发现本轮必要的语义、alias/lifetime、provider lowering 或性能证据缺陷；剩余慢项和保守路径如实保留。 | 2026-09-11T08:00:39.960Z |



## 结论

新的独立只读 Verifier /root/cpu_structural_comet_verifier 判定 A1–A4 全部通过；主代理核对关键原文及最终日志，补全 UniformValues 证据为 lib/Analysis 路径。未发现本轮必要的语义、alias/lifetime、provider lowering 或性能证据缺陷；剩余慢项和保守路径如实保留。
