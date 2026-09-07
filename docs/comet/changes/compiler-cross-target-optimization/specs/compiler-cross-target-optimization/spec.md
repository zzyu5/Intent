# Compiler 横向优化与性能攻坚

## 目标与边界

本能力在现有 IntentDSL、canonical KIR、shared physical program 和已有 target lowering 上，统一改进真实作者表达、共同执行结构与参数搜索，使现有编译路径获得可复用的优化能力，并满足下面锁定的性能硬目标。它不修改 `doc/` 定义的语言语义，不以一个作者 kernel 的加速代表 compiler 通用能力，也不承诺覆盖尚无实际需求的新后端或语言构造。

## 横向实现

工作单位是共同语义或执行结构的问题类，覆盖该类在现有 kernel 和已有 target 消费者中的全部实际适用点。作者表达、矩阵输入与累加精度、blocking、ownership、循环携带状态、traversal、复用、materialization、provider form 和候选选择共同参与性能归因；不能先按 kernel 名称修补，再把偶然共性包装成 compiler policy。

作者可以用现有 DSL 显式改善中间精度和表达，保持现有外部 ABI 与容差；需要保留的高精度状态和累加不能被隐式改变。不一律把 f32 降成低精度，也不因 compiler 必须保持作者语义而拒绝修改作者代码。若确有算法变化，已有 Triton 使用或对齐的算法保持；没有 Triton 使用的 cuTile 专用算法可向同算法 baseline 对齐。

Shared facts、analysis 与 transformation 保持唯一 authority，已有 Triton/cuTile/TileLang lowering 从 current typed program 消费它们。纯 provider-local form、能力约束和 API 表达保留在对应边界；既不让 leaf 重建 shared 事实，也不因公共 op 出现在 leaf 就机械上移。只有真实程序缺少执行事实时才扩充相应 typed 表达，不增加 kernel/source/单 case matcher 或另一条 executable path。

实际修复对照 `ref/triton` 或 `ref/tilelang` 的同类实现及对应 provider surface，说明具体代码位置、差异与后果。调查与实现同轮推进，不以报告、重命名或文件搬动替代有效 lowering 和性能改善。

## 有效的有限调优

默认 JSON profile 与 compile-call 覆盖都是候选数据，而不是静态 winner。Shared transformation 按当前 typed role、domain、依赖与执行结构形成完整配置；provider 将其与当前程序实际需要且合法的参数/form 组合。候选绑定必须实际进入 fragment、loop、grid、access 或 provider compiler options。

候选数量可以有限，相关参数可形成少量完整组合；筛除可证明非法、重复或对该程序不起作用的选择。程序确实固定的维度不必制造多个候选；仍有有意义的合法选择时，不无依据地固定所有程序的同一 CTAs、occupancy 或 shared 配置。当前 cuTile 性能覆盖只是一组有限运行输入，不是通用最优策略。

Provider JIT 与实测 autotuning 保留：对未命中适用 winner 缓存的调用，将有效候选交给已有 provider tuner 编译、运行和选择实际耗时最优者；后续调用复用适用结果。Compiled artifact/candidate cache 与 winner cache 是不同职责，复用现有能力减少重复工作，不能以删除 JIT、固定候选 winner 或取消必要 specialization 控制成本。

参考 Triton 的有限完整配置、合法裁剪和实测选优，以及 TileLang 的有限搜索和并行编译机制；不照搬候选数量或其底层表示，不建立任意 physical-program 结构的笛卡尔积搜索。能够参数化的选择交给真实 tuner，必须改变执行结构的缺口由对应 compiler transformation 修复。

并发在资源预算内组织；准备/编译与最终计时协调，保留已准备程序，控制计时相互干扰。不能把等待计时隔离扩大成所有 worker 全流程串行，也不能把并发混合负载耗时声称为单算子稳定性能。

## 锁定的性能硬目标

下表是用户确认时 `report/baselinev2/cutile-5090.csv` 与 `report/baselinev2/cutile-h100.csv` 中 G/S >= 5 的完整集合。设备、entry 和 case 在本轮固定，后续数字下降不会将条目移出验收，也不将此阈值动态套到新的范围。表中的 kernel 名称只定义运行验收集合，不参与 compiler policy。

| 设备 | Entry | Case |
| --- | --- | --- |
| RTX 5090D | chunk_gated_delta | B2-S2048-H8-K128-V128-C64-bf16 |
| RTX 5090D | nvfp4_quantize | 8192x4096-block16-bf16 |
| H100 | grouped_flash_decode | B8-QH32-KVH8-S8192-D128-bf16 |
| H100 | mla_prefill | B1-QH128-KVH1-S2048-D128-R64-fp16 |
| H100 | attention_sink_prefill | B1-S4096-QH32-KVH8-D128-bf16 |
| H100 | gemma_prefill | B2-S4096-QH32-KVH8-D128-window1024-cap50-bf16 |
| H100 | gemma_decode | B32-S8192-QH32-KVH8-D128-window1024-cap50-bf16 |
| H100 | splitk_mla_decode | B8-H64-S8192-C512-R64-split512-fp16 |

每个条目必须在现有数值容差内运行，并达到 `generated_p50_ms / source_p50_ms <= 1.1`。原容差、算法、输入规模、外部 dtype 与明确的完整调用/计时范围不偷换；双方可独立调优，细微舍入、FTZ、近似数学和中间精度差异注明即可，不一概阻断比较。编译、预热和候选搜索成本不算算子时间。

这八个条目在确认时已有可运行的 generated/source 结果。后续超标、编译失败、数值错误或运行阻塞均保持硬目标未完成；不能用平均值、比旧版更快、删除条目或改写成 unsupported 代替达标。改变验收集合须用户明确确认。H100 不支持的原生 FP4 source 不在本集合中，不由此扩展硬件支持任务。

其余 kernel 是横向实现的适用范围，不自动新增全表 <= 1.1 门槛。同一作者程序已有可运行的 Triton/cuTile 比较可辅助归因，但不要求先扩展完整双 target registry，也不能仅凭两个后端都慢或只有一个后端慢判定归属。

本 change 完成后的下一轮按用户要求单独处理 cuTile 全部性能条目至 G/S <= 1.05；该后续目标不替换或扩大本轮的固定验收集合。

## 运行、交付与恢复

复用项目现有 production runner。仅围绕受影响的性能条目及上述硬目标执行 emit、JIT、launch 和计时，在同次 benchmark 的既有输入上进行一次原容差检查；不另开正确性测试轮、不扩展输入矩阵、不新增 pytest、fixture、临时测试脚本或长期验证基础设施。

每项完成后立即更新对应项目 CSV，保留其它未受影响结果，不为每次修改重跑双机全量。CSV 记录算子时间、ratio 和必要说明，不承担运行历史审计，不以候选耗时或 tmp 产物交付。实现按语义完整节点提交，未达标结果如实保留。

本 change 的 brief 与本 Spec 记录当前目标，`doc/` 保持设计权威；个人记忆和历史归档只辅助定位。已停止归档的 cutile-provider-closure-round-six 的旧门槛与暂停快照不再成为执行要求，不重开已完成的 cutile-execution-closure。

验收仅使用 brief 的 A1—A3；本 Spec 完整定义对应行为和固定性能集合，不重复增加独立 Scenario 或检查项。
