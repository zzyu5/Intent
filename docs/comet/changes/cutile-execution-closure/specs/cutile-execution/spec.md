# cuTile 执行与性能收束

## 目标与适用范围

现有 cuTile registry 通过唯一生产编译和运行路径，在 RTX 5090D 与 H100 上完成性能测量。算法已实现且硬件支持的条目必须在约定容差内运行；真实硬件限制明确记录，不用实现缺口冒充 unsupported。结果保存在项目现有两张 cuTile CSV，并随每个 entry 完成增量更新。

本能力使用当前 DSL、canonical KIR、shared physical program 与 provider lowering，不重新定义语言语义，也不增加独立测试体系。

## 当前编译入口与配置

生产入口使用当前工作区构建的 compiler、相应 profiles 和 Python runtime。默认 JSON 候选表与其 shared/provider 职责相邻，编译调用可显式覆盖；读取和 materialization 属于 compile 阶段，不恢复旧 runtime provider candidate policy。

Intent 编译生成当前 physical program 和 cuTile source；Python source 装载、provider JIT、autotune 与 GPU launch 是不同环节。来源清楚不等于必须建立版本审计表，但不能把旧默认 binary 的运行声称为当前实现。

## 有界并发与 JIT 成本

Runner 可在同机同时推进多个独立 entry，按可用资源限制在途 worker；不同机器独立并发。不能以计时需要隔离为由让每个 worker 从准备到结束全程串行，也不能在同 GPU 竞争状态下把混合负载时间当单算子 latency。

准备、编译、必要初次执行与最终计时分阶段协调，保持已准备程序和 worker 可复用。只围绕实际性能运行处理重复编译、重复调优和不必要候选/specialization；不无依据移除 constexpr、编译必要参数或 provider JIT。编译产物缓存与调优 winner 缓存分别处理，利用外部 provider 既有能力，不把缓存命中或重启一概解释为全量重编译。

## 同算法、容差与性能

双方使用相同算法、输入 shape、外部 dtype 和明确的调用/计时范围。在同一次 benchmark 中使用现有 entry 约定容差检查输出，通过后测量性能；不另设数值测试轮次，不追求 bitwise 或逐操作一致。

FTZ、近似数学、中间舍入等细微差异在容差内即可，不再由 source contract gap 标签直接跳过计时。ABI 表示、辅助输出、布局转换与 workspace 差异注明相应计时范围；不要求先对齐完整候选集合或穷举所有配置。

超出容差、异常 NaN/Inf、编译或 launch 失败必须如实记录并修复。已知 absorbed MLA 的 NaN 属于实现缺口；NVFP4 的缺失作者实现属于本范围；H100 E8M0 scaled MMA 的实际硬件限制单独归类。不得为了通过而放宽容差，也不能把数值错误程序的速度称为有效加速。

若实际算法不同，已被 Triton 使用或已与 Triton 对齐的作者算法保持；没有 Triton 使用的 cuTile 专用算法可以对齐 cuTile baseline。该调整只影响对应作者算法，不修改共享 helper 或语言语义。chunked softmax 的 online summary/三遍扫描差异按此处理。

## 结果与恢复

`report/baselinev2/cutile-5090.csv` 与 `report/baselinev2/cutile-h100.csv` 记录真实 generated/source 算子时间、ratio 及必要失败说明。编译、预热和候选调优耗时不冒充算子时间，表格不承担运行历史审计；不因某项失败或任务中断覆盖已完成结果。

全量表示当前完整 registry 都得到实际运行结论，不是扩展输入、边界、回归、兼容或组合矩阵。修复只重跑受影响性能项，不反复重启全部。真实高 ratio 保留，用于定位当前 physical mapping/blocking/ownership/traversal/materialization、provider form 或运行层问题，不以旧全表阈值作为继续推进的门禁。

所选 change 的 brief/spec、项目规则和当前设计原文参与知识检索。个人记忆和项目知识仅辅助定位；旧 change 的暂停指令、测试任务和验收门槛不成为新 change 的 authority。过期或冲突副本停用，可追溯到原文，不维护项目外临时测试集合。

验收仅使用 brief 的 A1—A4；本 Spec 描述完整行为，不重复增加独立检查项。
