### 一、真正存在的问题

  #### 1. shared access relation 仍有一条不安全的“按 dimension 猜相同 traversal”路径

  这是本次发现里最重要的 shared correctness 问题。

  refinePhysicalSchema 在对齐 physical schema 时会调用 lockstepTraversal，确认两个 source occurrence 的 start/stop/step 真正同步：

  - lib/Dialect/GPU/Transforms/Utilities.cpp:1511

  但 refineAccessResultSchema 在无法按精确 AxisMap/source 匹配时，会退化为仅凭 dimensionId 匹配：

  - lib/Dialect/GPU/Transforms/Utilities.cpp:1546
  - lib/Dialect/GPU/Transforms/Utilities.cpp:1587

  同类问题还出现在 extent 传播：代码已经有 source-aware 的 retargetSourceExtent，但一些路径仍使用只看 dimensionId 的 retargetDimensionExtent：

  - lib/Dialect/GPU/Transforms/Utilities.cpp:2539
  - lib/Dialect/GPU/Transforms/Utilities.cpp:2551

  两个 occurrence 可以引用同一个逻辑 shape dimension，却具有不同的切片边界、不同 physical tile 或不同 traversal。dimension 相同不能证明它们是同一个 physical
  relation。

  Triton 的 AxisInfo 在合并不同路径的事实时采用保守 join，不会因为逻辑维度相同就把更强事实传播过去：

  - /home/kingdom/phdworks/ref/triton/lib/Analysis/AxisInfo.cpp:1418

  因此这里不是“多跑几次 relation repair”的问题，而是一条真实的 authority 漏洞：分析 unknown 被错误提升成 exact relation。它目前还没有被绑定到一个已知数值失败，但
  从不变量上已经不健全。

  #### 2. I.sparse_contract_2to4 是公开但必然失败的 API

  它仍然是 public builtin，并被注册、导出：

  - python/intent/language/builtins.py:90
  - python/intent/language/builtins.py:176

  规格也把它定义为正式 shorthand：

  - doc/dsl/core.md:301

  但 frontend handler 无条件报错，要求改用另一个接口：

  - python/intent/frontend/lowering/intrinsics/structured.py:732

  这不是“某个 provider 不支持”，而是根本无法形成 canonical KIR。要么它机械 lowering 到正式 sparse contract schema，要么不应出现在 public surface；当前状态明确自相
  矛盾。

  #### 3. 编译时选择的 GPU device 没有进入 artifact/runtime binding

  target 会读取指定设备的 SM、shared memory、register 和 matrix capability：

  - python/intent/targets/triton.py:34
  - python/intent/targets/gpu/device.py:27

  但 ResolvedTarget 不携带 device identity：

  - python/intent/targets/base.py:8

  三家生成的 run() 最终都改用：

  - 有输入：第一个输入的 .device；
  - 无输入：裸 'cuda'。

  对应位置：

  - lib/Target/Triton/Serialization/Serializer.cpp:474
  - lib/Target/CuTile/Serialization/Serializer.cpp:450
  - lib/Target/TileLang/Serialization/Serializer.cpp:499

  因此可以出现“按 GPU 1 的资源和 capability 编译，却在 GPU 0 上分配与 launch”。多输入跨设备也没有在 wrapper 边界明确拒绝。

  这不是 emitter 权限越界，而是 compile target、artifact 和 runtime device 三者没有形成同一个 typed binding。

  #### 4. TileLang Serializer 正在成为第二个 config legality authority

  TileLang legalizer 先判断某个 thread candidate 是否存在合法 MMA warp partition：

  - lib/Target/TileLang/Transforms/Legalize.cpp:90

  但它只证明“这个 thread count 对某个 shared tuple 可行”。到 Serializer 后，又读取 GEMM 的 M/N、threads 和 config，针对每个具体组合重新执行
  configurationIsLegal()，并删除候选：

  - lib/Target/TileLang/Serialization/Serializer.cpp:284
  - lib/Target/TileLang/Serialization/Serializer.cpp:342

  这不是简单重复检查：最终发给 autotuner 的 config 集合是在 serialization 阶段才形成的，因此 provider-legal IR 并未完整指定终端候选空间。

  TileLang 自己的 warp/instruction policy仍在下层：

  - /home/kingdom/phdworks/ref/tilelang/tilelang/tileop/base.py:65
  - /home/kingdom/phdworks/ref/tilelang/src/op/gemm.cc:188

  所以真正的问题不是“Intent 不应检查任何 TileLang legality”，而是这项 Intent 自己能证明的具体 tuple/thread 约束，应在 provider legalization 中形成 closed
  candidate set；Serializer 只能打印它。现在确实存在两个 Intent authority，再加一个 TileLang 下层 authority。

  Triton emitter没有发现同类问题。它打印已经形成的 config、grid、launch 和 host output allocation，属于正常 executable-module adapter，不是在重做 TTGIR 或 Triton
  compiler。

  #### 5. TileLang 的 scaled contraction 是真实 provider implementation gap，不是 target 边界

  shared KIR/GPU 已经支持 ScaledContractOp，但 TileLang closed surface 与 bufferization 没有接受它：

  - lib/Target/TileLang/Transforms/Verify.cpp:25
  - lib/Target/TileLang/Transforms/Bufferize.cpp:408

  而当前参考 TileLang 已经有明确的 block-scaled GEMM：

  - /home/kingdom/phdworks/ref/tilelang/tilelang/language/gemm_op.py:291
  - /home/kingdom/phdworks/ref/tilelang/tilelang/language/gemm_op.py:456

  它定义了 operand、scale tensor、K granularity、accumulator 与 format 关系。因此这不能再归类成“TileLang 不具备能力”；准确性质是 Intent TileLang provider 尚未映射
  已经存在的 provider-native form。

  这也说明“所有 public construct 必须对所有 provider 都直接支持”是错误要求，但“有当前 ref/source 能表达，Intent 却在 provider surface 前拒绝”是明确实现缺口。

  #### 6. cuTile rewrite 后没有真正执行新 op 的 MLIR verifier

  cuTile pipeline 在 formNativeTiles 后只执行自定义 closed-surface verifier：

  - lib/Target/CuTile/Transforms/Legalize.cpp:734

  但生成出来的 cuTile-specific ops 各自有 schema verifier：

  - lib/Target/CuTile/IR/CuTileOps.cpp:72

  当前流程没有像 TileLang 那样在 rewrite 后调用 mlir::verify(module)：

  - lib/Target/TileLang/Transforms/Legalize.cpp:152

  所以 custom verifier 可以确认“只剩允许的 op”，却不能保证每个新 op 的 operands、results、attributes 和 regions 满足自身 schema。

  这是已经确认的 invariant-enforcement 缺口，但尚未发现一个当前 kernel 已生成 malformed cuTile op。因此它应被称为真实的 verifier 漏洞，而不是已经证实的数值 bug。

  ### 二、前一版自查中不成立或证据不足的问题

  #### “tuning 主要是硬编码 profile”不是缺陷

  固定 tuning table 正是目前确定的设计：

  - compiler 根据 operation category、dtype、target 查少量完整 tuple；
  - 删除自己能证明非法的候选；
  - provider tuner 实测 winner；
  - 不建 cost model，不预测 winner。

  lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp:49 使用人工维护的 profile，本身没有问题。Triton autotuner也要求调用者提供离散 Config 集合，而不是要求上层
  先预测 winner。

  真正需要证据回答的是：某个默认 tuple 是否导致现实结构退化，或某个候选是否能被 Intent 已有 physical facts 证明非法。没有具体候选、资源计算和实测，不能把“没有完整
  register/shared-memory cost model”写成 compiler bug。layout、register allocation 和 pipeline 仍属于 provider compiler。

  #### profile index 拉链组合目前不能定性为 bug

  少量完整、相关联 tuple 本来就优于独立参数笛卡尔积。按相同 profile index组合，可能就是人为定义的 profile identity。

  只有在证明这些数组来自互不相关的参数族，或者存在一个被漏掉的合法高性能完整 tuple 后，才能说当前 zipper 有问题。现在只有代码形态，没有这种外部证据。

  #### relation pass 重复运行不是问题

  shared pipeline 在 mutation 后重新 align/refresh/verify，和 Triton pass 在 rewrite 后重新建立 analysis/verification 是同一类正常组织：

  - lib/Dialect/GPU/Transforms/Passes.cpp:29
  - /home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:77

  真正的问题是某个 repair 内部有没有重新猜事实。上面发现的 dimensionId fallback 是问题；“这个 pass 固定重复几次”本身不是。

  #### TileLang 空 autotune config 不是当前生产路径漏洞

  Serializer 在文本层面确实可能打印 configs=[]，但正常 producer 已保证 shared tuple 与 thread candidate domain 非空；TileLang 自己的 tuner也明确拒绝空 config：

  - lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp:237
  - /home/kingdom/phdworks/ref/tilelang/tilelang/autotuner/tuner.py:1032

  除非先找到一份通过所有现有 verifier、却在具体 tuple 交集上变成空集的真实 program，否则它只是 forged-IR 防御场景，不是当前漏洞。

  #### “DSL/KIR 和 provider 没有统一闭合面”这个说法过宽

  语言构造不需要在所有 provider 上都映射为同名 native op。正确结果可以是：

  - provider-native direct form；
  - provider-local expansion；
  - 精确的 typed rejection。

  所以 atomic load、buffer、sparse contraction 被某家拒绝，不能整批写成 bug。必须逐项查看同语言 source/ref 是否存在等价实现，以及拒绝发生在正确层次。

  sparse_contract_2to4 的问题成立，是因为它在 frontend 就无条件失败；TileLang scaled contract 的问题成立，是因为 ref 已经有对应 form。其它项目目前证据不足。

  #### “Serializer 很宽”不是越界证据

  生成 provider source、host ABI、output allocation、autotune decorator、launch wrapper 和 module-local cache，本来就是 terminal executable module 的组成部分。

  真正的边界是：Serializer 是否重新选择 program structure 或 legality。当前：

  - Triton：没有发现第二套结构决策；
  - TileLang：具体 config legality过滤确实越界；
  - 三家：device binding 确实有 bug；
  - 文件职责较宽只是维护风险，不等于语义越权。

  ### 三、最准确的当前判断

  当前编译器并不是“因为没有 cost model 所以仍然是玩具”。真正未闭合的是：

  - shared relation authority 还有一条把 dimension equality 当 traversal equality 的不安全捷径；
  - public surface 有一个必然失败的入口；
  - compile device 没有进入 artifact/runtime binding；
  - TileLang config legality仍有一部分落在 terminal Serializer；
  - TileLang 缺已被 ref 证明存在的 scaled provider form；
  - cuTile provider rewrite 后的 op verifier没有被真正执行。

  此外，当前性能表是否代表这些代码仍是证据问题，不是编译器设计问题。没有基于当前实现的真实运行数据，就不能声称性能已经闭合；但也不能反过来从“CSV 没更新”推导出
  tuning policy 有错。