# Shared GPU Program IR 与 Triton 纵向闭合

## 1. 本轮结论

本轮删除了旧 `Plan + KernelModel + 三家 materializer` 执行链，建立了唯一的：

```text
canonical KIR
  → shared executable GPU program
  → shared structural passes
  → Triton provider legalization
  → terminal Triton source
```

`grouped_gemm` 已通过这条新链完成真实 Triton JIT、GPU 执行和数值对照。第一次正确性闭合后的结果为 generated `30.493343 ms`、source `4.636912 ms`、ratio `6.576218`；随后从 current physical range 识别 data-dependent member traversal，不再把全局 `R` 复制到每个 group 的 launch ownership，并加入显式 program-internal row traversal 与可调 row workers。最终结果为 generated `4.502480 ms`、source `4.648112 ms`、ratio `0.968669`。这个结果证明纵向链与一项真实 shared mapping refinement 成立，但不等于所有 canonical family 已经闭合；横向审计仍确认 region fold/scan、multi-contract joint realization 等明确的 implementation gap。

## 2. Shared GPU IR

新的 `intent_gpu` dialect 位于 `include/Intent/Dialect/GPU/` 与 `lib/Dialect/GPU/`，包含：

- `PhysicalExprAttr`：区分 constant、ABI dimension/scalar、physical parameter 及受限整数表达式；launch expression可读host-visible metadata，而fragment extent只允许constant/parameter及其组合；
- `ParameterAttr`：保存 typed role 和合法 candidate domain，winner 不写回 IR；
- `ViewType`：保存 external ABI、shape、stride 与 alias facts；
- `FragmentType`：保存 element type、parameterized shape、logical source-axis mapping、validity class 与 owner；
- `RangeType`、`BufferType`、`RecordType`；
- program mapping、range、value、access/effect、structured compute、atomic、buffer 和 structured-control operations。

Physical kernel 显式保存 capabilities、program space、grid rank、ABI role/name、effect origins 与 executable body。`verifyGPUProgram` 检查 MLIR dominance/region legality、唯一 physical kernel、launch-expression bindings、ABI schema、program-id 使用、physical parameter uniqueness、dialect closure、fragment expression bindings和 observable-effect coverage。Canonical `intent.*` op 一旦进入 physical kernel 会立即被拒绝。

本轮没有把 provenance 当作 executable authority。`intent_gpu.origin` 只用于 effect coverage 与诊断；program mapping、coordinates、validity、loops、accumulators 和 accesses 都必须存在于 current SSA/operations。

## 3. KIR → GPU construction

`lowerCanonicalKIRToGPU` 先验证 canonical KIR，再建立 public view/scalar ABI以及 launch-visible dimension/stride metadata。外层 unordered parallel domains 被组成有限 worksets；单 workset保持一份直接 executable body，多 workset使用disjoint one-dimensional program-space segments与uniform dispatch。Construction返回前强制执行 access composition、contraction co-realization与reduction co-realization；只有这些步骤之后的program通过shared verifier，原KIR才被删除。因此runtime logical extent不会作为非法fragment shape逃到下一层。

Construction 为 canonical value/control/access/structured families创建 typed GPU operations，而不是 `exec_*` clone：

- scalar、fragment、range、record 与 ordinary `if/for/while`；
- explicit load/store/gather/scatter/atomic coordinates、source axes、valid/fill 与 effects；
- reduce/scan/contract/scaled contract/sparse contract/histogram schemas；
- logical buffer allocation、read/write resource identity，以及由lexical ordered-control depth确定的program-private/iteration-private scope、initialization obligation、optional initializer和lifetime；
- nested parallel axes的row-major workset mapping。

Access conversion 分开维护source axis与result axis，不再用“已生成多少source coordinates”索引result dimensions；`weight[group, reduction, columns]`这类前置scalar index不再错位。动态subregion extent由其runtime `RangeOp` SSA绑定，嵌套control继承同一dimension binding；`i32` ragged offsets显式cast为physical index。`I.indices`的shape identity只描述extent，其axis map直接继承source range provenance。Load/gather结果的axis map由`IndexRelation + current coordinate SSA`机械组成，不再从tensor shape反查逻辑轴。typed constant也在边界重建为准确result type。

`assume_in_bounds`保留为explicit GPU operation。无法完成mandatory physicalization时会在shared construction处给出具体诊断；不再让非法runtime fragment或未物化structured op进入leaf。

## 4. 产生真实 IR 变化的 shared passes

### 4.1 Access composition

当in-register gather的source是一个无残余validity/fill的direct external load时，pass组合两层current coordinate mappings，生成一次直接`LoadOp`，保留external read origin、result relation与gather validity/fill，再删除whole-value load和dead ranges。`group_offsets`由“全向量load再scalar gather”收敛为一次scalar load；该rewrite只依据typed def-use与axis provenance。

### 4.2 Contraction blocking 与 ragged traversal

对具有 typed source-axis mapping、unit-step direct loads、单 reduction pair、零 tail fill、scalarizable accumulator和 unique store path 的 contraction，pass 会真实创建：

- `BLOCK_M/BLOCK_N/BLOCK_K` physical parameters；
- row/column ownership coordinates；
- M/N/K fragment types；
- K `scf.for` 与 loop-carried accumulator；
- tail predicates、masked loads、blocked contract 和 masked stores；
- 改写后的 one-dimensional program mapping。

普通 launch-visible row domain 形成 row×column tile ownership。若 row range 来自 program 内的 runtime subregion，pass 不把 data-dependent member count 伪装成 launch extent；它保留外层 workset 与 column ownership，在每个 program 内创建 row-tile `scf.for`。`ROW_WORKERS` 是显式 physical parameter，直接约束 grid coordinate、row-loop start 与 stride，并与 `BLOCK_M/N/K` 一起交给 Triton autotuner。候选 `{1,2,4}` 会对同一结构产生不同合法 ownership；最终实测从 `6.576218×` 收敛到 `0.968669×`。

该pass只在mapping覆盖整个current program space且kernel只有一个可独立改写的contract时运行。这个限制避免局部rewrite覆盖其它execution-group segment；若原contract含runtime-shaped fragments而判据不满足，shared construction直接诊断，不能静默保留半成品。已经具有合法compile-time fragments的contract可以保持原physical form，由provider legality决定是否有native lowering。

原始load fill只有在为数值零时才可与compiler-introduced contraction tail合并；blocking pass保留construction按canonical zero-origin contract语义建立的exact accumulator SSA，而不是在rewrite中另造一个零。这两项是节点二审计发现并修掉的静默语义风险。

### 4.3 Reduction blocking

对compile-time extent、direct-load、unit-step、单physical axis的first-class reduce，pass用`next_power_of_two` extent创建实际range、tail predicate、identity fill、blocked load与replacement reduce。对runtime extent，pass声明 typed `REDUCE_CHUNK_*` parameter，创建真实 `scf.for`、固定形状chunk ranges、identity-masked loads、chunk-local reduce，并把原combine region机械内联为loop-carried accumulator update。Captures保持显式SSA operands；runtime extent从不进入fragment type。Generic reduce允许合法reassociation；tail padding仅在原invalid fill与对应component identity是同一SSA值或同一scalar constant时启用。当前尚未闭合的是多个reduction axes的联合chunking，以及free axis尚未先physicalize的runtime reduction。

每次 shared rewrite 后都重新运行 GPU verifier。Shared 层不选择 Triton warps/stages/CTAs，也不包含 provider名称或设备型号分支。

## 5. Triton provider boundary

Triton leaf 被拆为 `Transforms` 与 `Serialization`：

### 5.1 Provider transforms

- `ProgramGrid` 只在唯一 mapping 可由 Triton 的一至三维 launch surface直接表示时，将 shared one-dimensional delinearization改写为多个 `program_id`；任意更高秩或多 execution-group program继续使用已存在的一维 mapping。
- masked in-register gather 先把 invalid index显式选择为安全 index，再执行 `tl.gather`，最后按原 validity/fill `select`；serializer 不再用不短路的 `tl.where(tl.gather(...))` 冒充 masked access。
- typed add `scatter_reduce` 仅在combine region确为交换参数不变的add、dtype属于Triton atomic-add surface时改写为`AtomicRMWOp(kind=add)`；其余保留到provider verifier明确拒绝。logical sharing domain由shared op携带，leaf不再根据resource type临时猜scope。
- provider parameters声明 `num_warps/num_stages/num_ctas` candidate domains；verifier要求每种 provider role唯一并检查 warp candidates为二次幂。它们进入 `triton.Config`，winner仍由 Triton autotuner实测。
- provider verifier在serialization前检查 grid、static/constexpr fragment extents、access source axes、native contract/reduce/scan/gather/scaled-contract forms、atomic dtype/order/scope和 closed operation surface。

### 5.2 Terminal serialization

Serializer 不再调用 provider verifier，也不读取 KIR。它只：

- 打印 imports、autotune configs、kernel signature、grid 和 launch wrapper；
- 按 current IR 顺序打印 SSA values、structured control与operations；
- 将已验证的 type/enum/axis/access spelling机械映射到 Triton API；
- 为 public `Out` ABI生成 host wrapper allocation并传入同一次 kernel launch。

Pointer arithmetic只消费 current access coordinates、`source_axes` 与 view ABI stride；它不解析 KIR index relation。被 provider verifier拒绝的 operation没有 serializer fallback或慢路径。

## 6. 删除的双份 authority

本轮完整删除：

- `Intent::Plan` dialect及 `intent_plan.exec_*`；
- `Target/Common` 的 KernelModel、analysis、replay、combiner、lifecycle 与 operation registry；
- 旧 `Target/GPU` PhysicalPlan/decision records和一次性 realizer；
- 三家旧 handler/materializer/program-form/syntax-spelling执行链；
- Triton scalarized/serial fallback与 materializer 中按 KIR relation、role或result shape重建结构的路径；
- serializer与provider verifier的重复耦合。

CMake 与 `intent-compile` 只保留一条 KIR→GPU→shared passes→Triton transforms→serialization 路径。cuTile/TileLang 在本轮明确未实现，不存在兼容或旧 fallback。

## 7. 横向接纳审计

静态 consumer 审计确认 canonical KIR operation families均有 KIR→GPU conversion 分支或明确的 enclosing-control consumer；没有 canonical op 被默认 clone、按名称 dispatch到旧 handler或悄悄删除。Examples 中也没有残留 `state_stream`、`partition(...)`、`I.ordered` 或 `I.auto(...)` surface。

但“有 schema/conversion”不等于“已闭合到 Triton”。当前边界如下。

### 7.1 Shared implementation gap

- `region_fold` / `region_scan`：physical schema、helper regions与segment parameter已建立，但physical segment loop、source-slice active validity、summary/state carry和scan output assembly尚未物化。shared verifier明确拒绝，不能让serializer重建或用serial fallback冒充支持。
- multi-axis/runtime-free-axis reduce：单reduction axis的runtime chunk loop已经闭合；多个reduction axes的联合chunking，以及尚带runtime free-axis的source，仍需先形成对应ownership/fragments。
- multi-contract joint realization：单个runtime-shaped contract可以整体改写；多个相关contract尚不能共同重写program mapping、reuse与effects，shared层明确诊断。
- execution-group联合physicalization：单个contract已经能区分launch-visible dense row ownership与program-internal runtime subregion traversal；多个相关contract或多个不同execution groups仍不能共同重写mapping、reuse与effects。

### 7.2 Provider-realization gap

- kernel-local mutable `BufferOp`：shared construction已经区分program-private与iteration-private allocation，并保存initialization obligation、optional initializer、instance、owner、lifetime与visibility；Triton-local SSA/mutable-buffer functionalization与 invocation-workspace ABI 尚未实现，provider在serialization前明确拒绝。
- multi-axis/captured reduce、exclusive/captured scan以及一般多轴 in-register gather：Triton存在可组成的底层能力，但当前 provider rewrite 尚未形成对应 current program；不能归类为目标语言缺能力。
- multi-workset或多个相互关联contraction的联合blocking尚未实现；需要runtime physicalization的组合会在shared层失败，不会覆盖其它execution groups或流入leaf。

### 7.3 真实 Triton surface boundary

- atomic load：Triton没有保持 memory-order semantics的 `atomic_load`；普通 load不等价，CAS模拟会增加write effect，因此明确 unsupported。
- native structured-sparse contraction：当前 Triton surface没有携带2:4 metadata的 sparse dot原语；`input_precision`与sparsity无关。未实现显式typed decode→dense contract前，不发射伪 native path。
- `tl.dot_scaled` 只接受其定义的rank/operand format/e8m0 scale/group/axis/f32 accumulator组合；其它 canonical scaled-contract schema在provider legality处拒绝，不丢弃group或axis attributes后强行打印。

这些状态说明第三轮的纵向架构已经替换旧链，但 Triton横向接纳尚未达到 prompt 所要求的全部 families；报告不把实现缺口写成下层能力边界或完成。

## 8. 节点二与节点三自查

节点二发现并修正：

- non-scalar/nonzero fill被 contraction pass静默替换为零；
- compiler-generated tail与canonical fill混为一个 `other`；
- contract accumulator被无条件清零；
- contraction局部 rewrite覆盖整个多-group program space；
- shared 层错误持有 Triton三维 grid refinement；
- masked gather先越界访问再 `where`；
- scatter sharing scope在leaf按resource type重猜；
- access relation result-dimension和view-axis缺少边界检查；
- scalar source index与result-axis被混用，导致前置scalar index后的range错位；
- dynamic subregion dimension没有沿nested control继承其runtime SSA binding；
- tensor shape identity被错误当成coordinate provenance；
- ragged offsets先whole-load再in-register gather，留下runtime-sized fragment；
- `i32` ragged bounds没有显式转换为physical index；
- canonical literal attribute与physical result type不一致；
- runtime-shaped contract/reduce未被realizer接纳时曾静默success；
- data-dependent ragged row count曾被全局view extent替代并复制进每个group的launch ownership；
- ordinary range与runtime subregion曾都因`RangeBoundOp`形态相同而被误判为ragged traversal；现在construction显式保留subregion provenance，blocking只消费该事实；
- runtime reduction replacement曾丢失origin，physical parameter同名复用也未核对role/candidate schema；现在loop/chunk op保留origin，同名schema冲突直接诊断；
- transform通过“module中第一个函数”而非唯一 physical-kernel标识取入口；
- serializer重复调用provider verifier并保留不可达 scatter helper。

节点三确认：旧 Plan/Common/Target-GPU/三家 materializer目录均已从构建图和唯一 executable path 移除；当前执行实现不再构造或消费 `intent_plan`、`exec_*`、`KernelModel`、`analyzeKernel`、`kernel.nodes`、`intent.result_shapes`、legacy fallback、kernel-name branch或device-model branch。保留在 canonical verifier 中的旧名称只用于拒绝遗留 IR；未删除的厚 serializer代码只打印current IR和host launch wrapper。

## 9. 验证

构建：

```bash
cmake --build build -j 12
```

唯一端到端 repro：

```bash
./examples/run/baseline-v2.sh triton /tmp/intentdsl-gpu-ir.csv grouped_gemm
```

第一次数值闭合后的输出：

```text
triton:grouped_gemm: pass generated=30.493343 ms source=4.636912 ms ratio=6.576218
```

修正 ragged ownership 并让 row-worker 参数进入真实 grid/loop 后，最终输出：

```text
triton:grouped_gemm: pass generated=4.502480 ms source=4.648112 ms ratio=0.968669
```

`pass`表示Triton source/JIT、GPU launch与数值对照均成功；两次结果的结构差异来自shared physical program，不是serializer字符串替换。临时CSV位于`/tmp`，未进入仓库，也没有更新六张baseline表。

## 10. 第四轮边界

本轮没有为想象中的 cuTile/TileLang 差异增加 shared fields，也没有保留它们的旧 lowering。第四轮需要从同一 shared GPU program分别建立两条真实 provider legalization/serialization链；只有 storage/copy/sync/native operand等真实差异逼出时才增加 local extension。上述 shared region traversal 与buffer realization缺口必须先在共同层/明确provider层关闭，不能在另外两家 materializer中各重建一次。
