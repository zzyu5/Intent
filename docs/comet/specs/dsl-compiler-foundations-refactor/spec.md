# DSL 与编译基础重构

## 目标与当前状态

本 change 将 `report/compiler-foundations-reassessment.md` 的重构结论落实到实际 DSL/compiler，并在完成后归档、本地合入 `main`、清理本 change 的干净工作区。用户已确认作者接口/函数复用、编译阶段/config 组织和四类已知语义修复全部纳入本轮，并确认下述公开 API/配置契约、授权进入实现。本文描述完成后的目标行为，不表示实现已完成。

## 作者接口与归一

常用收缩与 prefix 计算以用户能直接理解的具名 op 表达；已存在的具名入口首先考虑一致性，不无条件制造同义 API。每个选定操作具有完整公开签名、输入/输出 rank、batch/transpose、dtype/promotion、empty/NaN/default 与诊断契约，binding 不再要求用户查找隐藏在 lowering 中的参数规则。

由操作定义唯一确定的 paired axes、logical transforms、builtin combine/identity 由前端生成；完整等价时复用现有 canonical 路径，不因公开名字不同形成另一套 backend/config/serializer policy。

### 具名操作契约

已确认的公开签名：

```python
I.dot(lhs, rhs, *, acc_dtype)
I.matvec(matrix, vector, *, acc_dtype, transpose=False)
I.vecmat(vector, matrix, *, acc_dtype, transpose=False)
I.matmul(lhs, rhs, *, acc_dtype, transpose_lhs=False, transpose_rhs=False)
I.outer(lhs, rhs)

I.cumsum(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)
I.cummax(value, *, axis, inclusive=True, reverse=False, acc_dtype=None)

I.scaled_matmul(lhs, lhs_scale, rhs, rhs_scale, *,
                lhs_format, rhs_format, group_size, acc_dtype)
I.sparse_matmul(compressed, metadata, rhs, *, format, acc_dtype)
```

`dot` 仅接收两个 rank-1 tensors，返回 rank-0 tensor。`matvec` 的 matrix 最低 rank 2、vector 最低 rank 1；`vecmat` 对应相反方向；`matmul` 两侧最低 rank 2，不承担向量的隐式升/降维。矩阵的最后两轴是运算核心，vector 的最后一轴是运算核心；其余轴为 batch。矩阵转置只交换其最后两轴，默认不转置。

矩阵类前导 batch axes 右对齐，使用现行显式 broadcast relation：extent 相等或一侧为 1；0 与 1 结果为 0；动态条件及 extent identity 保留。展开后生成唯一 batch/reduction pairs；K extent 必须相等，不在 reduction 轴广播。结果按 broadcast 后的 batch axes，再接 M/N free axes；dot/matvec/vecmat/matmul 分别得到 `[]`、`[...,M]`、`[...,N]`、`[...,M,N]`。不能把双方相等的非 batch extent 自动视为 batch。

普通乘加操作要求 operands 具有相同 numeric dtype，异型输入由作者显式 cast；累加与结果使用显式 `acc_dtype`，不暗含 TF32、复数共轭、alpha/beta 或可变 C 初值。empty K 返回 accumulator dtype 的零，empty free/batch extent 产生对应 empty result。其余数值、NaN、舍入与允许的 reassociation 继承现行 contract。`outer` 仅接收同 dtype 的两个 rank-1 numeric tensors，结果 `[M,N]`；它是普通 broadcast multiply，不增加 reduction，不继承不同的累加数值行为。

`cumsum/cummax` 只接受 ranked tensor，axis 为一个显式 logical axis，负轴按 rank 归一。两者返回同 shape 的 values，不隐含 argmax indices；inclusive 默认为 true，reverse 默认为 false。cumsum 的未指定 accumulator dtype 使用现行 reduce.sum 的固定 widening 规则，cummax 默认保持输入 dtype；显式 accumulator dtype 的转换沿现行 structured-operation 规则。cumsum 使用加法零；cummax 使用该 dtype 的最小值（有负无穷时为负无穷），传播 NaN；empty source 返回相同 shape 的 empty tensor。它们是允许保持 logical order 的 reassociation 的 scan，不承诺 strict left fold。

既有 `reduce.sum/max/any/all` 保留名字与已有轴/数值职责，内置 combine/identity 由操作定义产生，不再要求调用者提供 identity 或 combine；源码调用同步迁移，不保留另一条旧执行路径。sum 沿现行固定 widening，max 保持现行 propagating maximum；any/all 的 bool identity 分别为 false/true。需要自定义 identity/combine 的调用使用 generic reduce。Generic reduce/scan 的显式 schema、captures、purity 与顺序契约不变。

`scaled_matmul` 保持现行 closed schema：lhs carrier/scale 为 `[M,G,C_lhs]/[M,G]`，rhs 为 `[G,C_rhs,N]/[N,G]`，两侧共享一个显式 group_size；前端生成固定 reduction pairs 和空 batch。format/carrier packing/scale interpretation/rounding 不变，不自动接受其它 scaled ranks 或 batch convention。`sparse_matmul` 对 rank-2 compressed lhs 和 rank-2 dense rhs 生成固定矩阵轴对及空 batch；format、logical metadata、压缩轴与 group 整除契约继承现行 sparse schema，不从 source packed bits 猜 metadata，不增加新 sparse format。其它配轴使用保留的 generic scaled/sparse contract。

具体设计变更先同步到对应 `doc/dsl/` 和必要编程模型章节，再实现。没有新的不可表达语义时，不为每个 public op 新建 canonical operation；确有独有语义时不得为了统一而丢弃。

## 算法组合与函数复用

保持已有 FA、Mamba、Welford、MQA、三角求解等程序体现的组合能力。typed `@intent.fn` 可复用算法 body；使用者不必重写已定义的 summary 协议，算法作者仍可进入同一 DSL body 修改组合。函数复用不调用 provider source 模板，也不形成隐藏 kernel launch。

Generic contract、custom reduce/scan、region fold/scan、普通 loop/carry、逻辑索引与显式读写保留各自职责。算法名称不决定采用哪种 construct；严格依赖不强制 region 化，ABI-visible chunks 不成为 compiler-selected slices。

报告指出的退化 region→ordinary reduce/scan canonicalization 在相关范围中按同一语义实现；不得因此删除真正包含 slice-level tensor algorithm 的 region。

## 实现覆盖与语义兑现

新增公开操作从现有 kernel 与 adapter 实际调用链出发复用已成立路径。若本轮涉及的表达/provider 形态缺少 lowering，补齐通用 typed relation/form/realization，不为某个临时 kernel 或 CSV 条目创建特化模板。

本轮包括报告明确的四类语义兑现：Triton f32 contraction 不静默改变输入精度；signed quotient/remainder 保持 floor relation；cuTile logical index 缩宽须有证明；autotune trials 不额外污染一次调用的可观察状态。对其余静态风险只沿实际涉及路径调查，不声称未知部分已自动闭合。

成功记录、当前实现、历史 repro 和局部失败分别限定到实际程序/provider/形态；不从一次 failure 推导整门语言不可表达，也不从已有 CSV pass 推导所有目标和形态已完成。

## 编译阶段与策略组织

本轮明确现有 transformation groups 的语义完整边界、顺序依赖和 postconditions，使 construction、关系维护与实际 transformation 的职责可理解，不以无关大重构或新的 profiling 框架完成此目标。

有限候选数据与 IR 分析、适用条件、角色绑定、capability/legality 判断分离，结果继续写入已有 typed IR。它不能成为开放式策略语言、serializer 重新解释程序或 runtime 算法选择器。

### 配置输入契约

在 `intent.compile` 与 `compile_shared_gpu` 的 keyword-only 参数中提供 `tuning_config: str | Path | None = None`，对应 `intent-compile --tuning-config <path>`。未提供参数时使用随 compiler 分发的默认有限数据表；显式 JSON 文件按 shared/provider 命名空间和现有 profile 家族覆盖纯数据，某家族一旦提供，其完整候选行替换该家族默认行，未提供家族继续使用默认数据。不存在按 kernel 名、registry、source template 或任意脚本选择 profile 的入口。

文件读取是该次 compiler invocation 的输入，发生在配置物化前；默认数据和外部输入使用同一 typed 数据契约。调表无需重编译 C++，重新编译受影响 artifact 才采用新候选；已形成 artifact 不在 launch/autotune 时重读文件。配置输入不进入 KIR、不改变作者算法、数值、effects 或 host-visible kernel 数量。

文件不存在/不可读、格式错误、未知字段/命名空间/家族、错误类型或空候选行集直接给出诊断，不静默采用默认文件。语法和数据类型正确的候选仍经过现行 typed role/domain、physical program 与 provider 约束；可证明非法的候选被过滤，所有候选均不合法则编译失败，不用另一算法或默认表兜底。IR 分析、family 适用条件和合法性判断保留在相应 shared/provider 代码中，不写入配置文件。实际选入的候选完整保存在 typed IR 和 artifact 中，不新建版本、迁移或缓存校验体系。

provider 可接受值/结构约束与所选 mapping/search budget 分开；不将当前 occupancy/warps/stages 搜索集合当作唯一数学合法集合，也不无条件扩大候选。

## 非目标与不变量

不重做核心 IR，不新增硬件后端，不恢复旧 cuTile 性能追平任务，不要求任意 examples×provider 全面覆盖。算法、dtype、effects、source order、alias 和 host-visible kernel 编排仍由作者语义定义。

没有 kernel-name matcher、source template、精度放宽兜底、兼容迁移体系、测试目录或长期 benchmark 工程。Shape 不运行代码；实际实现的必要验证遵守项目单条 production repro 规则，并以真实 current/ref 对照支持判断。

## 交付与收尾

验收使用 brief 的 A1—A4 结果型要求，不从本节额外派生验收场景。单个 Native Change 顺序实施，子代理只作探索/只读复核；不创建相互争用相同 shared/provider 边界的并行实现子任务。

用户已要求最终本地合并到 `main` 并保持干净。只提交本 change 所有的实现和正式产物，独立验收后归档/合并，清理已合并且干净的 worktree/分支；不推送或创建 PR。先前 cuTile 试验的具名 stash 保持独立，不自动应用或丢弃。合并与清理是 Archive 的必需完成条件，不要求在独立代码验收之前完成合并。
