# 最新算子表达与编译闭环报告

## 结论

这一轮先按公开算法选定三个当前推理结构，再原样落成 DSL，没有先看编译器能否接受：DeepSeek 式吸收 MLA、FlashMLA 式 token-sparse MLA，以及 DeepGEMM 式 FP8 MQA logits。吸收 MLA 又自然包含 query 吸收投影、latent attention 和 value 重构投影三次作者调用，因此全量表新增 5 条 `(kernel, case)` 记录。

RTX 5090 D 上完成真实数值运行的 provider-case 为 9/15：两个 MLA 投影三后端全部通过，吸收式 causal MLA 在 cuTile 通过，FP8 MQA 在 Triton 与 TileLang 通过。其余 6 格没有伪装成支持：Triton absorbed MLA 超出当前设备 shared-memory 资源，TileLang absorbed MLA 没有下层合法配置；token-sparse MLA 三个下层均未在限定时间内完成可用编译；cuTile FP8 MQA 明确缺少 runtime-sized matrix-M lane 的等价原语。

这批陌生算法确实逼出共享编译器缺口，但没有增加任何 kernel 名分支。修复集中在四个结构事实：任意秩 `new_axis` 广播、外部 View 的多轴间接 gather、布尔组合的三目标映射，以及非复用 row-vector 的物理 tile 投影。两份固定 CSV 都只追加 5 行，旧数字和 source 数字未动。

## 一、算法选择与公开依据

| 记录 | 选择原因 | 算法依据 | DSL 中保留的算法结构 |
|---|---|---|---|
| `mla_head_projection/query_absorb` | 检验低秩 KV 潜变量、不同形状收缩共享同一 head 轴 | DeepSeek-V3 `inference/model.py` 的吸收式 MLA | 每个 head 将 no-PE query 从 128 维投影到 512 维 latent 空间 |
| `absorbed_mla_prefill` | 检验低秩 content score、独立 RoPE score 与因果在线归一化的组合 | DeepSeek-V3 `inference/model.py` | latent content dot 与 64 维 RoPE dot 分开计算后相加；只读到当前位置；值仍在 latent 空间累加 |
| `mla_head_projection/value_reconstruct` | 补齐吸收式 MLA 的输出端 | DeepSeek-V3 `inference/model.py` | 每个 head 将 512 维 latent attention output 投影回 128 维 value 空间 |
| `token_sparse_mla_prefill` | 检验学习式 token 选择、间接多轴读取和三输出 ABI | FlashMLA sparse prefill README/实现接口 | 每个 query 的 token index 列表驱动 latent/RoPE cache gather；在线 softmax 同时输出 latent output、maximum 和 LSE |
| `fp8_mqa_logits` | 检验原生 E4M3 收缩、per-key scale、per-head 权重和动态范围 mask | DeepGEMM FP8 MQA logits 实现 | FP8 q×kv 先 FP32 MMA，再 ReLU、head weight、head reduction、KV scale、范围 mask |

公开结构参考：

- DeepSeek-V3 MLA：<https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py>
- FlashMLA token-sparse prefill：<https://github.com/deepseek-ai/FlashMLA/blob/main/README.md>
- DeepGEMM FP8 MQA：<https://github.com/deepseek-ai/DeepGEMM>

这里没有把三次作者调用偷偷融合为一个 kernel。`mla_head_projection` 是同一个通用投影 kernel 的两个运行 case，`absorbed_mla_prefill` 是独立的 latent attention 调用；调用次数和中间 ABI仍由作者编排。

吸收式 MLA 的 score scale 保持 `1/sqrt(128 + 64)`，而不是按吸收后的 512 维 latent 坐标改成 `1/sqrt(512 + 64)`。这是因为权重吸收改变了坐标表示，但要保持原始 128 维 no-PE query 与 64 维 RoPE 的数值合同。token-sparse MLA 直接在 512+64 维上定义 score，因此使用 `1/sqrt(512 + 64)`。

## 二、RTX 5090 D 实际结果

单位为 ms；只有真实执行并与 PyTorch 数值 reference 对照通过的格子才有延迟。`not measured` 表示源码已经生成或编译开始，但没有完成可用运行，不等于 compiler 支持。

| Kernel / case | Triton | cuTile | TileLang | 数值结论 |
|---|---:|---:|---:|---|
| MLA projection / query absorb | 0.0097 / 0.0105 | 0.0098 / 0.0118 | 0.0103 / 0.0123 | 三者 PASS，max error `1.2207e-4` |
| MLA projection / value reconstruct | 0.0102 / 0.0104 | 0.0123 / 0.0130 | 0.0142 / 0.0144 | 三者 PASS，max error `2.4414e-4` |
| absorbed MLA prefill | not measured | 0.1536 / 0.1557 | not measured | cuTile PASS，max error `1.2207e-4` |
| token-sparse MLA prefill | not measured | not measured | not measured | 三者均未完成运行，不能给数值结论 |
| FP8 MQA logits | 0.0364 / 0.0384 | unsupported | 0.0895 / 0.0915 | Triton/TileLang PASS，max error `3.5763e-7` |

具体未完成原因：

- Triton absorbed MLA 当前 realization 形成 102400 B shared footprint，超过本机该 kernel 可用的 101376 B；算法 IR 与源代码已经生成，但不能把资源失败记成 PASS。
- TileLang absorbed MLA 的 autotuner 没有一个配置完成编译与验证。
- token-sparse MLA 已经在三条路径上生成目标源码；Triton 与 TileLang 首次 JIT 超过 300 秒，cuTile 的候选出现下层编译超时或全部无效。它暴露的是当前多轴间接 gather + 长 latent reduction 对三个下层的编译压力；证据不足以宣称某个 surface 在语义上永久不支持，所以 CSV 记 `not_measured`。
- cuTile FP8 MQA 的 Q 轴主体是单行 head reduction，生成矩阵原语需要 runtime-sized matrix-M lane。当前 cuTile 模型没有等价原语，编译器在物理轴的源码位置明确报 unsupported，不再串行降级冒充支持。

FlashMLA 的公开 sparse-prefill ABI 把 indices 写成 `[s_q, h_kv, topk]`，但当前实现同时要求 `h_kv=1`，等价 reference 进入计算前直接 squeeze 成 `[s_q, topk]`。因此 DSL 使用 `[Q,T]` 是该公开算法当前真实限制下的等价 ABI；它没有删掉一个参与计算的 KV-head 维度。公开实现把 576 维 Q/KV 打包在一起，DSL 将它透明拆成 512 维 latent 与 64 维 RoPE 两个 view 后分别做 dot 再相加，数学路径不变。

本轮没有接任何 source baseline；CSV 的 source 列保持空白。PyTorch reference 只用于数值正确性，不被写成高性能上游对照。

## 三、暴露并修掉的编译器问题

### 1. 任意秩广播不再只认一两个固定形状

FP8 MQA 的 `head_weight[query, head, None]` 与 sparse MLA 的 `valid_token[:, :, None]` 都是作者已经写出的 `new_axis`。原来的公共分类和目标叶子只覆盖少数 rank-1 append/prepend 形态，陌生关系会落入没有机械映射的路径。

当前共享 pointwise relation 只检查一个明确合同：relation 至少含一个 `new_axis`，其余项只能是 `full_slice`。三个 emitter 分别投影为各自的 reshape/expand 语法；没有推导新的广播轴，也没有按 MQA 或 MLA 名字判断。

### 2. 外部 View 的多轴间接 gather 端到端闭合

token-sparse MLA 的 cache 访问是 `[Q-tile, selected-token, latent-channel]`，索引来自外部 `selected_tokens` 张量。Kernel IR 原先已经完整保存这个 relation，但叶子仍只兑现 rank-1 向量 gather，或者从物理轴重拼地址。

现在三个 target 都直接消费同一个 pointwise relation 和 transfer boundary：

- Triton 从 relation 生成 pointer、bounds 和作者的 valid predicate；
- cuTile 使用 `ct.gather` 后按作者的 valid/fill 兑现；
- TileLang 对当前 target 允许的元素域机械生成地址、bounds 和 `if_then_else`。

共享 GPU analysis 只判断它是 indirect relation，不把“source 是否为外部 View”错误提升成全局限制。原因是 scan 的 scalar-access materialization 等既有路径会在更专门的阶段正确兑现局部 tensor gather。三个普通 target 叶子的新通用路径只接受外部 View；其他局部 gather 继续由其已有结构处理，落不到任何专门路径时才在实际叶子给源码位置诊断。

### 3. 已有逻辑运算的映射表补齐

sparse token 的有效性和 MQA key range 都需要两个比较结果合取。前端已经将它们表达成 bool tensor 上的 `logical_and`/`logical_or`，缺的是共享 capability 与三个目标的逐 op 拼写。当前三个叶子都在 bool mask 类型上机械使用目标位逻辑运算；没有短路语义，也没有控制流改写。

### 4. row-vector 与 deferred contraction 只读物理计划

FP8 MQA 首轮生成的 Triton 源码使用未定义的 `BLOCK_SIZE`，根因是非复用 row-vector 的物理宽度没有从轴计划读取。当前 Triton 统一通过 physical-axis projection 得到 `next_power_of_2(logical extent)`；reuse-worker 轴保持使用其已选 tile。

同一审计还发现 cuTile 的 deferred contraction accumulator 虽然已有 `physicalAxisTile` 投影，却仍直接读取逻辑 `axis.tile`。当前已和 Triton 收敛为唯一物理轴投影，避免逻辑范围与实际 lane extent 不一致。

收缩 load 的 defer 条件也不再依赖“这个 relation 恰好有某种旧 components 形态”。共享 helper 当前要求：load 只有该收缩一个消费者、两侧都是直接外部 load、Plan 决定两侧 shared、并且两侧 boundary 共同绑定恰好一个 reduction axis。三个 target 在所有 boundary 建完之后统一读取这一事实。没有 reduction-axis provenance 的 full-slice contraction不会被错误推入 deferred 路径，多消费者 load 也不会因延迟而在其他消费点丢失定义。

## 四、失败分类与边界

| 现象 | 分类 | 处理 |
|---|---|---|
| `new_axis` 只接受固定 rank 模式 | 真实 lowering 缺口 | 补共享 relation 合同与三目标 op 映射 |
| 外部 View 的 rank-N data-dependent gather 丢在叶子 | 假发射 | 地址、bounds、valid 直接从 IR relation + Plan boundary 投影，删除按轴重拼的限制 |
| Triton MQA 出现未定义 `BLOCK_SIZE` | 假发射 | 非复用 row-vector 统一读取物理轴投影 |
| cuTile deferred accumulator 使用逻辑 tile | 假发射 | 改为共享物理轴 tile 的机械消费 |
| cuTile runtime-sized matrix-M lane | 真实 target 边界 | 源码位置明确 unsupported |
| 本机 Triton absorbed MLA shared OOR | 当前设备资源边界 | 不改算法、不写 target 特判，记 `not_measured` |
| TileLang absorbed MLA 无合法配置 | 下层编译边界 | 不构造第二套决策路径，记 `not_measured` |
| sparse MLA 三家 JIT 未完成 | 尚不能定性的下层压力 | 如实记录，既不写 PASS，也不提前宣称永久 unsupported |

本轮 compiler diff 中没有出现 `absorbed_mla`、`token_sparse_mla`、`fp8_mqa` 或其他 kernel 名判断。所有新增分叉只依据 op、index relation、axis role、transfer space 和 target capability。

一次提交前审计曾尝试在共享 GPU analysis 中拒绝所有“从局部 tensor 做 indirect gather”的关系；`nonzero_compact` 手动 repro 立即证明这会误伤 scan 的专门物化路径，因此该全局限制已撤回。最终保留的是叶子在自身通用路径上的精确 capability check，而不是一个把不同消费方式混在一起的总开关。

## 五、当时无法完全判断的取舍

1. token-sparse MLA 的全 invalid selection 行没有进入这次输入；当前 DSL kernel 定义了 safe denominator，PyTorch reference 的朴素 `-inf - -inf` 会产生非有限值。真实运行输入每行保留 254 个合法 token，因此本轮数值合同一致；没有为未被当前算法输入制造的情形扩充验证设施。
2. sparse MLA 的三个超时不能仅凭表象归为共享计划缺失。三家都已拿到同一 index relation 与 stream 计划，失败发生在各自下层编译/搜索；在没有一条等价且更机械的目标投影证据前，本轮不往共享层增加机制。
3. absorbed MLA 选择保留三次作者调用，而不是把投影吸收到 attention kernel。后者可能更快，但会改变作者写下的调用结构，违反一份源码对应一次调用的边界。
4. 本轮没有再选树形 speculative attention 或新的 chunk SSM。已有语料已经有显式 mask attention、Mamba chunk scan；相比之下，吸收 MLA、学习式 token sparse 和 FP8 MQA 同时带来了当前语料没有闭合的 index/broadcast/row-vector 组合，因此优先级更高。

## 六、验证与两台机器表

验证仍只使用可手工执行的一条入口：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

本轮实际运行了五条新增记录可完成的 provider，并复验了共享判定的受影响读者：cuTile MLA projection、Triton row index-select、TileLang FP8 MQA，以及此前完成的 GEMM、grouped GEMM 和三目标 index-select。数值通过项没有退化；外置 C++ build `/tmp/intentdsl-build` 完整编译成功，没有在项目内创建构建目录、测试目录或 fixture。

两份表现在各有 106 个 case：

- RTX 5090 D：`report/baseline/kernel-performance.csv`；
- H100：`report/baseline/kernel-performance-h100.csv`。

H100 当前有外部 VLLM 占用约 78 GiB，远端工作树又不是这次未提交代码的干净镜像。为了不覆盖远端状态、不终止用户服务，也不复制本机数字，5 条新记录在 H100 表全部明确为 `not_measured`。旧 H100 101 行及其 source 数字没有改动。
