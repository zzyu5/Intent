# 理想化 DSL 示例

这些文件展示作者应当表达的算法，不展示为迁就某一实现而加入的写法。它们刻意不包含 physical tile、`I.auto`、program id、warp、storage scope、target primitive 或 autotune config。

- [`pointwise.py`](pointwise.py)：完整 logical domain 上的纯张量定义；
- [`reduction.py`](reduction.py)：generic reduce 与 typed combine；
- [`gemm.py`](gemm.py)：完整域 contraction 与算法 constexpr 分支；
- [`online_softmax.py`](online_softmax.py)：用 typed record reduction 表达 online-softmax summary algebra，不使用 `state_stream`；
- [`ragged_grouped_gemm.py`](ragged_grouped_gemm.py)：用可机械展开的 ragged helper 表达 offsets/subregion/index relation，再进行 gather、contract 与 unique scatter；
- [`split_k_pipeline.py`](split_k_pipeline.py)：不用 `partition`，显式写 part domain、boundary arithmetic、source subregion、partial tensor 与两个 kernels 的 host orchestration。

示例是编程模型的规范性说明，不是测试 fixture；其中每种surface spelling都必须机械归一到对应的canonical semantics。
