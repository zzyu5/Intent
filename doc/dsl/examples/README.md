# 理想化 DSL 示例

这些文件展示作者应当表达的算法，不展示当前 compiler 的迁就写法。它们刻意不包含 physical tile、`I.auto`、program id、warp、storage scope、target primitive 或 autotune config。

- [`pointwise.py`](pointwise.py)：完整 logical domain 上的纯张量定义；
- [`reduction.py`](reduction.py)：generic reduce 与 typed combine；
- [`gemm.py`](gemm.py)：完整域 contraction 与算法 constexpr 分支；
- [`online_softmax.py`](online_softmax.py)：compiler-selected segmentation 的 state stream；
- [`ragged_grouped_gemm.py`](ragged_grouped_gemm.py)：ragged relation、gather、contract 与 unique scatter；
- [`split_k_pipeline.py`](split_k_pipeline.py)：ABI 可见 parts 和两个 kernel 的 host orchestration。

示例是编程模型的规范性说明，不是测试 fixture，也不承诺当前 frontend 已接受全部拼写。
