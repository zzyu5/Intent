# TritonBench-T Agent Results

Codex / gpt-5.6-luna / max. Fixed denominator: 50 tasks x 3 independent repetitions per arm.
Times are median CUDA Graph operator milliseconds among correct repetitions; `-` means no correct measured program.
The PyTorch task implementation is a performance anchor, not an optimized Triton upper bound. Failures remain in the denominator.

Optimization columns show the best correct program after optimization starts, including the unchanged seed when it remains best; they are not a claim of improvement. Stage and stopping status are in summary.csv.

budget.csv records best-so-far absolute time and cumulative stage/full-workflow costs. Its common performance target is the median PyTorch seed-reference time of the two paired arms; no paired target is reported when either seed is missing. Cached input tokens are a subset of input tokens. Missing provider usage is disclosed, not estimated.

| Task | Direct correct / 3 | Intent correct / 3 | Direct seed ms | Intent static ms | Direct best after optimization ms | Intent-start best after optimization ms |
|---|---:|---:|---:|---:|---:|---:|
| abs | 1 | 1 | 0.006280 | 0.005928 | 0.005824 | 0.005920 |
| asin | 0 | 0 | - | - | - | - |
| cos | 0 | 0 | - | - | - | - |
| exp | 0 | 0 | - | - | - | - |
| floor | 0 | 0 | - | - | - | - |
| gelu | 0 | 0 | - | - | - | - |
| leaky_relu | 0 | 0 | - | - | - | - |
| log | 0 | 0 | - | - | - | - |
| log1p | 0 | 0 | - | - | - | - |
| mul | 0 | 0 | - | - | - | - |
| relu | 0 | 0 | - | - | - | - |
| rsqrt | 0 | 0 | - | - | - | - |
| sigmoid | 0 | 0 | - | - | - | - |
| sqrt | 0 | 0 | - | - | - | - |
| tanh | 0 | 0 | - | - | - | - |
| argmax | 0 | 0 | - | - | - | - |
| max | 0 | 0 | - | - | - | - |
| mean | 0 | 0 | - | - | - | - |
| min | 0 | 0 | - | - | - | - |
| sum | 0 | 0 | - | - | - | - |
| std | 0 | 0 | - | - | - | - |
| softmax | 1 | 1 | 0.007824 | 0.024352 | 0.007704 | 0.007840 |
| logsumexp | 0 | 0 | - | - | - | - |
| add_mean | 0 | 0 | - | - | - | - |
| exp_mean | 0 | 0 | - | - | - | - |
| matmul | 1 | 1 | 0.030352 | 0.018192 | 0.017568 | 0.016368 |
| addmm | 0 | 0 | - | - | - | - |
| tensordot | 0 | 0 | - | - | - | - |
| tensordot_rsqrt | 0 | 0 | - | - | - | - |
| conv2d | 0 | 0 | - | - | - | - |
| symmetric_mm_and_abs_sum | 0 | 0 | - | - | - | - |
| fused_mv_sigmoid_sub | 0 | 0 | - | - | - | - |
| solve | 0 | 0 | - | - | - | - |
| fused_cholesky_solve | 0 | 0 | - | - | - | - |
| fused_qr_solve | 0 | 0 | - | - | - | - |
| add_gelu | 0 | 0 | - | - | - | - |
| mul_relu | 0 | 0 | - | - | - | - |
| mul_sub | 0 | 0 | - | - | - | - |
| sub_gelu | 0 | 0 | - | - | - | - |
| relu_conv2d | 0 | 0 | - | - | - | - |
| gelu_conv2d | 0 | 0 | - | - | - | - |
| conv2d_add | 0 | 0 | - | - | - | - |
| combined_activation | 0 | 0 | - | - | - | - |
| fused_layer_norm_relu_linear | 0 | 0 | - | - | - | - |
| softplus_linear | 0 | 0 | - | - | - | - |
| permute_copy | 0 | 0 | - | - | - | - |
| ifftshift | 0 | 0 | - | - | - | - |
| grid_sample | 0 | 0 | - | - | - | - |
| fused_gather_masked_fill | 0 | 0 | - | - | - | - |
| fused_index_select_eq | 0 | 0 | - | - | - | - |

## Compiler Rechecks

Human compiler development, reusing the original agent DSL. These measurements do not change agent first-submission correctness, budgets or optimization seeds above.

| Task | Before | Current ms | Reference ms | Current / reference | Status |
|---|---|---:|---:|---:|---|
| matmul / full-result indexing | compiler rejected a legal full-domain gather | 0.018176 | 0.018208 | 0.998243 | pass |
| matmul / direct-result store | numerical failure, max_abs=289.25 | 0.018208 | 0.017168 | 1.060578 | pass |
| softmax | 0.024352 ms | 0.007984 | 0.007984 | 1.000000 | pass |

Compiler commits, original programs and ref comparisons are recorded in compiler-rechecks/index.json.
