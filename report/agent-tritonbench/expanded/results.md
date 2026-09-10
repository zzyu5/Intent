# TritonBench-T Agent Results

Codex / gpt-5.6-luna / max. Fixed denominator: 50 tasks x 3 independent repetitions per arm.
Current coverage: triton: 50/150 generation trials attempted, 42 first-correct, 49 budget-correct, 100 not run; intent: 50/150 generation trials attempted, 21 first-correct, 40 budget-correct, 100 not run. Not-run repetitions are pending, not observed failures. Fixed-denominator rates are incomplete until coverage is complete.
Times are median CUDA Graph operator milliseconds among correct repetitions; `-` means no correct measured program.
The PyTorch task implementation is a performance anchor, not an optimized Triton upper bound. Failures remain in the denominator.

Uncapturable references remain unchanged numerical oracles; candidate CUDA Graph times remain usable without a reference ratio. Reference-only evaluation repairs preserve raw observations and original agent programs. Canceled calls without a submission retain their cost but do not consume a submission slot.

Budgets include the work actually incurred, including retries after erroneous reference-only feedback. Repaired trials are flagged in summary.csv; their observed costs are not an estimate of an error-free workflow. The candidate time limit includes preparation and GPU queue time, not only device execution.

Optimization columns show the best correct program after optimization starts, including the unchanged seed when it remains best; they are not a claim of improvement. Stage and stopping status are in summary.csv.

paired.csv compares Intent/direct absolute operator times without requiring a PyTorch timing anchor. Paired performance is conditional on both arms succeeding; all failures remain in the separate fixed-denominator correctness results.

budget.csv records best-so-far absolute time and cumulative stage/full-workflow costs. Its common performance target is the median PyTorch seed-reference time of the two paired arms; no paired target is reported when either seed is missing. Cached input tokens are a subset of input tokens. Missing provider usage is disclosed, not estimated.

| Task | Direct correct / 3 | Intent correct / 3 | Direct seed ms | Intent static ms | Direct best after optimization ms | Intent-start best after optimization ms |
|---|---:|---:|---:|---:|---:|---:|
| abs | 1 | 1 | 0.007280 | 0.005920 | - | - |
| asin | 1 | 1 | 0.005920 | 0.007872 | - | - |
| cos | 1 | 1 | 0.007936 | 0.006880 | - | - |
| exp | 1 | 1 | 0.005936 | 0.005952 | - | - |
| floor | 1 | 1 | 0.007792 | 0.006680 | - | - |
| gelu | 1 | 1 | 0.007808 | 0.006128 | - | - |
| leaky_relu | 1 | 1 | 0.007376 | 0.006136 | - | - |
| log | 1 | 1 | 0.007792 | 0.006000 | - | - |
| log1p | 1 | 1 | 0.006864 | 0.006864 | - | - |
| mul | 1 | 1 | 0.007600 | 0.006272 | - | - |
| relu | 1 | 1 | 0.005920 | 0.006752 | - | - |
| rsqrt | 1 | 1 | 0.005888 | 0.006264 | - | - |
| sigmoid | 1 | 1 | 0.007824 | 0.006192 | - | - |
| sqrt | 1 | 1 | 0.005840 | 0.006152 | - | - |
| tanh | 1 | 1 | 0.006704 | 0.007824 | - | - |
| argmax | 1 | 0 | 0.007968 | - | - | - |
| max | 1 | 0 | 0.009320 | - | - | - |
| mean | 1 | 1 | 0.007840 | 0.113296 | - | - |
| min | 1 | 0 | 0.142432 | - | - | - |
| sum | 1 | 0 | 0.007952 | - | - | - |
| std | 1 | 1 | 0.008200 | 0.153360 | - | - |
| softmax | 1 | 1 | 0.007960 | 0.007824 | 0.007824 | 0.007824 |
| logsumexp | 1 | 1 | 0.008848 | 0.008384 | - | - |
| add_mean | 1 | 1 | 0.011320 | 0.018224 | - | - |
| exp_mean | 1 | 1 | 0.007840 | 0.007968 | - | - |
| matmul | 1 | 1 | 0.028672 | 0.018208 | 0.016016 | 0.016056 |
| addmm | 1 | 1 | 0.032544 | 0.018408 | - | - |
| tensordot | 1 | 1 | 0.006064 | 0.007944 | - | - |
| tensordot_rsqrt | 1 | 1 | 0.046768 | 0.044720 | - | - |
| conv2d | 1 | 1 | 0.090808 | 0.969272 | - | - |
| symmetric_mm_and_abs_sum | 1 | 0 | 0.141352 | - | - | - |
| fused_mv_sigmoid_sub | 1 | 1 | 0.007840 | 0.018080 | - | - |
| solve | 1 | 0 | 6.626768 | - | - | - |
| fused_cholesky_solve | 1 | 1 | 7.501408 | 28.984287 | - | - |
| fused_qr_solve | 1 | 0 | 16.265184 | - | - | - |
| add_gelu | 1 | 1 | 0.007816 | 0.006896 | - | - |
| mul_relu | 1 | 1 | 0.010000 | 0.009888 | - | - |
| mul_sub | 1 | 1 | 0.012064 | 0.012064 | - | - |
| sub_gelu | 0 | 0 | - | - | - | - |
| relu_conv2d | 1 | 1 | 0.012048 | 1.933952 | - | - |
| gelu_conv2d | 1 | 1 | 0.063280 | 3.651816 | - | - |
| conv2d_add | 1 | 1 | 0.031488 | 1.669920 | - | - |
| combined_activation | 1 | 1 | 0.044704 | 0.014128 | - | - |
| fused_layer_norm_relu_linear | 1 | 1 | 0.173872 | 0.203840 | - | - |
| softplus_linear | 1 | 0 | 0.007840 | - | - | - |
| permute_copy | 1 | 1 | 0.005928 | 0.016160 | - | - |
| ifftshift | 1 | 1 | 0.006152 | 0.005880 | - | - |
| grid_sample | 1 | 1 | 0.028320 | 0.358176 | 0.028320 | 0.030352 |
| fused_gather_masked_fill | 1 | 0 | 0.028384 | - | - | - |
| fused_index_select_eq | 1 | 1 | 0.003872 | 0.003760 | 0.003872 | 0.003760 |

## Reference Issues

- sub_gelu (superseded_by_corrected_reference): The default branch computes 0.5*x*erf(x/sqrt(2)), omitting the 1 + required by GELU. Both arms implement standard GELU and repeatedly differ from this reference by max_abs=0.49934613704681396. These numerical failures cannot be attributed to agent or compiler arithmetic without resolving the reference/task conflict. The user authorized continued implementation after the proposed correction. The project evaluator restores the missing 1 + in a new batch; this batch retains its original reference-contract failure and raw observations. The upstream source is unchanged.

## Compiler Rechecks

Human compiler development, reusing the original agent DSL. These measurements do not change agent first-submission correctness, budgets or optimization seeds above.

| Task | Before | Current ms | Reference ms | Current / reference | Status |
|---|---|---:|---:|---:|---|
| grid_sample | 0.358032 ms; one scalar program per output element | 0.028320 | 0.028368 | 0.998308 | pass |
| mean | no legal full-coverage extent for FULL_D1 at 1048576 elements | 0.110240 | 0.010424 | 10.575595 | pass |
| logsumexp | no legal full-coverage extent for FULL_D1 | 0.208688 | 0.028464 | 7.331647 | pass |
| fused_index_select_eq | Triton rejected int1 values for an int8 block pointer store | 0.003888 | 0.005936 | 0.654987 | pass |
| max | rank-zero output rejected; builtin arg_reduce incorrectly required author identity | 0.228128 | 0.011568 | 19.720609 | pass |
| min | builtin arg_reduce required identity; negated whole view was not loaded | 0.007984 | 0.014816 | 0.538877 | pass |
| sum | rank-zero output rejected | 0.110240 | 0.010472 | 10.527120 | pass |
| softplus_linear | broadcast consumed an unloaded view; bias ownership and physical broadcast shape were inconsistent | 0.052992 | 0.018128 | 2.923213 | pass |

Compiler commits, original programs and ref comparisons are recorded in compiler-rechecks/index.json.
