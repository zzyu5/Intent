# TritonBench-T Agent Results

Codex / gpt-5.6-luna / max. Fixed denominator: 50 tasks x 3 independent repetitions per arm.
Current coverage: triton: 50/150 generation trials attempted, 43 first-correct, 50 budget-correct, 100 not run; intent: 50/150 generation trials attempted, 29 first-correct, 45 budget-correct, 100 not run. Not-run repetitions are pending, not observed failures. Fixed-denominator rates are incomplete until coverage is complete.
Times are median CUDA Graph operator milliseconds among correct repetitions; `-` means no correct measured program.
The PyTorch task implementation is a performance anchor, not an optimized Triton upper bound. Failures remain in the denominator.

Uncapturable references remain unchanged numerical oracles; candidate CUDA Graph times remain usable without a reference ratio. Reference-only evaluation repairs preserve raw observations and original agent programs. Canceled calls without a submission retain their cost but do not consume a submission slot.

Budgets include the work actually incurred, including retries after erroneous evaluator feedback. Repaired trials are flagged in summary.csv; their observed costs are not an estimate of an error-free workflow. The candidate time limit includes preparation and GPU queue time, not only device execution.

Optimization columns show the best correct program after optimization starts, including the unchanged seed when it remains best; they are not a claim of improvement. Stage and stopping status are in summary.csv.

paired.csv compares Intent/direct absolute operator times without requiring a PyTorch timing anchor. Paired performance is conditional on both arms succeeding; all failures remain in the separate fixed-denominator correctness results.

budget.csv records best-so-far absolute time and cumulative stage/full-workflow costs. Its common performance target is the median PyTorch seed-reference time of the two paired arms; no paired target is reported when either seed is missing. Cached input tokens are a subset of input tokens. Missing provider usage is disclosed, not estimated.

| Task | Direct correct / 3 | Intent correct / 3 | Direct seed ms | Intent static ms | Direct best after optimization ms | Intent-start best after optimization ms |
|---|---:|---:|---:|---:|---:|---:|
| abs | 1 | 1 | 0.007280 | 0.006200 | - | - |
| asin | 1 | 1 | 0.005920 | 0.006784 | - | - |
| cos | 1 | 1 | 0.007936 | 0.006408 | - | - |
| exp | 1 | 1 | 0.005936 | 0.005952 | - | - |
| floor | 1 | 1 | 0.007792 | 0.006192 | - | - |
| gelu | 1 | 1 | 0.007808 | 0.006800 | - | - |
| leaky_relu | 1 | 1 | 0.007376 | 0.006880 | - | - |
| log | 1 | 1 | 0.007792 | 0.006144 | - | - |
| log1p | 1 | 1 | 0.006864 | 0.007984 | - | - |
| mul | 1 | 1 | 0.007600 | 0.006880 | - | - |
| relu | 1 | 1 | 0.005920 | 0.006128 | - | - |
| rsqrt | 1 | 1 | 0.005888 | 0.007488 | 0.005536 | 0.006144 |
| sigmoid | 1 | 1 | 0.007824 | 0.007840 | - | - |
| sqrt | 1 | 1 | 0.005840 | 0.006128 | 0.005840 | 0.005904 |
| tanh | 1 | 1 | 0.006704 | 0.007968 | 0.005920 | 0.006160 |
| argmax | 1 | 1 | 0.007968 | 0.139072 | - | - |
| max | 1 | 1 | 0.009320 | 0.228368 | - | - |
| mean | 1 | 1 | 0.007840 | 0.110368 | - | - |
| min | 1 | 1 | 0.142432 | 0.007968 | - | - |
| sum | 1 | 1 | 0.007952 | 0.110384 | - | - |
| std | 1 | 1 | 0.008200 | 0.192288 | - | - |
| softmax | 1 | 1 | 0.007960 | 0.007824 | 0.007824 | - |
| logsumexp | 1 | 1 | 0.008848 | 0.201448 | - | - |
| add_mean | 1 | 1 | 0.011320 | 0.177840 | - | - |
| exp_mean | 1 | 1 | 0.007840 | 0.120104 | - | - |
| matmul | 1 | 1 | 0.028672 | 0.018208 | 0.016016 | - |
| addmm | 1 | 1 | 0.032544 | 0.018224 | - | - |
| tensordot | 1 | 0 | 0.006064 | - | - | - |
| tensordot_rsqrt | 1 | 1 | 0.046768 | 0.044832 | - | - |
| conv2d | 1 | 0 | 0.090808 | - | - | - |
| symmetric_mm_and_abs_sum | 1 | 0 | 0.141352 | - | - | - |
| fused_mv_sigmoid_sub | 1 | 1 | 0.007840 | 0.018208 | - | - |
| solve | 1 | 1 | 6.626768 | 241.674667 | - | - |
| fused_cholesky_solve | 1 | 1 | 7.501408 | 42.763870 | - | - |
| fused_qr_solve | 1 | 0 | 16.265184 | - | - | - |
| add_gelu | 1 | 1 | 0.007816 | 0.006856 | - | - |
| mul_relu | 1 | 1 | 0.010000 | 0.010032 | - | - |
| mul_sub | 1 | 1 | 0.012064 | 0.012080 | - | - |
| sub_gelu | 1 | 1 | 0.010016 | 0.010032 | - | - |
| relu_conv2d | 1 | 1 | 0.012048 | 1.902352 | - | - |
| gelu_conv2d | 1 | 0 | 0.063280 | - | - | - |
| conv2d_add | 1 | 1 | 0.031488 | 1.702560 | - | - |
| combined_activation | 1 | 1 | 0.044704 | 0.014112 | - | - |
| fused_layer_norm_relu_linear | 1 | 1 | 0.173872 | 0.202496 | - | - |
| softplus_linear | 1 | 1 | 0.007840 | 0.053024 | - | - |
| permute_copy | 1 | 1 | 0.005928 | 0.016144 | - | - |
| ifftshift | 1 | 1 | 0.006152 | 0.006440 | - | - |
| grid_sample | 1 | 1 | 0.028320 | 0.030480 | 0.028320 | 0.030480 |
| fused_gather_masked_fill | 1 | 1 | 0.028384 | 0.018096 | - | - |
| fused_index_select_eq | 1 | 1 | 0.003872 | 0.003872 | 0.003872 | 0.003872 |

Unchanged direct Triton programs, measurements and their actual budgets were reused from: report/agent-tritonbench/expanded. Tasks with changed reference contracts were not imported.

Project-local reference corrections: sub_gelu: sub_gelu_standard_gelu. Upstream files are unchanged; sub_gelu restores the missing 1 + in the exact GELU formula.

## Compiler Rechecks

Human compiler development, reusing the original agent DSL. These measurements do not change agent first-submission correctness, budgets or optimization seeds above.

| Task | Before | Current ms | Reference ms | Current / reference | Status |
|---|---|---:|---:|---:|---|
| fused_gather_masked_fill | flat M*N domain had no ABI-dimension ownership extent | 0.018096 | 0.026272 | 0.688794 | pass |
| relu_conv2d | 1.934112 ms; scalar output ownership despite an explicit reduce graph | 0.034528 | 0.034512 | 1.000464 | pass |
| conv2d | 0.967440 ms; ordered scalar accumulation and one output per program instance | 0.090928 | 0.034608 | 2.627369 | pass |
| conv2d_add | 1.702560 ms; ordered scalar accumulation | 0.096048 | 0.035392 | 2.713834 | pass |
| softmax | pure-reduce free-axis graph excluded from structured blocking | 0.007952 | 0.007960 | 0.998995 | pass |
| tensordot | equivalent reduction steps rejected because they were distinct SSA values | 0.007952 | 0.005888 | 1.350543 | pass |
| matmul | f16 accumulator rejected by the installed Triton float32 dot default | 0.016160 | 0.018208 | 0.887522 | pass |

Compiler commits, original programs and ref comparisons are recorded in compiler-rechecks/index.json.
