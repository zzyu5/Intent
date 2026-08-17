from __future__ import annotations

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.cache.reshape_and_cache import BLOCKS as CACHE_BLOCKS
from kernels.cache.reshape_and_cache import BLOCK_SIZE as CACHE_BLOCK_SIZE
from kernels.cache.reshape_and_cache import HEAD_DIMENSION as CACHE_HEAD_DIMENSION
from kernels.cache.reshape_and_cache import HEADS as CACHE_HEADS
from kernels.cache.reshape_and_cache import TOKENS as CACHE_TOKENS
from kernels.cache.reshape_and_cache import reshape_and_cache
from kernels.factorization.cholesky import BATCH as CHOLESKY_BATCH
from kernels.factorization.cholesky import SIZE as CHOLESKY_SIZE
from kernels.factorization.cholesky import batched_cholesky_lower
from kernels.optimization.adafactor import COLUMNS as ADAFACTOR_COLUMNS
from kernels.optimization.adafactor import ROWS as ADAFACTOR_ROWS
from kernels.optimization.adafactor import adafactor_apply
from kernels.optimization.adamw import PARAMETERS as ADAMW_PARAMETERS
from kernels.optimization.adamw import adamw_update
from kernels.ragged.nested_pool import DOCUMENTS as NESTED_DOCUMENTS
from kernels.ragged.nested_pool import FEATURES as NESTED_FEATURES
from kernels.ragged.nested_pool import SENTENCES as NESTED_SENTENCES
from kernels.ragged.nested_pool import TOKENS as NESTED_TOKENS
from kernels.ragged.nested_pool import nested_jagged_mean_pool
from kernels.routing.moe_align import EXPERTS as MOE_ALIGN_EXPERTS
from kernels.routing.moe_align import TOKENS as MOE_ALIGN_TOKENS
from kernels.routing.moe_align import TOP_K as MOE_ALIGN_TOP_K
from kernels.routing.moe_align import moe_count_routes
from kernels.streaming.attention import HEAD_DIMENSION
from kernels.streaming.attention import SCALE as ATTENTION_SCALE
from kernels.streaming.attention import flash_attention_fwd
from kernels.streaming.ordered_prefix import BATCH as PREFIX_BATCH
from kernels.streaming.ordered_prefix import COLUMNS as PREFIX_COLUMNS
from kernels.streaming.ordered_prefix import ROWS as PREFIX_ROWS
from kernels.streaming.ordered_prefix import ordered_product_prefix
from kernels.variants.decomposition import adafactor_apply_scalar_product
from kernels.variants.decomposition import adamw_update_moments
from kernels.variants.decomposition import adamw_update_parameter
from kernels.variants.decomposition import batched_cholesky_right_looking
from kernels.variants.decomposition import matrix_transpose_product_domains
from kernels.variants.decomposition import moe_count_routes_product_domain
from kernels.variants.decomposition import nested_document_pool
from kernels.variants.decomposition import nested_jagged_mean_pool_identity
from kernels.variants.decomposition import nested_sentence_pool
from kernels.variants.decomposition import ordered_prefix_nested
from kernels.variants.decomposition import reshape_key_cache
from kernels.variants.decomposition import reshape_value_cache
from kernels.variants.layout import matrix_transpose_scalar_domains
from kernels.variants.streaming import flash_attention_full_causal_stream_fwd
from kernels.layout.transpose import COLUMNS as TRANSPOSE_COLUMNS
from kernels.layout.transpose import ROWS as TRANSPOSE_ROWS

from .evaluation import Runner
from .evaluation import compare_variant_outputs
from .evaluation import outputs
from .evaluation import report_variant_pipeline
from .evaluation import run_variant
from .support import prepare_kernel_call


def _run_moe_product_domain(
    compiler: str, target: Target, target_name: str
) -> None:
    topk_ids = torch.randint(
        0,
        MOE_ALIGN_EXPERTS,
        (MOE_ALIGN_TOKENS, MOE_ALIGN_TOP_K),
        device="cuda",
        dtype=torch.int32,
    )
    original_counts = torch.zeros(
        (MOE_ALIGN_EXPERTS,), device="cuda", dtype=torch.int32
    )
    variant_counts = torch.zeros_like(original_counts)
    expected = torch.bincount(
        topk_ids.flatten().long(), minlength=MOE_ALIGN_EXPERTS
    ).to(torch.int32)
    original_artifact = intent.compile(
        moe_count_routes, target=target, compiler=compiler
    )
    variant_artifact = intent.compile(
        moe_count_routes_product_domain, target=target, compiler=compiler
    )
    original_artifact.run(topk_ids, original_counts)
    variant_artifact.run(topk_ids, variant_counts)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_counts,
        variant=variant_counts,
        reference=expected,
        tolerance=0.0,
        target_name=target_name,
        kernel_name="MoE nested versus product-domain counting",
    )
    original_call = prepare_kernel_call(
        original_artifact, (topk_ids, original_counts), ()
    )
    variant_call = prepare_kernel_call(
        variant_artifact, (topk_ids, variant_counts), ()
    )
    report_variant_pipeline(
        variant_artifacts=(variant_artifact,),
        original_launch=original_call,
        variant_launch=variant_call,
        original_prepare=original_counts.zero_,
        variant_prepare=variant_counts.zero_,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="MoE nested versus product-domain counting",
        performance_scope="kernel-only",
    )


def _run_adamw_split_pipeline(
    compiler: str, target: Target, target_name: str
) -> None:
    gradient = torch.randn(
        (ADAMW_PARAMETERS,), device="cuda", dtype=torch.float32
    ) * 0.01
    initial_parameter = torch.randn_like(gradient)
    initial_first = torch.randn_like(gradient) * 0.01
    initial_second = torch.rand_like(gradient) * 0.01
    original_parameter = initial_parameter.clone()
    original_first = initial_first.clone()
    original_second = initial_second.clone()
    variant_parameter = initial_parameter.clone()
    variant_first = initial_first.clone()
    variant_second = initial_second.clone()
    expected_parameter = initial_parameter.clone()
    expected_first = initial_first.clone()
    expected_second = initial_second.clone()
    learning_rate, beta1, beta2 = 1.0e-3, 0.9, 0.999
    inverse_bias1, inverse_bias2 = 10.0, 1000.0
    epsilon, weight_decay = 1.0e-8, 0.01
    expected_first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    expected_second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    expected_parameter.mul_(1.0 - learning_rate * weight_decay).add_(
        expected_first
        * inverse_bias1
        / (torch.sqrt(expected_second * inverse_bias2) + epsilon),
        alpha=-learning_rate,
    )
    original_artifact = intent.compile(adamw_update, target=target, compiler=compiler)
    moments_artifact = intent.compile(
        adamw_update_moments, target=target, compiler=compiler
    )
    parameter_artifact = intent.compile(
        adamw_update_parameter, target=target, compiler=compiler
    )
    original_arguments = (
        gradient, original_parameter, original_first, original_second,
        learning_rate, beta1, beta2, inverse_bias1, inverse_bias2,
        epsilon, weight_decay,
    )
    moments_arguments = (gradient, variant_first, variant_second, beta1, beta2)
    parameter_arguments = (
        variant_parameter, variant_first, variant_second, learning_rate,
        inverse_bias1, inverse_bias2, epsilon, weight_decay,
    )
    original_artifact.run(*original_arguments)
    moments_artifact.run(*moments_arguments)
    parameter_artifact.run(*parameter_arguments)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=(original_parameter, original_first, original_second),
        variant=(variant_parameter, variant_first, variant_second),
        reference=(expected_parameter, expected_first, expected_second),
        tolerance=(2.0e-6, 2.0e-6, 2.0e-6),
        target_name=target_name,
        kernel_name="AdamW fused versus two-stage update",
    )
    original_call = prepare_kernel_call(original_artifact, original_arguments, ())
    moments_call = prepare_kernel_call(moments_artifact, moments_arguments, ())
    parameter_call = prepare_kernel_call(parameter_artifact, parameter_arguments, ())

    def original_launch() -> None:
        original_call()

    def variant_launch() -> None:
        moments_call()
        parameter_call()

    def prepare_original() -> None:
        original_parameter.copy_(initial_parameter)
        original_first.copy_(initial_first)
        original_second.copy_(initial_second)

    def prepare_variant() -> None:
        variant_parameter.copy_(initial_parameter)
        variant_first.copy_(initial_first)
        variant_second.copy_(initial_second)

    report_variant_pipeline(
        variant_artifacts=(moments_artifact, parameter_artifact),
        original_launch=original_launch,
        variant_launch=variant_launch,
        original_prepare=prepare_original,
        variant_prepare=prepare_variant,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="AdamW fused versus two-stage update",
        performance_scope="end-to-end GPU pipeline",
    )


def _run_reshape_cache_split(
    compiler: str, target: Target, target_name: str
) -> None:
    key = torch.randn(
        (CACHE_TOKENS, CACHE_HEADS, CACHE_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    slots = torch.randperm(
        CACHE_BLOCKS * CACHE_BLOCK_SIZE, device="cuda", dtype=torch.int64
    )[:CACHE_TOKENS].to(torch.int32)
    cache_shape = (
        CACHE_BLOCKS, CACHE_BLOCK_SIZE, CACHE_HEADS, CACHE_HEAD_DIMENSION
    )
    original_key = torch.zeros(cache_shape, device="cuda", dtype=torch.float16)
    original_value = torch.zeros_like(original_key)
    variant_key = torch.zeros_like(original_key)
    variant_value = torch.zeros_like(original_key)
    expected_key = torch.zeros_like(original_key)
    expected_value = torch.zeros_like(original_value)
    blocks = slots.long() // CACHE_BLOCK_SIZE
    offsets = slots.long() % CACHE_BLOCK_SIZE
    expected_key[blocks, offsets] = key
    expected_value[blocks, offsets] = value
    original_artifact = intent.compile(
        reshape_and_cache, target=target, compiler=compiler
    )
    key_artifact = intent.compile(reshape_key_cache, target=target, compiler=compiler)
    value_artifact = intent.compile(
        reshape_value_cache, target=target, compiler=compiler
    )
    original_arguments = (key, value, slots, original_key, original_value)
    key_arguments = (key, slots, variant_key)
    value_arguments = (value, slots, variant_value)
    original_artifact.run(*original_arguments)
    key_artifact.run(*key_arguments)
    value_artifact.run(*value_arguments)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=(original_key, original_value),
        variant=(variant_key, variant_value),
        reference=(expected_key, expected_value),
        tolerance=(0.0, 0.0),
        target_name=target_name,
        kernel_name="reshape-and-cache fused versus split writes",
    )
    original_call = prepare_kernel_call(original_artifact, original_arguments, ())
    key_call = prepare_kernel_call(key_artifact, key_arguments, ())
    value_call = prepare_kernel_call(value_artifact, value_arguments, ())

    def original_launch() -> None:
        original_call()

    def variant_launch() -> None:
        key_call()
        value_call()

    def prepare_original() -> None:
        original_key.zero_()
        original_value.zero_()

    def prepare_variant() -> None:
        variant_key.zero_()
        variant_value.zero_()

    report_variant_pipeline(
        variant_artifacts=(key_artifact, value_artifact),
        original_launch=original_launch,
        variant_launch=variant_launch,
        original_prepare=prepare_original,
        variant_prepare=prepare_variant,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="reshape-and-cache fused versus split writes",
        performance_scope="end-to-end GPU pipeline",
    )


def _nested_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    document_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(NESTED_DOCUMENTS)],
        device="cuda",
        dtype=torch.int32,
    )
    sentence_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(NESTED_SENTENCES)],
        device="cuda",
        dtype=torch.int32,
    )
    document_offsets = torch.zeros(
        (NESTED_DOCUMENTS + 1,), device="cuda", dtype=torch.int32
    )
    sentence_offsets = torch.zeros(
        (NESTED_SENTENCES + 1,), device="cuda", dtype=torch.int32
    )
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn(
        (NESTED_TOKENS, NESTED_FEATURES),
        device="cuda",
        dtype=torch.float32,
    )
    return (
        document_lengths, sentence_lengths, document_offsets, sentence_offsets, values
    )


def _nested_reference(
    document_lengths: torch.Tensor,
    sentence_lengths: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sentence_ids = torch.repeat_interleave(
        torch.arange(NESTED_SENTENCES, device="cuda"), sentence_lengths.long()
    )
    sentence_sums = torch.zeros(
        (NESTED_SENTENCES, NESTED_FEATURES), device="cuda"
    )
    sentence_sums.index_add_(0, sentence_ids, values)
    sentence_means = sentence_sums / sentence_lengths[:, None]
    document_ids = torch.repeat_interleave(
        torch.arange(NESTED_DOCUMENTS, device="cuda"), document_lengths.long()
    )
    document_sums = torch.zeros(
        (NESTED_DOCUMENTS, NESTED_FEATURES), device="cuda"
    )
    document_sums.index_add_(0, document_ids, sentence_sums)
    document_tokens = torch.zeros(
        (NESTED_DOCUMENTS,), device="cuda", dtype=torch.int32
    )
    document_tokens.index_add_(0, document_ids, sentence_lengths)
    return sentence_means, document_sums / document_tokens[:, None]


def _run_nested_ragged_identity(
    compiler: str, target: Target, target_name: str
) -> None:
    (
        document_lengths, sentence_lengths, document_offsets, sentence_offsets, values
    ) = _nested_inputs()
    document_indices = torch.arange(
        NESTED_SENTENCES, device="cuda", dtype=torch.int32
    )
    token_indices = torch.arange(
        NESTED_TOKENS, device="cuda", dtype=torch.int32
    )
    expected = _nested_reference(document_lengths, sentence_lengths, values)
    original_artifact = intent.compile(
        nested_jagged_mean_pool, target=target, compiler=compiler
    )
    variant_artifact = intent.compile(
        nested_jagged_mean_pool_identity, target=target, compiler=compiler
    )
    original_arguments = (document_offsets, sentence_offsets, values)
    variant_arguments = (
        document_offsets, sentence_offsets, document_indices, token_indices, values
    )
    original_output = original_artifact.run(*original_arguments)
    variant_output = variant_artifact.run(*variant_arguments)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_output,
        variant=variant_output,
        reference=expected,
        tolerance=(2.0e-5, 2.0e-5),
        target_name=target_name,
        kernel_name="nested ragged compact versus identity mapping",
    )
    original_call = prepare_kernel_call(
        original_artifact, original_arguments, original_output
    )
    variant_call = prepare_kernel_call(
        variant_artifact, variant_arguments, variant_output
    )
    report_variant_pipeline(
        variant_artifacts=(variant_artifact,),
        original_launch=original_call,
        variant_launch=variant_call,
        original_prepare=None,
        variant_prepare=None,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="nested ragged compact versus identity mapping",
        performance_scope="kernel-only",
    )


def _run_nested_ragged_split(
    compiler: str, target: Target, target_name: str
) -> None:
    (
        document_lengths, sentence_lengths, document_offsets, sentence_offsets, values
    ) = _nested_inputs()
    expected = _nested_reference(document_lengths, sentence_lengths, values)
    original_artifact = intent.compile(
        nested_jagged_mean_pool, target=target, compiler=compiler
    )
    sentence_artifact = intent.compile(
        nested_sentence_pool, target=target, compiler=compiler
    )
    document_artifact = intent.compile(
        nested_document_pool, target=target, compiler=compiler
    )
    original_arguments = (document_offsets, sentence_offsets, values)
    sentence_arguments = (sentence_offsets, values)
    original_output = original_artifact.run(*original_arguments)
    sentence_output = sentence_artifact.run(*sentence_arguments)
    sentence_sums, sentence_means = outputs(sentence_output)
    document_output = document_artifact.run(
        document_offsets, sentence_offsets, sentence_sums
    )
    variant_output = (sentence_means, document_output)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_output,
        variant=variant_output,
        reference=expected,
        tolerance=(2.0e-5, 2.0e-5),
        target_name=target_name,
        kernel_name="nested ragged fused versus two-stage pooling",
    )
    original_call = prepare_kernel_call(
        original_artifact, original_arguments, original_output
    )
    sentence_call = prepare_kernel_call(
        sentence_artifact, sentence_arguments, sentence_output
    )
    document_call = prepare_kernel_call(
        document_artifact,
        (document_offsets, sentence_offsets, sentence_sums),
        document_output,
    )

    def variant_launch() -> None:
        sentence_call()
        document_call()

    report_variant_pipeline(
        variant_artifacts=(sentence_artifact, document_artifact),
        original_launch=original_call,
        variant_launch=variant_launch,
        original_prepare=None,
        variant_prepare=None,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="nested ragged fused versus two-stage pooling",
        performance_scope="end-to-end GPU pipeline",
    )


def _run_cholesky_right_looking(
    compiler: str, target: Target, target_name: str
) -> None:
    seed = torch.randn(
        (CHOLESKY_BATCH, CHOLESKY_SIZE, CHOLESKY_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    initial = seed @ seed.transpose(1, 2)
    initial.add_(
        torch.eye(CHOLESKY_SIZE, device="cuda")[None], alpha=CHOLESKY_SIZE
    )
    expected = torch.linalg.cholesky(initial)
    original_matrix = initial.clone()
    variant_matrix = initial.clone()
    original_artifact = intent.compile(
        batched_cholesky_lower, target=target, compiler=compiler
    )
    variant_artifact = intent.compile(
        batched_cholesky_right_looking, target=target, compiler=compiler
    )
    original_artifact.run(original_matrix)
    variant_artifact.run(variant_matrix)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_matrix,
        variant=variant_matrix,
        reference=expected,
        tolerance=3.0e-4,
        target_name=target_name,
        kernel_name="Cholesky left-looking versus right-looking",
    )
    original_call = prepare_kernel_call(
        original_artifact, (original_matrix,), ()
    )
    variant_call = prepare_kernel_call(
        variant_artifact, (variant_matrix,), ()
    )
    report_variant_pipeline(
        variant_artifacts=(variant_artifact,),
        original_launch=original_call,
        variant_launch=variant_call,
        original_prepare=lambda: original_matrix.copy_(initial),
        variant_prepare=lambda: variant_matrix.copy_(initial),
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="Cholesky left-looking versus right-looking",
        performance_scope="kernel-only",
    )


def _run_adafactor_scalar_product(
    compiler: str, target: Target, target_name: str
) -> None:
    gradient = torch.randn(
        (ADAFACTOR_ROWS, ADAFACTOR_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    row_state = torch.rand((ADAFACTOR_ROWS,), device="cuda") + 0.01
    column_state = torch.rand((ADAFACTOR_COLUMNS,), device="cuda") + 0.01
    row_mean = row_state.mean().reshape(1)
    initial_parameter = torch.randn_like(gradient)
    original_parameter = initial_parameter.clone()
    variant_parameter = initial_parameter.clone()
    learning_rate, epsilon = 1.0e-2, 1.0e-8
    variance = row_state[:, None] * column_state[None, :] / row_mean
    expected = initial_parameter - learning_rate * gradient * torch.rsqrt(
        variance + epsilon
    )
    original_artifact = intent.compile(adafactor_apply, target=target, compiler=compiler)
    variant_artifact = intent.compile(
        adafactor_apply_scalar_product, target=target, compiler=compiler
    )
    original_arguments = (
        gradient, row_state, column_state, row_mean, original_parameter,
        learning_rate, epsilon,
    )
    variant_arguments = (
        gradient, row_state, column_state, row_mean, variant_parameter,
        learning_rate, epsilon,
    )
    original_artifact.run(*original_arguments)
    variant_artifact.run(*variant_arguments)
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_parameter,
        variant=variant_parameter,
        reference=expected,
        tolerance=2.0e-6,
        target_name=target_name,
        kernel_name="Adafactor row-region versus scalar product",
    )
    original_call = prepare_kernel_call(original_artifact, original_arguments, ())
    variant_call = prepare_kernel_call(variant_artifact, variant_arguments, ())
    report_variant_pipeline(
        variant_artifacts=(variant_artifact,),
        original_launch=original_call,
        variant_launch=variant_call,
        original_prepare=lambda: original_parameter.copy_(initial_parameter),
        variant_prepare=lambda: variant_parameter.copy_(initial_parameter),
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name="Adafactor row-region versus scalar product",
        performance_scope="kernel-only",
    )


def _run_transpose_product_domain(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (TRANSPOSE_ROWS, TRANSPOSE_COLUMNS),
        device="cuda",
        dtype=torch.float16,
    )
    run_variant(
        original=matrix_transpose_scalar_domains,
        variant=matrix_transpose_product_domains,
        arguments=(x,),
        reference=lambda: x.transpose(0, 1).contiguous(),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="transpose nested versus product scalar domains",
        tolerance=0.0,
    )


def _run_ordered_prefix_nested(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (PREFIX_BATCH, PREFIX_ROWS, PREFIX_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    run_variant(
        original=ordered_product_prefix,
        variant=ordered_prefix_nested,
        arguments=(x,),
        reference=lambda: x.flatten(1).cumsum(dim=1).reshape_as(x),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="ordered product versus nested traversal",
        tolerance=2.0e-5,
        cuda_graph=False,
    )


def _run_attention_full_causal_stream(
    compiler: str, target: Target, target_name: str
) -> None:
    shape = (2, 8, 1024, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    run_variant(
        original=flash_attention_fwd,
        variant=flash_attention_full_causal_stream_fwd,
        arguments=(q, k, v, ATTENTION_SCALE),
        reference=lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=ATTENTION_SCALE
        ),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="causal attention logical stop versus full masked stream",
        tolerance=2.0e-2,
        constexprs={"CAUSAL": True},
    )


DECOMPOSITION_RUNNERS: dict[str, Runner] = {
    "variant_moe_product_domain": _run_moe_product_domain,
    "variant_adamw_split_pipeline": _run_adamw_split_pipeline,
    "variant_reshape_cache_split": _run_reshape_cache_split,
    "variant_nested_ragged_identity": _run_nested_ragged_identity,
    "variant_nested_ragged_split": _run_nested_ragged_split,
    "variant_cholesky_right_looking": _run_cholesky_right_looking,
    "variant_adafactor_scalar_product": _run_adafactor_scalar_product,
    "variant_transpose_product_domain": _run_transpose_product_domain,
    "variant_ordered_prefix_nested": _run_ordered_prefix_nested,
    "variant_attention_full_causal_stream": _run_attention_full_causal_stream,
}
