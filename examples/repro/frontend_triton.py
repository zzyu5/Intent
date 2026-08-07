from __future__ import annotations

import argparse
import subprocess

import torch

import intent
import intent.language as I
from intent.ir import OpCode
from intent.ir import walk_operations


POINTS = 1_000_003
CONTIGUOUS_NOALIAS = I.constraints(layout="contiguous", noalias=True)


@intent.fn
def add_values(lhs, rhs):
    return lhs + rhs


@intent.kernel
def vector_add(
    lhs: I.In[I.f32, (POINTS,), CONTIGUOUS_NOALIAS],
    rhs: I.In[I.f32, (POINTS,), CONTIGUOUS_NOALIAS],
    output: I.Out[I.f32, (POINTS,), CONTIGUOUS_NOALIAS],
):
    for index in range(POINTS):
        output[index] = lhs[index] + rhs[index]


@intent.kernel
def tensor_showcase(
    x: I.In[I.f32, (8, 4)],
    output: I.Out[I.f32, (4, 8)],
    flag: I.bool,
):
    reshaped = I.reshape(x, (4, 8))
    transposed = I.transpose(x)
    zero = I.zeros((4, 8), dtype=I.f32)
    one = I.full((4, 8), 1.0, dtype=I.f32)
    paired = add_values(reshaped, transposed)
    exponential = I.exp(paired) + I.exp2(paired)
    logged = I.log(I.maximum(exponential, 1.0))
    bounded = I.minimum(logged, one)
    inverse = I.rsqrt(I.maximum(bounded, 1.0))
    negative = -inverse
    valid = negative < 0.0
    inverted = not flag
    choose_negative = (flag and inverted) or flag
    selected = negative if choose_negative else zero
    masked = I.mask(selected, valid=valid, fill=0.0)
    scanned = I.scan(
        masked,
        axis=1,
        identity=0.0,
        combine=I.add,
        inclusive=True,
    )
    total = I.reduce(
        scanned,
        axis=1,
        identity=0.0,
        combine=add_values,
    )
    maximum = I.reduce.max(scanned, axis=1, identity=-I.inf)
    state = I.record(values=scanned, total=total, maximum=maximum)
    combined = state.values + state.total[:, None] - state.maximum[:, None]
    output[:, :] = I.cast(combined, I.f32)


@intent.kernel
def control_showcase(
    x: I.In[I.f32, (8, 8)],
    output: I.InOut[I.f32, (8, 8)],
    limit: I.i64,
):
    rows = I.domain(0, 8)
    columns = I.domain(0, 8, 1)
    for row, column in I.parallel((rows, columns)):
        value = x[row, column]
        if value >= 0.0:
            normalized = value
        else:
            normalized = -value
        output[row, column] = normalized

    state = I.cast(0.0, I.f32)
    for row in I.ordered(rows):
        value = x[row, 0]
        if value < -10.0:
            continue
        state = state + value
        if state > 10.0:
            break
    output[0, 0] = state

    counter = 0
    while counter < limit:
        counter = counter + 1
        if counter % 2 == 0:
            continue
        if counter > 8:
            break
    quotient = counter // 2
    powered = counter**2
    output[1, 0] = I.cast(quotient + powered, I.f32)


@intent.kernel
def stream_showcase(
    x: I.In[I.f32, (64,)],
    output: I.Out[I.f32, (1,)],
    initial: I.f32,
):
    axis = I.domain(0, 64)
    stream = I.state_stream(axis, extent=8, init=initial)
    with stream:
        for segment, state in stream:
            values = x[segment]
            next_state = state + I.reduce.sum(values, axis=0, identity=0.0)
            stream.yield_(next_state)
    output[0] = stream.result


@intent.kernel
def memory_showcase(
    source: I.In[I.f32, (128,)],
    destination: I.InOut[I.f32, (128,)],
    seed: I.i64,
):
    axis = I.domain(0, 128)
    for region in I.parallel(I.partition(axis, extent=32)):
        indices = I.indices(region)
        values = I.gather(source, index=indices)
        I.scatter_unique(destination, index=indices, value=values)
        I.scatter_reduce(
            destination,
            index=indices,
            value=values,
            combine=I.add,
        )

    work = I.buffer(shape=(128,), dtype=I.f32, init=0.0)
    for index in range(128):
        sample = I.random(seed, index=index, dtype=I.f32)
        I.store(work, index=index, value=sample)
        loaded = I.mutable_load(work, index=index)
        I.atomic_add(destination, index=index, value=loaded)
        previous = I.atomic_cas(
            destination,
            index=index,
            compare=loaded,
            value=loaded,
        )
        I.store(destination, index=index, value=previous)
    I.fence()


class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    ACTIVATION: I.Constexpr[Activation],
):
    M, K = a.shape
    _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            accumulator = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            if ACTIVATION == Activation.RELU:
                accumulator = I.maximum(accumulator, 0.0)
            c[mr, nr] = I.cast(accumulator, I.f16)


@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        maximum = I.reduce.max(values, axis=0, identity=-I.inf)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0, identity=0.0)
        y[row, columns] = numerator / denominator


@intent.kernel
def flash_attention_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, D = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for q_region in I.parallel(
                I.partition(q_axis, extent=I.auto("Q_TILE"))
            ):
                q_block = q[batch, head, q_region, :]
                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((q_region,), -I.inf, dtype=I.f32),
                        I.zeros((q_region,), dtype=I.f32),
                        I.zeros((q_region, DV), dtype=I.f32),
                    ),
                )
                with stream:
                    for k_region, (maximum, denominator, accumulator) in stream:
                        k_block = k[batch, head, k_region, :]
                        v_block = v[batch, head, k_region, :]
                        scores = I.contract(
                            q_block,
                            k_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        scores = scores * (scale * I.LOG2E)
                        if CAUSAL:
                            q_index = I.indices(q_region)
                            k_index = I.indices(k_region)
                            valid = q_index[:, None] >= k_index[None, :]
                            if not I.any(valid):
                                stream.yield_(maximum, denominator, accumulator)
                                continue
                            scores = I.mask(scores, valid=valid, fill=-I.inf)
                        local_maximum = I.reduce.max(
                            scores, axis=1, identity=-I.inf
                        )
                        next_maximum = I.maximum(maximum, local_maximum)
                        alpha = I.exp2(maximum - next_maximum)
                        probability = I.exp2(scores - next_maximum[:, None])
                        next_denominator = alpha * denominator + I.reduce.sum(
                            probability, axis=1, identity=0.0
                        )
                        low_probability = I.cast(probability, I.f16)
                        next_accumulator = alpha[:, None] * accumulator + I.contract(
                            low_probability,
                            v_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                _, denominator, accumulator = stream.result
                output[batch, head, q_region, :] = I.cast(
                    accumulator / denominator[:, None], I.f16
                )


@intent.kernel
def reduction_pass1(
    x: I.In[I.f32, ("M", "N")],
    partial: I.Out[I.f32, ("M", "P")],
):
    M, N = x.shape
    P = partial.shape[1]
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        for part, region in I.parallel(I.partition(columns, count=P)):
            partial[row, part] = I.reduce.max(
                x[row, region], axis=0, identity=-I.inf
            )


@intent.kernel
def reduction_pass2(
    partial: I.In[I.f32, ("M", "P")],
    output: I.Out[I.f32, ("M",)],
):
    M, P = partial.shape
    for row in I.parallel(I.domain(0, M)):
        output[row] = I.reduce.max(
            partial[row, :], axis=0, identity=-I.inf
        )


@intent.kernel
def moe_expert_ffn(
    x: I.In[I.f16, ("T", "D")],
    route_offsets: I.In[I.i32, ("E_PLUS_1",)],
    member_routes: I.In[I.i32, ("R",)],
    route_token: I.In[I.i32, ("NR",)],
    route_weights: I.In[I.f32, ("NR",)],
    w1: I.In[I.f16, ("E", "D", "F")],
    w2: I.In[I.f16, ("E", "F", "D")],
    y: I.InOut[I.f32, ("T", "D")],
):
    T, D = x.shape
    E, _, F = w1.shape
    groups = I.ragged(
        outer=I.domain(0, E),
        offsets=route_offsets,
        indices=member_routes,
    )
    for expert in I.parallel(groups.outer):
        for route_region in I.parallel(
            I.partition(groups[expert], extent=I.auto("ROUTE_TILE"))
        ):
            routes = I.members(route_region)
            token = I.gather(route_token, index=routes)
            weight = I.gather(route_weights, index=routes)
            values = I.gather(x, index=(token, slice(None)))
            hidden = I.contract(
                values,
                w1[expert, :, :],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            hidden = I.maximum(hidden, 0.0)
            route_output = I.contract(
                hidden,
                w2[expert, :, :],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            I.scatter_reduce(
                y,
                index=(token, slice(None)),
                value=weight[:, None] * route_output,
                combine=I.add,
            )


def _lower_all(intent_opt: str) -> set[OpCode]:
    definitions = (
        (vector_add, None),
        (tensor_showcase, None),
        (control_showcase, None),
        (stream_showcase, None),
        (memory_showcase, None),
        (gemm, {"ACTIVATION": Activation.RELU}),
        (stable_softmax, None),
        (flash_attention_fwd, {"CAUSAL": True}),
        (reduction_pass1, None),
        (reduction_pass2, None),
        (moe_expert_ffn, None),
    )
    covered: set[OpCode] = set()
    for definition, constexprs in definitions:
        module = intent.lower_to_kernel_ir(definition, constexprs=constexprs)
        covered.update(
            operation.opcode
            for function in module.functions
            for operation in walk_operations(function.body)
        )
        mlir = intent.emit_mlir(module)
        subprocess.run(
            [intent_opt],
            input=mlir,
            text=True,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        print(f"frontend MLIR: {definition.__name__}: PASS")
    missing = set(OpCode) - covered
    if missing:
        names = ", ".join(sorted(opcode.value for opcode in missing))
        raise RuntimeError(f"frontend opcode coverage is incomplete: {names}")
    print(f"frontend opcode coverage: PASS ({len(covered)} opcodes)")
    return covered


def _run_triton() -> None:
    artifact = intent.compile(vector_add, target=intent.TritonTarget(device=0))
    device = torch.device("cuda", 0)
    lhs = torch.randn(POINTS, device=device, dtype=torch.float32)
    rhs = torch.randn(POINTS, device=device, dtype=torch.float32)
    output = torch.empty_like(lhs)
    artifact(lhs, rhs, output)
    reference = lhs + rhs
    maximum_error = torch.max(torch.abs(output - reference)).item()
    if maximum_error != 0.0:
        raise RuntimeError(f"Triton numerical comparison failed: max error {maximum_error}")
    print("=== Generated Triton source ===")
    print(artifact.source, end="")
    print("backend IR levels: " + ", ".join(sorted(artifact.backend_ir)))
    print(
        f"Triton numerical comparison: PASS ({POINTS} f32 elements, max error {maximum_error})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent-opt",
        default="/tmp/intentdsl-build/tools/intent-opt/intent-opt",
    )
    arguments = parser.parse_args()
    _lower_all(arguments.intent_opt)
    _run_triton()


if __name__ == "__main__":
    main()
