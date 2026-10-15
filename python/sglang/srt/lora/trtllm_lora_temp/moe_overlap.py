"""Two-stream MoE LoRA dispatch (O1).

Monkey-patches ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora`` in
``layers/moe/moe_runner/flashinfer_trtllm.py`` (when
``SGLANG_LORA_TWO_STREAM=1``) so the gate_up LoRA shrink+expand runs on a
side stream concurrent with the main-stream FP8 quant.

Batches that don't qualify for two-stream (prefill / non-virtual-lora /
batch without active LoRA) fall through to the saved-original function so
their behavior is byte-identical to the unpatched code path.
"""

import torch

from sglang.srt.lora.trtllm_lora_temp import (
    get_lora_side_stream,
    get_original_bf16_moe_lora_func,
    get_original_fp4_moe_lora_func,
    get_original_moe_lora_func,
    get_original_mxfp4_moe_lora_func,
    is_two_stream_active,
)

# GEMM1-LoRA overlap: keep LoRA-ready events recorded during cuda-graph capture alive so the
# captured cross-stream wait (resolved inside the trtllm op before activation) isn't torn down
# before graph instantiation. Only appended while capturing; eager runs rely on CUDA's
# deferred cudaEventDestroy.
_LORA_OVERLAP_EVENTS: list = []


def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Drop-in replacement for the like-named function in flashinfer_trtllm.py.

    Two-stream fast path: only fires when the batch is decode-shaped AND uses
    virtual-experts LoRA. Everything else delegates to the original function.
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    import os as _o2, sys as _s2
    if _o2.environ.get("SGLANG_TS_DIAG") == "1" and not getattr(_o2, "_ts_printed_hit", False) and use_virtual_lora_store:
        _o2._ts_printed_hit = True
        print("[TS-ACTIVE] fp4 two-stream ENTERED (tokens=%s max_rank=%s)" % (
            hidden_states.shape[0], lora_info.max_lora_rank), flush=True, file=_s2.stderr)
    # Two-stream requires virtual-experts LoRA AND a decode-shaped batch.
    # Fall back to the original implementation for anything else (prefill,
    # non-virtual LoRA, non-LoRA capture, etc.).
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
        maybe_overlap_staged_shared_add,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils.common import next_power_of_2

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm LoRA currently supports the gated SwiGLU FP8 "
        "Qwen path only."
    )
    assert quant_info.block_quant and not quant_info.use_mxfp8, (
        "experimental_sgl_trtllm LoRA currently supports DeepSeekFp8 block-quant "
        "checkpoints only."
    )
    assert quant_info.weight_block_k is not None
    assert quant_info.w13_weight_scale_inv is not None
    assert quant_info.w2_weight_scale_inv is not None

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    side_stream = get_lora_side_stream()

    # EP-aware LoRA: under MoE EP each rank computes the delta only for its owned experts
    # (passed via local_expert_offset/local_num_experts below). gate_up_delta stays
    # new_empty even though non-owned [token, k] slots are then left unwritten -- the
    # trtllm MoE is itself EP-aware, so those slots never feed the all-reduced output.
    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    gate_up_delta = hidden_states.new_zeros(gate_up_delta_shape)

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # GEMM1-LoRA overlap: fire the gate_up LoRA on the side stream + record an event; the
    # trtllm op waits on it right before activation (the only consumer of gate_up_delta), so
    # permute+GEMM1 overlap the side-stream LoRA shrink/expand instead of joining before the
    # whole op.
    lora_event = torch.cuda.Event()

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph
    # allocator safety -- see the "routing" stage in virtual_experts.py):
    # pre-warm the routing cache and pre-allocate the shrink intermediate here,
    # so the side-stream block below launches kernels only.
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_zeros(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )

    # O1 fork — gate_up shrink/expand on side stream concurrent with the main-stream
    # per-token-group FP8 quant + the trtllm op's permute+GEMM1 below.
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    # Fuse the per-token scale transpose into the quant kernel: column-major scales make
    # the `.t()` a free view, dropping the standalone ~2us transpose+copy. The trtllm MoE
    # kernel wants the [K, M]-contiguous scale, which `.t()` of the column-major buffer is
    # exactly -- byte/shape-identical to the old `a_sf.t().contiguous()`.
    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, quant_info.intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # SGLANG_OPT_LORA_FUSED_TOPK_PACK: the routed pack may already have been produced
    # fused inside the gating kernel (StandardTopKOutput.packed_topk_ids) — including
    # the padded-region id=-1 mask. Fall back to the separate pack otherwise.
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # No pre-op join: the trtllm op waits on lora_event right before its activation kernel,
    # so permute+GEMM1 run concurrent with the side-stream LoRA. Keep the event alive through
    # cuda-graph capture so the captured cross-stream wait isn't torn down before instantiation.
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    # Down-LoRA/finalize overlap (env-gated): the op records gemm2_done_event right after the
    # base down GEMM (before finalize); the side stream waits on it and runs ONLY the down-proj
    # LoRA shrink (gemm A) + routing prep concurrent with finalizeKernel. The expand-add
    # (gemm B) atomic-adds into `output` -- which finalize WRITES concurrently -- so it stays
    # on the main stream after the op (post-finalize), exactly like the serial path.
    # DISABLED: the down/finalize overlap is bench-verified net-neutral-to-negative AND
    # corrupts the base/decode path — the captured gemm2_done cross-stream event under
    # cuda-graph replay perturbs no-active-LoRA (base) requests (qwen base gsm8k 0.81 -> 0.56
    # with it on; bisect-confirmed). The serial down-LoRA path below is used unconditionally.
    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()
        # Materialize the underlying cudaEvent (torch creates it lazily on first record) so
        # .cuda_event is a real handle; the op re-records it after GEMM2.
        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    moe_result = trtllm_fp8_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=a_q,
        hidden_states_scale=a_sf_t,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale_inv,
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale_inv,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=quant_info.intermediate_size,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=(
            RoutingMethodType.TopK
            if quant_info.routing_method_type == RoutingMethodType.DeepSeekV3
            else quant_info.routing_method_type
        ),
        use_shuffled_weight=False,
        do_finalize=True,
        output=direct_down_output,
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
        gemm2_done_event=gemm2_done_handle,
    )

    output = moe_result

    def _run_down_lora(
        out, stage="all", intermediate_buffer=None, expand_wait_event=None
    ):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, quant_info.intermediate_size),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
            expand_wait_event=expand_wait_event,
        )

    # Shared-add overlap: the trtllm op above already finalized `output`, so the
    # staged shared-expert add (if any) can run on the producer (main) stream
    # concurrent with the down-LoRA shrink below; the expand waits on it via
    # expand_wait_event before atomic-adding into the same buffer.
    shared_add_done = maybe_overlap_staged_shared_add(output)

    if down_overlap:
        # Fork at "base down GEMM done": ONLY the shrink (gemm A) + routing prep run on the
        # side stream, concurrent with the main-stream finalizeKernel. The expand-add (gemm B)
        # joins back on the MAIN stream after the op -- i.e. strictly after finalize wrote
        # `output` -- and atomic-adds into it exactly like the serial path: same kernels,
        # same buffers, identical numerics; the shrink just starts earlier.
        # The shrink intermediate is allocated HERE (main = consumer stream of the expand),
        # per the SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC lesson on side-stream allocations.
        down_intermediate = hidden_states.new_zeros(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(
            output,
            stage="expand",
            intermediate_buffer=down_intermediate,
            expand_wait_event=shared_add_done,
        )
    else:
        _run_down_lora(output, expand_wait_event=shared_add_done)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_fp4_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Two-stream NVFP4 sibling of the FP8 two-stream MoE LoRA dispatch.

    Fires only for virtual-experts LoRA + decode-shaped batches; everything else
    delegates to the saved-original single-stream FP4 dispatch (byte-identical).
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        import os as _o, sys as _s
        if _o.environ.get("SGLANG_TS_DIAG") == "1" and not getattr(_o, "_ts_printed_fallback", False):
            _o._ts_printed_fallback = True
            print("[TS-ACTIVE] fp4 two-stream FALLBACK (virtual=%s active=%s tokens=%s)" % (
                use_virtual_lora_store, is_two_stream_active(hidden_states), hidden_states.shape[0]), flush=True, file=_s.stderr)
        return get_original_fp4_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.runtime_context import get_parallel

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm NVFP4 LoRA currently supports the gated SwiGLU path only."
    )
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    # Down-proj LoRA runs serially on the main stream (after the trtllm op) by default. The old
    # side-stream down-overlap was removed: bench-verified net-neutral-to-negative (the extra
    # side-stream all-reduce cancels any overlap gain), AND its act_ready_event cross-stream sync
    # corrupted decode state under sustained heavy LoRA load (cuda-graph replay -> persistent
    # garbage). SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1 re-introduces a more conservative variant:
    # fork at gemm2_done (not act_ready), and only the SHRINK (gemm A) overlaps the finalize
    # kernel -- the expand-add (gemm B) stays on the MAIN stream post-finalize (it writes the
    # same `output` finalize writes), so its kernels/numerics match the serial path exactly.
    inter = quant_info.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_zeros(
        (hidden_states.shape[0], runner_config.top_k, quant_info.w13_weight.shape[1])
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph
    # allocator safety -- see the "routing" stage in virtual_experts.py).
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_zeros(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )

    # O1-fp4 fork: gate_up shrink/expand on the side stream, concurrent with the
    # FP4 op's permute + gate_up GEMM1 below. The op waits on lora_event right
    # before its activation kernel (the only consumer of gate_up_delta).
    lora_event = torch.cuda.Event()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = fused_pack_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # Keep the event alive through cuda-graph capture so the captured wait inside
    # the FP4 op isn't torn down before instantiation (eager relies on deferred destroy).
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    # Down-LoRA/finalize overlap (env-gated, see the comment above): record gemm2_done inside
    # the op; the side stream runs only the down-LoRA shrink + routing prep concurrent with
    # finalize; the expand-add joins back on the main stream after the op.
    # DISABLED: the down/finalize overlap is bench-verified net-neutral-to-negative AND
    # corrupts the base/decode path — the captured gemm2_done cross-stream event under
    # cuda-graph replay perturbs no-active-LoRA (base) requests (qwen base gsm8k 0.81 -> 0.56
    # with it on; bisect-confirmed). The serial down-LoRA path below is used unconditionally.
    down_overlap = False
    gemm2_done_handle = 0
    if down_overlap:
        gemm2_done_event = torch.cuda.Event()
        # Materialize the underlying cudaEvent (torch creates it lazily on first record) so
        # .cuda_event is a real handle; the op re-records it after the down GEMM.
        gemm2_done_event.record()
        gemm2_done_handle = gemm2_done_event.cuda_event

    output = trtllm_fp4_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
        output1_scales_scalar=quant_info.g1_scale_c,
        output1_scales_gate_scalar=quant_info.g1_alphas,
        output2_scales_scalar=quant_info.g2_alphas,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        lora_ready_event=lora_ready_handle,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=quant_info.routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        gemm2_done_event=gemm2_done_handle,
    )

    def _run_down_lora(out, stage="all", intermediate_buffer=None):
        return merged_experts_fused_moe_lora_add(
            output=out,
            hidden_states=activation_lora_input.view(-1, inter),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=False,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            stage=stage,
            intermediate_buffer=intermediate_buffer,
        )

    if down_overlap:
        # Fork at "base down GEMM done": shrink (gemm A) + routing prep on the side stream
        # concurrent with the main-stream finalize; the expand-add (gemm B) joins back on the
        # MAIN stream after the op (strictly post-finalize) -- serial-path kernels/numerics.
        # Intermediate allocated on main (= consumer stream of the expand), per the
        # SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC lesson.
        down_intermediate = hidden_states.new_zeros(
            (
                hidden_states.shape[0],
                topk_ids.shape[1],
                lora_info.down_lora_a_weights.shape[2],
            )
        )
        side_stream.wait_event(gemm2_done_event)
        shrink_done_event = torch.cuda.Event()
        with torch.cuda.stream(side_stream):
            _run_down_lora(
                output, stage="shrink", intermediate_buffer=down_intermediate
            )
            shrink_done_event.record()
        if torch.cuda.is_current_stream_capturing():
            _LORA_OVERLAP_EVENTS.append(gemm2_done_event)
            _LORA_OVERLAP_EVENTS.append(shrink_done_event)
        torch.cuda.current_stream().wait_event(shrink_done_event)
        _run_down_lora(output, stage="expand", intermediate_buffer=down_intermediate)
    else:
        _run_down_lora(output)
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_bf16_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """Two-stream BF16 sibling of the FP8/FP4 two-stream MoE LoRA dispatches.

    O1-bf16 fork: the gate_up LoRA shrink/expand runs on the side stream
    concurrent with the bf16 op's routing + permute + gate_up GEMM; the op
    waits on ``lora_ready_event`` right before its activation kernel (the only
    consumer of ``gate_up_delta``). Fires only for virtual-experts LoRA +
    decode-shaped batches; everything else delegates to the saved-original
    single-stream bf16 dispatch (byte-identical). Down-LoRA stays serial on
    the main stream — the down/finalize overlap was bench-verified
    net-neutral-to-negative on the FP8/FP4 paths and corrupted the base
    decode path under cuda-graph replay (see the comment in the FP4 variant).
    """
    hidden_states = dispatch_output.hidden_states

    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if not (use_virtual_lora_store and is_two_stream_active(hidden_states)):
        return get_original_bf16_moe_lora_func()(
            dispatch_output, quant_info, runner_config, lora_info
        )

    # ---- two-stream fast path ----
    from sglang.kernels.ops.moe.trtllm_lora_temp import trtllm_bf16_routed_moe_lora
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.runtime_context import get_parallel

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm BF16 LoRA currently supports the gated SwiGLU path only."
    )
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = runner_config.intermediate_size_per_partition
    side_stream = get_lora_side_stream()

    gate_up_delta = hidden_states.new_zeros(
        (hidden_states.shape[0], runner_config.top_k, 2 * inter)
    )

    def _run_gate_up_lora():
        merged_experts_fused_moe_lora_add(
            output=gate_up_delta,
            hidden_states=hidden_states,
            lora_a=lora_info.gate_up_lora_a_weights,
            lora_b=lora_info.gate_up_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=False,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=False,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=runner_config.num_local_experts,
            intermediate_buffer=gate_up_lora_intermediate,
        )

    # O1-bf16 fork: gate_up shrink/expand on the side stream, concurrent with the
    # bf16 op's routing + permute + gate_up GEMM below. The op waits on lora_event
    # right before its activation kernel (the only consumer of gate_up_delta).
    lora_event = torch.cuda.Event()

    # Hoist every side-chain allocation onto the MAIN stream (cuda-graph allocator
    # safety -- see the "routing" stage in virtual_experts.py): pre-warm the routing
    # cache and pre-allocate the shrink intermediate here, so the side-stream block
    # below launches kernels only. Without this, the routing tensors + shrink
    # intermediate get allocated inside the side-stream context during cuda-graph
    # capture, where cross-stream tracking is off -> pool blocks reused with no graph
    # edge -> '!!!!' decode corruption at max-loras>=2 (matches the fp8/fp4 fix).
    merged_experts_fused_moe_lora_add(
        output=gate_up_delta,
        hidden_states=hidden_states,
        lora_a=lora_info.gate_up_lora_a_weights,
        lora_b=lora_info.gate_up_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=False,
        experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
        experts_shared_outer_loras_b=False,
        routing_cache=fused_lora_routing_cache,
        stage="routing",
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    gate_up_lora_intermediate = hidden_states.new_zeros(
        (
            hidden_states.shape[0],
            topk_ids.shape[1],
            lora_info.gate_up_lora_a_weights.shape[2],
        )
    )
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _run_gate_up_lora()
        lora_event.record()

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    packed_topk_ids = getattr(topk_output, "packed_topk_ids", None)
    if packed_topk_ids is None:
        packed_topk_ids = fused_pack_topk(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    routing_method_type = runner_config.routing_method_type
    if routing_method_type is None:
        routing_method_type = RoutingMethodType.Default
    elif routing_method_type == RoutingMethodType.DeepSeekV3:
        routing_method_type = RoutingMethodType.TopK

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    # Keep the event alive through cuda-graph capture so the captured wait inside
    # the bf16 op isn't torn down before instantiation (eager relies on deferred destroy).
    if torch.cuda.is_current_stream_capturing():
        _LORA_OVERLAP_EVENTS.append(lora_event)
    lora_ready_handle = lora_event.cuda_event

    output = trtllm_bf16_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=quant_info.gemm1_weights,
        gemm2_weights=quant_info.gemm2_weights,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
        num_experts=quant_info.global_num_experts,
        top_k=runner_config.top_k,
        intermediate_size=inter,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
        routing_method_type=routing_method_type,
        do_finalize=True,
        output=direct_down_output,
        activation_type=get_activation_type(
            runner_config.activation, is_gated=runner_config.is_gated
        ),
        lora_ready_event=lora_ready_handle,
        # Down-LoRA/finalize overlap intentionally NOT wired (gemm2_done_event=0):
        # bench-verified net-neutral-to-negative on FP8/FP4 and corrupts the base
        # decode path under cuda-graph replay. Serial down-LoRA below.
        gemm2_done_event=0,
    )

    merged_experts_fused_moe_lora_add(
        output=output,
        hidden_states=activation_lora_input.view(-1, inter),
        lora_a=lora_info.down_lora_a_weights,
        lora_b=lora_info.down_lora_b_weights,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
        mul_routed_weight=True,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=True,
        use_direct_expand_add=lora_info.max_lora_rank <= 64,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=runner_config.num_local_experts,
    )
    return StandardCombineInput(hidden_states=output)


# M2-v2 rebase (step3): known-good single-stream mxfp4 body + incremental side-stream delta.
_MB_ENTERED = [False]  # M2-v2 tracer state
_POSTDELTA_TEST_A = None  # R66 test-mode parallel adapter A' (fixed random, seed=42)
_POSTDELTA_TEST_B = None  # R66 test-mode parallel adapter B' (fixed random, seed=42)
import os as _o


def fused_experts_none_to_experimental_sgl_trtllm_mxfp4_lora_two_stream(
    dispatch_output,
    quant_info,
    runner_config,
    lora_info,
):
    """DeepSeek-V4.1 MXFP4 (SM100 trtllm-gen) MoE-LoRA dispatch.

    Sibling of ``fused_experts_none_to_experimental_sgl_trtllm_fp4_lora`` for the
    sglang MXFP4 checkpoint format and its gated SwiGLU with the OAI / GPT-OSS
    activation controls (alpha=1.702, beta=1.0, clamp_limit=7.0 — the same
    parameters the non-LoRA ``_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen``
    path passes to the trtllm-gen kernel).

    Decomposed (unfused-activation) pipeline: routing -> gather -> gate_up grouped
    GEMM (raw 2*inter) -> activation that adds ``gate_up_lora_delta`` pre-SwiGLU
    and captures ``activation_lora_input`` -> MXFP8 quant -> down grouped GEMM ->
    finalize, then the virtual-experts down-LoRA is merged into the output.

    Weight/scale layout (see ``mxfp4.py``): ``w13_weight``/``w2_weight`` are packed
    e2m1, ``w13_weight_scale`` is e4m3x2, ``w2_weight_scale`` is ue8m0, and both
    biases are fp32 per-expert. The kernel consumes them as-is.
    """
    # M2-v2: replay of lora_dispatch.py module-level imports
    from typing import TYPE_CHECKING
    import torch
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.distributed.device_communicators.pynccl_allocator import use_symmetric_memory
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils.common import next_power_of_2
    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
        get_activation_type,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    if _o.environ.get("SGLANG_TS_DIAG") == "1" and not _MB_ENTERED[0]:
        _MB_ENTERED[0] = True
        print("[TS-ACTIVE] rebased mxfp4 two-stream ENTERED tokens=%s rank=%s"
              % (dispatch_output.hidden_states.shape[0], lora_info.max_lora_rank),
              flush=True, file=_o.sys.stderr)

    assert runner_config.is_gated, (
        "experimental_sgl_trtllm MXFP4 LoRA requires a gated activation."
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    # GLM->DSV4 MoL perf port: the dual TopK produces BYPASSED-format
    # routing for no-LoRA batches. When BYPASSED arrives here, route to the
    # stock fused MoE entry (the same one a no-MoL deployment uses, which
    # natively consumes the bypassed format and keeps the base batch on the
    # fast fused route). LoRA-active batches always arrive STANDARD.
    if TopKOutputChecker.format_is_bypassed(topk_output):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            fused_experts_none_to_flashinfer_trtllm,
        )

        _combine = fused_experts_none_to_flashinfer_trtllm(
            dispatch_output, quant_info, runner_config
        )
        # POSTDELTA: add delta OUTSIDE graph (BYPASSED dispatch → graph == base
        # → no STANDARD topk materialization in graph → no degradation under
        # 256-bench). Delta computed eagerly after graph replay.
        if _o.environ.get("SGLANG_LORA_FUSED_POSTDELTA") == "1":
            from sglang.srt.model_executor.runner_utils.capture_mode import (
                get_is_capture_mode,
            )
            if not get_is_capture_mode():
                _y_base = _combine.hidden_states
                _x = dispatch_output.hidden_states
                # Test mode: fixed random B'A'x (scale 0.01, seed=42)
                if _o.environ.get("SGLANG_LORA_POSTDELTA_TEST") == "1":
                    _d = _x.shape[-1]
                    _g = torch.Generator(device=_x.device).manual_seed(42)
                    _ta = torch.randn(16, _d, device=_x.device, dtype=_x.dtype, generator=_g) * 0.01
                    _tb = torch.randn(_d, 16, device=_x.device, dtype=_x.dtype, generator=_g) * 0.01
                    _delta = torch.nn.functional.linear(_x, _ta)
                    _delta = torch.nn.functional.linear(_delta, _tb)
                    _y_base = _y_base + _delta
                # Parallel adapter delta: B'A'x (real weights, when available)
                _a = getattr(lora_info, "down_lora_a_weights", None)
                if _a is not None:
                    if _a.ndim == 3:
                        _a = _a[0]
                    if _a.shape[-1] == _x.shape[-1]:
                        _b = getattr(lora_info, "down_lora_b_weights", None)
                        if _b is not None:
                            if _b.ndim == 3:
                                _b = _b[0]
                            _delta = torch.nn.functional.linear(_x, _a)
                            _delta = torch.nn.functional.linear(_delta, _b)
                            _y_base = _y_base + _delta
                _combine = StandardCombineInput(hidden_states=_y_base)
        return _combine
    assert TopKOutputChecker.format_is_standard(topk_output), (
        "experimental_sgl_trtllm MXFP4 LoRA needs materialized topk_ids; the "
        "logits-based (bypassed) routing does not expose them."
    )
    assert runner_config.top_k is not None

    # REAL FIX — routed scaling on the decomposed (packed-topk) kernel path.
    # The vendored trtllm_fused_moe_runner.cu sets routingData.mRouteScale only
    # in the DeepSeekV3 (logits) routing branches; the TopK-only branch used
    # by every packed-topk call with pre-materialized ids never assigns it, so
    # routed_scaling_factor is silently ignored on this path. A/B on the
    # GLM-5.3 fp8 build of the same kernel family measured the decomposed
    # output at exactly 1/routed_scaling_factor of stock (0.400x for factor
    # 2.5); for DeepSeek-V4.1-Flash (factor 1.5) every MoE layer would
    # contribute 0.667x of stock, compounding per layer/chunk over long
    # prefills. Applying the factor on the materialized weights here is
    # mathematically identical (sum((s*w_i)*y_i) == s*sum(w_i*y_i)) and covers
    # the pass-through, the decode-graph capture and the adapter/VE path (the
    # hooks consume the same weights, so base expert outputs and LoRA deltas
    # stay scaled consistently).
    if (
        runner_config.routed_scaling_factor is not None
        and runner_config.routed_scaling_factor != 1.0
    ):
        topk_output = topk_output._replace(
            topk_weights=(
                topk_output.topk_weights
                * runner_config.routed_scaling_factor
            )
        )
        # The pass-through call below receives `dispatch_output`, so the
        # scaled routing must be written back into it as well (NamedTuples
        # are immutable; a local rebind alone would leave the pass-through
        # consuming the unscaled weights).
        dispatch_output = dispatch_output._replace(topk_output=topk_output)

    if _o.environ.get("SGLANG_LORA_FUSED_POSTDELTA") == "1":
        # R50 fused-kernel + output-side delta. The routed (STANDARD topk)
        # cubin MMU-faults under 256-bench graph replay (R56 Xid 13/31).
        # FIX: route through the BYPASSED logits cubin
        # (trtllm_fp4_block_scale_moe) — verified safe under 256-bench graph
        # replay (base 256/256 pass). Construct a BypassedTopKOutput from
        # the StandardTopKOutput's preserved router_logits so _fused_experts
        # takes the BYPASSED Situ branch (same path base uses — both
        # prefill AND decode). For zero-weight / pass-through (delta=0)
        # this is bitwise-identical to base (0+0=0). If router_logits is
        # None, fall through to the decomposed path below.
        # R59: removed the decode-only gate — prefill on the decomposed path
        # accumulates resource degradation across consecutive 256-bench
        # runs (R58b detokenizer hang after 3rd 256 bench, no Xid). Routing
        # prefill through the same BYPASSED logits cubin eliminates the
        # decomposed-path accumulation (prefill == base path, verified safe).
        _topk_output = dispatch_output.topk_output
        _rl = getattr(_topk_output, "router_logits", None)
        if _rl is not None:
            from sglang.srt.layers.moe.topk import (
                BypassedTopKOutput,
                TopKConfig,
            )
            # Prefer the layer's full topk_config (contains n_group/
            # topk_group/renormalize/routed_scaling_factor — matching
            # base exactly). Fall back to minimal TopKConfig if layer
            # is unavailable. R59 crash: the minimal TopKConfig may
            # miss renormalize or other fields, causing detokenizer
            # hang under prefill; using the layer's config eliminates
            # the mismatch with the base path.
            _layer = runner_config.layer
            _tc = (
                getattr(_layer, "topk_config", None)
                if _layer is not None
                else None
            )
            if _tc is None:
                _tc = TopKConfig(
                    top_k=runner_config.top_k,
                    routed_scaling_factor=(
                        runner_config.routed_scaling_factor
                        if runner_config.routed_scaling_factor is not None
                        else 1.0
                    ),
                )
            _bypassed = BypassedTopKOutput(
                hidden_states=dispatch_output.hidden_states,
                router_logits=_rl,
                topk_config=_tc,
            )
            _combine = _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
                dispatch_output._replace(topk_output=_bypassed),
                quant_info,
                runner_config,
            )
            _y_base = _combine.hidden_states

            # Delta computation: OUTSIDE graph for replay (capture skips delta
            # → graph only contains y_base, same as base graph → no degradation
            # under 256-bench). R72 nixl also degrades → root cause is graph-
            # internal delta, not transport backend. Moving delta outside
            # graph eliminates the degradation (graph == base graph).
            from sglang.srt.model_executor.runner_utils.capture_mode import (
                get_is_capture_mode,
            )
            if not get_is_capture_mode():
                # Replay or eager: add delta (outside graph for replay)
                # Parallel adapter delta: B'A'x (A' eats layer input x).
                _a = getattr(lora_info, "down_lora_a_weights", None)
                if _a is not None:
                    if _a.ndim == 3:
                        _a = _a[0]
                    _x = dispatch_output.hidden_states
                    if _a.shape[-1] == _x.shape[-1]:
                        _b = getattr(lora_info, "down_lora_b_weights", None)
                        if _b is not None:
                            if _b.ndim == 3:
                                _b = _b[0]
                            _delta = torch.nn.functional.linear(_x, _a)
                            _delta = torch.nn.functional.linear(_delta, _b)
                            _y_base = _y_base + _delta

                # Test mode: inject fixed random delta (B'A'x) to verify the
                # delta channel. Seed=42 for determinism, scale 0.01.
                if _o.environ.get("SGLANG_LORA_POSTDELTA_TEST") == "1":
                    _x = dispatch_output.hidden_states
                    _d = _x.shape[-1]
                    _g = torch.Generator(device=_x.device).manual_seed(42)
                    _ta = torch.randn(16, _d, device=_x.device, dtype=_x.dtype, generator=_g) * 0.01
                    _tb = torch.randn(_d, 16, device=_x.device, dtype=_x.dtype, generator=_g) * 0.01
                    _delta = torch.nn.functional.linear(_x, _ta)
                    _delta = torch.nn.functional.linear(_delta, _tb)
                    _y_base = _y_base + _delta
            # capture: delta=0 (graph only contains y_base, same as base graph)

            return StandardCombineInput(hidden_states=_y_base)
        # router_logits is None — fall through to decomposed path below

    # PASS-THROUGH PRECISION (zero-LoRA == base): no-adapter requests take the
    # SAME decomposed pipeline with zero deltas. The previous early-return to
    # the stock fused kernel made "zero adapter == base" structurally
    # impossible — the fused kernel's fp32 GEMM1 epilogue never materializes
    # the gate_up output, while the decomposed Gemm2::Runner writes bf16, so
    # every layer rounded differently and the greedy stream diverged. With
    # both sides on the decomposed path, a zero-weight adapter's VE delta
    # (exactly 0.0) and a pass-through's zero delta are bit-identical into the
    # activation's x + 0.0 -> x.
    has_active_lora = lora_info.has_active_lora
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    assert not has_active_lora or use_virtual_lora_store, (
        "MXFP4 trtllm LoRA with an active adapter requires virtual-experts."
    )
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = quant_info.intermediate_size_per_partition

    # --- Pad/unpad bridge: the VE world and the C++ op world use different
    # intermediate widths on the trtllm-gen MXFP4 path. The flashinfer C++ op
    # consumes the PADDED per-rank width (128-aligned, V4.1: 2304/4=576 -> 640)
    # because its gemm1 output is laid out padded; the VE/LoRA buffers are
    # sized by get_hidden_dim (UNPADDED 576). Derive the VE width from the
    # actual B buffer (the two must always agree) and bridge explicitly:
    #   gate_up: VE writes 2*576=1152 -> zero-pad to 2*640=1280 for the C++
    #            activation kernel (the padded tail has no LoRA weights).
    #   down:    the C++ activation kernel writes 640 (padded) columns of
    #            activation_lora_input; down VE reads the first 576 (the A
    #            buffer's input width).
    _inter = quant_info.intermediate_size_per_partition  # padded (C++ op)
    _inter_ve = (
        lora_info.gate_up_lora_b_weights.shape[2] // 2
    )  # unpadded, from the shared gate_up B buffer (N = 2 * inter_ve)

    gate_up_delta_ve = hidden_states.new_zeros(
        (hidden_states.shape[0], runner_config.top_k, 2 * _inter_ve)
    )
    gate_up_delta = (
        hidden_states.new_zeros(
            (hidden_states.shape[0], runner_config.top_k, 2 * _inter)
        )
        if _inter > _inter_ve
        else gate_up_delta_ve
    )
    lora_event = torch.cuda.Event()
    _mb_side = get_lora_side_stream()
    _mb_side.wait_stream(torch.cuda.current_stream())
    # Static zero-weight detection for the gate_up LoRA (down-only adapters
    # deploy with zero gate_up weights, R29-verified path): when the gate_up
    # B weights are all zero the VE output is exactly zero, so the GEMM can
    # be skipped entirely. The flag lives on the pool weight tensor (stable
    # object; a reload binds a fresh tensor and recomputes).
    _gw = lora_info.gate_up_lora_b_weights
    _gu_zero = getattr(_gw, "_gu_zero_flag", None)
    if _gu_zero is None:
        _gu_zero = bool((_gw == 0).all().item())
        _gw._gu_zero_flag = _gu_zero
    with torch.cuda.stream(_mb_side):
        if has_active_lora and not _gu_zero:
            merged_experts_fused_moe_lora_add(
                output=gate_up_delta_ve,
                hidden_states=hidden_states,
                lora_a=lora_info.gate_up_lora_a_weights,
                lora_b=lora_info.gate_up_lora_b_weights,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                token_lora_mapping=token_lora_mapping,
                mul_routed_weight=False,
                experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
                experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
                routing_cache=fused_lora_routing_cache,
                fuse_add_to_output=False,
                use_direct_expand_add=lora_info.max_lora_rank <= 64,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
            )
        else:
            gate_up_delta_ve.zero_()
        if _inter > _inter_ve:
            gate_up_delta[..., : 2 * _inter_ve].copy_(gate_up_delta_ve)
        lora_event.record()
    lora_ready_handle = lora_event.cuda_event
    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # Pre-quantize activations to MXFP8 (E4m3), the same prep the non-LoRA
    # path uses (_fused_experts_flashinfer_mxfp4_sm100_trtllm_gen). The overlay
    # kernel already supports E4m3 input with a 32-vec scale
    # (kernel_launcher.cu:2796-2802: scale vec_size==32 -> mDtypeAct=MxE4m3,
    # no internal FP4 quant), which selects the MxE4m3_MxE2m1 GEMM cubins that
    # exist. Passing raw bf16 (scale=None) instead would trigger the internal
    # E2m1 quant path -> E2m1×E2m1 tile-8 GEMM -> "No kernel found" (no such
    # cubin in the package).
    from sglang.srt.layers.quantization.mxfp4 import (
        _prepare_flashinfer_mxfp8_activations,
    )

    x_padded, prepared_packed_topk, x_quant, x_scale = (
        _prepare_flashinfer_mxfp8_activations(hidden_states, quant_info.hidden_size)
    )

    # The gate_up VE delta must match the same padded layout the C++ activation
    # kernel consumes (2 * padded inter). VE writes the unpadded 2*inter_ve
    # wide slice; F.pad zero-pads it to the padded width (the padded tail has
    # no LoRA weights — correct by construction).

    packed_topk_ids = (
        prepared_packed_topk
        if prepared_packed_topk is not None
        else fused_pack_topk(topk_ids=topk_ids, topk_weights=topk_weights)
    )

    def _as_fp8(t: torch.Tensor) -> torch.Tensor:
        return (
            t.view(torch.float8_e4m3fn)
            if t.dtype == torch.uint8
            else t
        )

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        direct_down_output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    if _o.environ.get("SGLANG_LORA_DEEPGEMM_V3") == "1":
        # v3 path: bf16-bmm MoE-LoRA (DeepGEMM masked discarded; rel=3.6e-3 PASS).
        # GEMM segment only — delta VE stages and down-VE merge stay identical.
        # NVIDIA torch dequant (the quark mx.dq_mxfp4 import is AMD-only and
        # raised ImportError on CUDA — this was the sole blocker that kept the
        # v3 path dead). Weights: [N, K/2] int8 packed e2m1 (low nibble =
        # even k, high nibble = odd k); scales: [N, K/32] float8_e8m0fnu.
        # Chunked over the N dim + in-chunk scale broadcast to bound the
        # peak intermediate (~9.4GB float32 in one shot OOM'd at graph-capture
        # warmup with 858MB free; chunks keep it under ~700MB).
        def _dequant_mxfp4_torch(x_int8, scale_e8m0, float_dtype, chunk=8):
            LUT = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=torch.float32, device=x_int8.device,
            )
            n = x_int8.shape[-2]
            outs = []
            for i in range(0, n, chunk):
                xc = x_int8[..., i : i + chunk, :]
                sc = scale_e8m0[..., i : i + chunk, :].to(torch.float32)
                xu = xc.to(torch.uint8).to(torch.long)
                lo = (xu & 0xF)
                hi = (xu >> 4)
                vals = torch.stack([LUT[lo], LUT[hi]], dim=-1)
                # [.., n_c, K/2, 2] -> [.., n_c, K] -> [.., n_c, K/32, 32]
                vals = vals.reshape(*xc.shape[:-1], xc.shape[-1] * 2)
                vals = vals.reshape(*vals.shape[:-1], -1, 32)
                out = vals * sc.unsqueeze(-1)
                out = out.reshape(*out.shape[:-2], -1)
                outs.append(out.to(float_dtype))
            return torch.cat(outs, dim=-2)

        global _DG_V3_WCACHE
        try:
            _DG_V3_WCACHE
        except NameError:
            _DG_V3_WCACHE = {}
        _wkey = quant_info.w13_weight.data_ptr()
        if _wkey not in _DG_V3_WCACHE:
            _DG_V3_WCACHE[_wkey] = (
                _dequant_mxfp4_torch(quant_info.w13_weight,
                                     quant_info.w13_weight_scale, torch.bfloat16),
                _dequant_mxfp4_torch(quant_info.w2_weight,
                                     quant_info.w2_weight_scale, torch.bfloat16),
            )
        _w13_bf, _w2_bf = _DG_V3_WCACHE[_wkey]
        from sglang.kernels.ops.moe.trtllm_lora_temp.deepgemm_path import (
            v3_gemm_stage,
        )
        _g1a = (float(quant_info.gemm1_alpha)
                if quant_info.gemm1_alpha is not None else 1.702)
        _g1b = (float(quant_info.gemm1_beta)
                if quant_info.gemm1_beta is not None else 1.0)
        if _o.environ.get("SGLANG_V3_CONTIG") == "1":
            from sglang.kernels.ops.moe.trtllm_lora_temp.v3_contig import (
                v3_gemm_stage_contig,
            )
            import deep_gemm as _dg
            output, _ = v3_gemm_stage_contig(
                hidden_states, topk_ids, topk_weights, _w13_bf, _w2_bf,
                gate_up_delta, None, quant_info.gemm1_clamp_limit,
                activation_lora_input, quant_info.local_num_experts, _dg,
                gemm1_alpha=_g1a, gemm1_beta=_g1b)
        else:
            output, _ = v3_gemm_stage(
                hidden_states, topk_ids, topk_weights, _w13_bf, _w2_bf,
                gate_up_delta, None, quant_info.gemm1_clamp_limit,
                activation_lora_input, quant_info.local_num_experts, M_MAX=8,
                gemm1_alpha=_g1a, gemm1_beta=_g1b)
    else:
        output = trtllm_fp4_block_scale_routed_moe_lora(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=quant_info.w13_weight,
            gemm1_weights_scale=_as_fp8(quant_info.w13_weight_scale),
            gemm2_weights=quant_info.w2_weight,
            gemm2_weights_scale=_as_fp8(quant_info.w2_weight_scale),
            output1_scales_scalar=None,
            output1_scales_gate_scalar=None,
            output2_scales_scalar=None,
            gate_up_lora_delta=gate_up_delta,
            activation_lora_input=activation_lora_input,
            lora_ready_event=lora_ready_handle,
            num_experts=quant_info.global_num_experts,
            top_k=runner_config.top_k,
            intermediate_size=inter,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            routed_scaling_factor=(
                runner_config.routed_scaling_factor
                if runner_config.routed_scaling_factor is not None
                else 1.0
            ),
            routing_method_type=RoutingMethodType.TopK.value,
            do_finalize=True,
            output=direct_down_output,
            act_type=get_activation_type(
                runner_config.activation, is_gated=runner_config.is_gated
            ),
            # OAI / GPT-OSS gated-activation controls + FC1/FC2 biases, matching the
            # non-LoRA MXFP4 path (see _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen).
            gemm1_bias=quant_info.w13_weight_bias,
            gemm1_alpha=quant_info.gemm1_alpha,
            gemm1_beta=quant_info.gemm1_beta,
            gemm1_clamp_limit=quant_info.gemm1_clamp_limit,
            gemm2_bias=quant_info.w2_weight_bias,
        )

    # Down VE: the C++ activation kernel wrote PADDED inter (640) columns of
    # activation_lora_input; the down A buffer is UNPADDED (576) wide, so read
    # the first 576 columns (the padded tail has no LoRA weights). The slice
    # must be contiguous for the VE kernel's stride assumptions.
    if has_active_lora:
        merged_experts_fused_moe_lora_add(
            output=output,
            hidden_states=(
                activation_lora_input[..., :_inter_ve]
                if _inter > _inter_ve
                else activation_lora_input
            ).contiguous().view(-1, min(_inter, _inter_ve)),
            lora_a=lora_info.down_lora_a_weights,
            lora_b=lora_info.down_lora_b_weights,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_lora_mapping=token_lora_mapping,
            mul_routed_weight=True,
            experts_shared_outer_loras_a=lora_info.experts_shared_outer_loras,
            experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
            routing_cache=fused_lora_routing_cache,
            fuse_add_to_output=False,
            fuse_sum_all_reduce=True,
            use_direct_expand_add=lora_info.max_lora_rank <= 64,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
        )
    # else: pass-through — the down VE delta is zero, nothing to merge.
    if _o.environ.get("SGLANG_MLP_OUT_LORA") == "1":
        # Synthetic MLP-output-side dense LoRA (perf probe for the output-side
        # adapter topology; values are random, cost depends on shape only).
        global _MLP_OUT_W
        try:
            _MLP_OUT_W
        except NameError:
            _g = torch.Generator(device="cuda").manual_seed(7)
            _MLP_OUT_W = (
                torch.randn(16, output.shape[-1], device=output.device,
                            dtype=torch.bfloat16, generator=_g) * 0.02,
                torch.randn(output.shape[-1], 16, device=output.device,
                            dtype=torch.bfloat16, generator=_g) * 0.02)
        _wa, _wb = _MLP_OUT_W
        output = output + (output @ _wa.T) @ _wb
    return StandardCombineInput(hidden_states=output)
