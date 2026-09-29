"""experimental_sgl_trtllm MoE LoRA dispatch (original single-stream).

This is the LoRA-enabled fused-experts path added by the trtllm-lora work — it
was originally a function in ``layers/moe/moe_runner/flashinfer_trtllm.py`` and
is now hosted here so that file holds only a re-export. The function name
remains ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora`` for
import-site stability.

When ``SGLANG_LORA_TWO_STREAM=1`` is set, this is the function the
``install_two_stream_overrides()`` monkey-patch swaps for the side-stream
version in :mod:`sglang.srt.lora.trtllm_lora_temp.moe_overlap`. Otherwise it runs as
the active path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmBf16MoeQuantInfo,
        FlashInferTrtllmFp4MoeQuantInfo,
        FlashInferTrtllmFp8MoeQuantInfo,
    )
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def fused_experts_none_to_experimental_sgl_trtllm_fp8_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp8_block_scale_moe_lora_finalize,
        trtllm_fp8_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.lora.lora_moe_runners import build_lora_hooks
    from sglang.srt.lora.trtllm_lora_temp.sgl_fp8_moe import (
        fused_experts_fp8_sgl,
    )
    from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
        maybe_overlap_staged_shared_add,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

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

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    if not get_is_capture_mode() and not lora_info.has_active_lora:
        return fused_experts_fp8_sgl(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    if use_virtual_lora_store:
        hooks = None
        token_lora_mapping = lora_info.token_lora_mapping
        fused_lora_routing_cache: dict = {}
    else:
        hooks = build_lora_hooks(hidden_states, lora_info, topk_ids)
        token_lora_mapping = None
        fused_lora_routing_cache = {}

    # Fuse the per-token scale transpose into the quant kernel (column-major scales) so the
    # `.t()` is a free view -> drops the standalone ~2us transpose+copy. Byte/shape-identical.
    a_q, a_sf = per_token_group_quant_fp8(
        hidden_states, quant_info.weight_block_k, column_major_scales=True
    )
    a_sf_t = a_sf.t()

    # EP-aware LoRA: under MoE EP each rank computes the delta only for the experts it
    # owns (passed via local_expert_offset/local_num_experts below). gate_up_delta stays
    # new_empty even though non-owned [token, k] slots are then left unwritten -- the
    # trtllm MoE is itself EP-aware, so those slots never feed the all-reduced output.
    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    gate_up_delta = (
        hidden_states.new_empty(gate_up_delta_shape)
        if use_virtual_lora_store
        else hidden_states.new_zeros(gate_up_delta_shape)
    )
    if use_virtual_lora_store:
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
        )
    elif hooks.after_gate_up is not None:
        hooks.after_gate_up(hidden_states, gate_up_delta, topk_weights, topk_ids)

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

    direct_down_output = None
    if use_virtual_lora_store:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            direct_down_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

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
        do_finalize=use_virtual_lora_store,
        output=(
            direct_down_output
            if direct_down_output is not None
            else torch.empty_like(hidden_states)
        ),
        tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        activation_type=quant_info.activation_type,
    )
    if use_virtual_lora_store:
        output = moe_result
        # Shared-add overlap: the trtllm op above already finalized `output`, so the
        # staged shared-expert add (if any) can run on the main stream concurrent with
        # the down-LoRA shrink below; the expand waits on it via expand_wait_event.
        shared_add_done = maybe_overlap_staged_shared_add(output)
        merged_experts_fused_moe_lora_add(
            output=output,
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
            expand_wait_event=shared_add_done,
        )
        return StandardCombineInput(hidden_states=output)

    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = moe_result

    down_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        hidden_states.shape[1],
    )
    down_delta = (
        hidden_states.new_empty(down_delta_shape)
        if use_virtual_lora_store
        else hidden_states.new_zeros(down_delta_shape)
    )
    if use_virtual_lora_store:
        merged_experts_fused_moe_lora_add(
            output=down_delta,
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
        )
    elif hooks.after_down is not None:
        hooks.after_down(
            activation_lora_input.view(-1, quant_info.intermediate_size),
            down_delta,
            topk_weights,
            topk_ids,
        )

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        output = torch.empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    output = trtllm_fp8_block_scale_moe_lora_finalize(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        down_lora_delta=down_delta,
        output=output,
        routed_scaling_factor=(
            runner_config.routed_scaling_factor
            if runner_config.routed_scaling_factor is not None
            else 1.0
        ),
    )

    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_bf16_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
    """BF16 sibling of ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora``.

    Decomposed (unfused-activation) MoE-LoRA, bf16 end-to-end (no quantization):
    routing -> gather -> gate_up grouped GEMM (raw 2*inter, bf16) -> activation that
    adds ``gate_up_lora_delta`` pre-SwiGLU and captures ``activation_lora_input`` ->
    down grouped GEMM -> finalize, then the virtual-experts down-LoRA is merged into
    the output. Single-stream version (no two-stream overlap yet — phase 2).
    """
    from sglang.kernels.ops.moe.trtllm_lora_temp import trtllm_bf16_routed_moe_lora
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        fused_experts_none_to_flashinfer_trtllm_bf16,
        get_activation_type,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm BF16 LoRA currently supports the gated SwiGLU path only."
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    # No-LoRA non-capture decode -> fast bf16 path, valid only for 4-D block-shuffled
    # weights ([E, M//128, K//128, 128]); flat [E, 2F, D] stays on the decomposed kernel.
    if (
        not get_is_capture_mode()
        and not lora_info.has_active_lora
        and quant_info.gemm1_weights.dim() == 4
    ):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output, quant_info, runner_config, use_routed_topk=True
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    assert use_virtual_lora_store, "BF16 trtllm LoRA requires virtual-experts."
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = runner_config.intermediate_size_per_partition

    # Gated gate_up LoRA delta (same shape/semantics as the fp8/fp4 paths). EP args scope
    # the delta to this rank's experts, matching the EP-aware trtllm MoE base.
    # TEMP-DEBUG(bisect): skip the VE kernels; zero deltas isolate whether the
    # crash is in the VE Triton path or the flashinfer C++ op.
    gate_up_delta = torch.zeros(
        (hidden_states.shape[0], runner_config.top_k, 2 * inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if False:
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
    )

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
    )

    # TEMP-DEBUG(bisect): skip the down VE too — pure C++ op run.
    if False:
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


def fused_experts_none_to_experimental_sgl_trtllm_fp4_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
    """NVFP4 sibling of ``fused_experts_none_to_experimental_sgl_trtllm_fp8_lora``.

    Decomposed (unfused-activation) MoE-LoRA: routing -> gather -> gate_up grouped
    GEMM (raw 2*inter) -> activation that adds ``gate_up_lora_delta`` pre-SwiGLU and
    captures ``activation_lora_input`` -> NvFP4 quant -> down grouped GEMM -> finalize,
    then the virtual-experts down-LoRA is merged into the output. Single-stream
    version; ``moe_overlap.py`` provides the two-stream variant.
    """
    from sglang.kernels.ops.moe.trtllm_lora_temp import (
        trtllm_fp4_block_scale_routed_moe_lora,
    )
    from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import fused_pack_topk
    from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
        merged_experts_fused_moe_lora_add,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        fused_experts_none_to_flashinfer_trtllm_fp4,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    assert runner_config.activation == "silu" and runner_config.is_gated, (
        "experimental_sgl_trtllm NVFP4 LoRA currently supports the gated SwiGLU path only."
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)
    assert runner_config.top_k is not None

    # No active LoRA in a non-capture decode -> plain (fast) FP4 path.
    if not get_is_capture_mode() and not lora_info.has_active_lora:
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, quant_info, runner_config, use_routed_topk=True
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    assert use_virtual_lora_store, "NVFP4 trtllm LoRA requires virtual-experts."
    token_lora_mapping = lora_info.token_lora_mapping
    fused_lora_routing_cache: dict = {}

    inter = quant_info.intermediate_size_per_partition

    # Path 3: feed bf16 hidden DIRECTLY to the op (no python pre-quant). The op permutes the
    # bf16 hidden (moe::dev::permute, token->expert order) then NvFP4-quantizes ONCE with the
    # 1/(448*6) global + per-token scale — eliminating the dequant->permute->requant round-trip
    # (and its magnitude bug) that pre-quantized fp4 input forced. Requires the layer to be in
    # per-token-activation mode (SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1) so that
    # g1_scale_c == g1_alphas and g2_alphas == w2_weight_scale_2, making the decomposed
    # gate_up(g1_alphas)/SwiGLU/down(g2_alphas) scale composition match the plain fused path.

    gate_up_delta_shape = (
        hidden_states.shape[0],
        runner_config.top_k,
        quant_info.w13_weight.shape[1],
    )
    # Gated gate_up LoRA delta: a single merged_experts call on the full stacked [gate_A; up_A]
    # lora_a (rank 2r) and [gate_B; up_B] lora_b (rank r), via the rank-specialized direct expand
    # (use_direct_expand_add, rank <= 64). EP args scope the delta to this rank's experts, matching
    # the EP-aware trtllm MoE base.
    gate_up_delta = hidden_states.new_empty(gate_up_delta_shape)
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
        # V4.1 ships gate_up B shared too (3-D [1, N, r], same as A): a
        # per-expert B would explode the expand routing to
        # n_routed(384) * max_loras virtual buckets (>=1024 →
        # _align_block_size_large, whose JIT kernel faults) and the B buffer
        # to ~450MB/layer/rank.
        experts_shared_outer_loras_b=lora_info.experts_shared_outer_loras,
        routing_cache=fused_lora_routing_cache,
        fuse_add_to_output=False,
        use_direct_expand_add=lora_info.max_lora_rank <= 64,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )

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
        # EP-aware: scope the down delta to this rank's experts, matching the gate_up
        # call above and the FP8 down call. Harmless at EP=1 (local==global, Kimi today);
        # required for correctness if MoE-EP is turned on later (otherwise non-owned
        # experts' deltas get over-counted by the fuse_sum_all_reduce).
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
    )
    return StandardCombineInput(hidden_states=output)


def fused_experts_none_to_experimental_sgl_trtllm_mxfp4_lora(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmGenMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    lora_info,
) -> StandardCombineInput:
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

    assert runner_config.is_gated, (
        "experimental_sgl_trtllm MXFP4 LoRA requires a gated activation."
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output), (
        "experimental_sgl_trtllm MXFP4 LoRA needs materialized topk_ids; the "
        "logits-based (bypassed) routing does not expose them."
    )
    assert runner_config.top_k is not None

    # No active LoRA in a non-capture decode -> plain (fast) MXFP4 path.
    if not get_is_capture_mode() and not lora_info.has_active_lora:
        return _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output, quant_info, runner_config
        )

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    use_virtual_lora_store = bool(
        lora_info.lora_use_virtual_experts and lora_info.max_lora_rank > 0
    )
    assert use_virtual_lora_store, "MXFP4 trtllm LoRA requires virtual-experts."
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

    gate_up_delta_ve = hidden_states.new_empty(
        (hidden_states.shape[0], runner_config.top_k, 2 * _inter_ve)
    )
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
    gate_up_delta = (
        torch.nn.functional.pad(
            gate_up_delta_ve, (0, 2 * _inter - 2 * _inter_ve)
        )
        if _inter > _inter_ve
        else gate_up_delta_ve
    )

    activation_lora_input = torch.empty(
        (hidden_states.shape[0], runner_config.top_k, inter),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    packed_topk_ids = fused_pack_topk(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    # The op indexes the scale buffers as fp8-e4m3, exactly like the NVFP4 LoRA
    # path — the raw uint8 (e4m3 x2 / ue8m0) buffers must be viewed, not passed
    # packed, or the GEMM reads out-of-range block scales.
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

    output = trtllm_fp4_block_scale_routed_moe_lora(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_weights=quant_info.w13_weight,
        gemm1_weights_scale=_as_fp8(quant_info.w13_weight_scale),
        gemm2_weights=quant_info.w2_weight,
        gemm2_weights_scale=_as_fp8(quant_info.w2_weight_scale),
        output1_scales_scalar=None,
        output1_scales_gate_scalar=None,
        output2_scales_scalar=None,
        gate_up_lora_delta=gate_up_delta,
        activation_lora_input=activation_lora_input,
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
    return StandardCombineInput(hidden_states=output)
