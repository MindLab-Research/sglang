"""Registers the ``experimental_sgl_trtllm`` MoE fused-func.

``MoeRunner.__init__`` requires a registered fused-func at CONSTRUCTION time even
for the LoRA case, because LoRA is attached *after* the MoE layer is built (so
``lora_enabled`` is False inside ``MoeRunner.__init__``). At run time the runner
skips this for the LoRA path; the no-LoRA path delegates to the standard
flashinfer_trtllm dispatch.

Registration occurs when this module is imported during model construction.
"""

from sglang.srt.layers.moe.moe_runner.base import register_fused_func


@register_fused_func("none", "experimental_sgl_trtllm")
def fused_experts_none_to_experimental_sgl_trtllm(
    dispatch_output, quant_info, runner_config
):
    # Without LoRA, delegate by quant-info type. The sglang MXFP4
    # (DeepSeek-V4.1) payload has its own entry point on the flashinfer_trtllm
    # module; everything else uses the standard dispatch there.
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
        fused_experts_none_to_flashinfer_trtllm,
    )

    if isinstance(quant_info, FlashInferTrtllmGenMxfp4MoeQuantInfo):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
        )

        return _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output, quant_info, runner_config
        )

    return fused_experts_none_to_flashinfer_trtllm(
        dispatch_output, quant_info, runner_config
    )
