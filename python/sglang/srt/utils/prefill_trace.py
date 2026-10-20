"""CPU metadata for correlating prefill chunk boundaries with request scores."""


def prefill_batch_trace_record(batch):
    """Describe scheduled spans without inspecting token values or GPU tensors.

    This is a diagnostic record, not a kernel timing. Invalid/missing CPU
    metadata is explicit so a trace cannot silently invent request boundaries.
    The caller controls opt-in logging and rank selection.
    """
    mode = batch.forward_mode
    if mode is None or not mode.is_extend():
        return None
    record = {"forward_iter": batch.forward_iter, "mode": mode.name}
    prefixes, lengths = batch.prefix_lens, batch.extend_lens
    if (
        not isinstance(prefixes, list)
        or not isinstance(lengths, list)
        or len(prefixes) != len(lengths)
        or len(lengths) != len(batch.reqs)
        or not all(type(v) is int and v >= 0 for v in prefixes + lengths)
    ):
        return {**record, "cpu_metadata_available": False}
    requests = []
    for req, prefix, length in zip(batch.reqs, prefixes, lengths):
        origin = req.origin_input_ids
        requests.append(
            {
                "rid": req.rid,
                "lora_id": req.lora_id,
                "prompt_tokens": len(origin)
                if isinstance(origin, (list, tuple))
                else None,
                "prefix_tokens": prefix,
                "extend_tokens": length,
                "extend_end": prefix + length,
            }
        )
    return {
        **record,
        "cpu_metadata_available": True,
        "extend_num_tokens": batch.extend_num_tokens,
        "span_tokens_sum": sum(lengths),
        "requests": requests,
    }
