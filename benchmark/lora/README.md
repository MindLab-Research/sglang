# Dynamic LoRA versus merged-model comparison

This comparison uses two independent SGLang servers:

- D: original base model with the adapter loaded dynamically.
- M: the same base and adapter merged offline, served without `--enable-lora`.

Freeze model revisions, adapter files, tokenizer, RoPE, TP, dtype, scheduling,
cache settings and source/image versions before comparing. Both servers and the
client must see the same capture directory at the same absolute path. Both
servers require `--enable-custom-logit-processor`; enable this only for the
controlled benchmark environment.

## Prepare the reference

```bash
python benchmark/lora/merge_reference.py \
  --base /models/base --adapter /adapters/example \
  --output /models/merged-example --device cuda
```

The merger applies `W + (alpha / rank) * B @ A` in FP32 with TF32 disabled,
then exports BF16 by default. `merge-manifest.json` records hashes, mappings,
changed tensors and rounding statistics. Unchanged files are hardlinked: keep
both source and output immutable. Unsupported PEFT variants are rejected.
`--export-dtype float32` provides an unrounded diagnostic artifact; it does not
imply that the serving model supports FP32 execution.

## Score identical prefixes

Launch D and M with the same configuration, except model paths and LoRA support,
then run:

```bash
python benchmark/lora/compare_merged.py \
  --dynamic-url http://127.0.0.1:30000 \
  --merged-url http://127.0.0.1:30002 \
  --adapter-name example --adapter-path /adapters/example \
  --corpus /evidence/corpus.json --capture-dir /evidence/captures \
  --output /evidence/comparison
```

The corpus is a JSON list of objects with unique `id` and integer `input_ids`.
Defaults are 30 inputs, 256 continuation tokens, four rounds, cold and hot cache.
Reduced `--limit`, `--decode-tokens` and `--rounds` runs are diagnostics, not
completion of that matrix. Output directories must be new.

M generates a common continuation first. The processor then records unmodified
FP32 log-softmax and argmax before forcing each common token in both servers.
Step zero is `prefill_last`; later steps use actual cached decode. API output
logprobs after forcing are deliberately excluded. Prompt input logprobs are
reported separately as `prefill`. Asserted step indices and output IDs prevent
accidental comparisons of different prefixes, including with overlap scheduling.

Raw server responses, continuation tokens, capture files, per-case worst
positions, repeated-run noise and aggregate statistics are retained. The fixed
logprob gates are mean absolute difference <= 0.01 nat and P99 <= 0.1 nat,
separately for each phase. Maximum error and argmax agreement are also reported.
A short run passing, a kernel test passing, or HF/PEFT agreement does not establish
end-to-end acceptance. Do not relax thresholds to accommodate observed failures.
