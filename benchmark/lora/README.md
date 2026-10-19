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
For long-context diagnostics, `--skip-prefill-logprobs` avoids materializing
full-prompt vocabulary logits and scores only `prefill_last` and cached decode.
The omitted `prefill` phase is recorded as `not_measured` with a null gate result;
it cannot count as prompt-token equivalence passing. Keep the default full
prompt scoring for the primary short-input acceptance matrix.
`hot` means no explicit cache flush. Scoring every prompt token from position
zero can prevent prefix reuse; inspect per-case `cached_tokens` and test actual
cache restores separately.

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

## Check prefix consistency

Shape-dependent arithmetic can make an unchanged causal prefix score differently
when more tokens follow it. Isolate this within each server before attributing a
dynamic-versus-merged difference to LoRA:

```bash
python benchmark/lora/compare_prefix.py \
  --url http://127.0.0.1:30000 --corpus /evidence/corpus.json \
  --output /evidence/prefix-base
```

The longest prompt is selected by default; `--case-id` selects one explicitly.
`--positions` supplies prefix lengths. Optional `--adapter-name` and
`--adapter-path` load and score an adapter. The script flushes before every
request, asserts identical target token IDs, and retains full raw responses.
It changes the experiment server's cache and adapter state; use a dedicated
instance without concurrent benchmark clients.

For a deterministic control, restart both servers with the same
`--enable-deterministic-inference --attention-backend triton
--linear-attn-prefill-backend triton --bf16-gemm-backend torch` configuration and
repeat this diagnostic. Freeze the resolved configuration and measure its cost
separately before choosing a serving configuration. Prefix consistency alone
does not establish dynamic-versus-merged equivalence or long-context support.

## Check mixed request routing

Prepare an adapters JSON list such as
`[{"name": "A", "path": "/adapters/A"}, {"name": "B", "path": "/adapters/B"}]`,
then run against a dedicated server with sufficient LoRA slots:

```bash
python benchmark/lora/compare_mixed.py \
  --url http://127.0.0.1:30000 --adapters /evidence/adapters.json \
  --corpus /evidence/corpus.json --capture-dir /evidence/mixed-captures \
  --output /evidence/mixed
```

The tool establishes isolated base/adapter references, freezes a continuation for
each, and compares prefill-last/decode scores in three shuffled mixed batches.
It checks captured step/token indices, retains raw responses and per-case scores,
and unloads its adapters afterwards. Aggregate passing does not imply every
case passes; inspect individual failures and argmax differences. This test does
not establish merged-model equivalence, in-flight unload safety, or cache restore
correctness. Stateful processors must skip discarded intermediate prefill
chunks; otherwise a long or mixed request can advance the capture counter before
it produces its first output token.

Under deterministic inference, GDN scheduler chunks align to the Triton
recurrence's 64-token tiles, combined with the attention backend's alignment.
Splitting inside a recurrence tile can change FP32 state and later decode scores
depending on the other requests in the batch. Keep the per-rank prefill chunk
at least as large as this alignment; smaller chunks fail at startup. Validate
cache-state dtype and cache restore separately when changing serving settings.

Explicit adapter unload removes its automatic-reload metadata immediately and
waits for existing requests to drain. New requests using that alias are rejected
until an explicit load succeeds. Automatic LRU eviction retains reload metadata;
it does not have the same lifecycle semantics as explicit unload.

The deterministic Triton FP32 GEMM path now honors
`torch.backends.cuda.matmul.allow_tf32`: FP32 dot products use IEEE precision
unless TF32 is explicitly enabled. BF16 and FP16 dot products are unaffected.
High-precision arithmetic hooks used for diagnosis change the computation and
must not be reported as acceptance of the native BF16 LoRA path.
