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

## Experimental BF16 merged-rounding backend

The opt-in `--lora-backend merged_bf16` backend keeps one immutable BF16 base
model and FP32 adapter factors. For each active adapter and projection, it
computes `(W.float() + (B @ A) * scale).bfloat16()`, runs the BF16 projection,
then releases that temporary weight. It does not retain complete merged models
per adapter. This deliberately adds weight traffic and merge arithmetic to
every forward by default; measure latency and peak memory before choosing it
for serving.

`--lora-merged-bf16-cache-size-mb N` optionally retains merged projection
weights in a per-rank LRU bounded by N MiB. Zero, the default, keeps the original
transient behavior. The full budget is reserved before KV pool sizing; it is
not free KV capacity. Reused or unloaded adapter slots invalidate their cached
weights, and projections larger than the budget bypass caching. Cold fills and
evictions still execute the same FP32 merge and BF16 rounding. A warm hit reuses
that exact tensor. This option is under validation; measure cold-fill latency,
warm throughput, mixed-adapter eviction and actual peak memory together.

`--lora-merged-bf16-cache-policy active` is an experimental alternative to the
default `lru` policy and requires a positive cache budget. It retains projections
for adapters in the current local/global routing batch and bypasses admission
when those entries fill the budget, instead of evicting the whole working set
on a cyclic scan. One largest local BF16 target projection is reserved inside
the same budget for a bypassed temporary weight; FP32 merge/GEMM workspace is
separate. Unloaded or reused slots still invalidate their projections. CPU
counters for hits, misses, materialized BF16 bytes, bypasses and evictions are
logged as `projection_cache_stats` at slot updates. Byte counters describe tensor
extents, not measured memory bandwidth. Eight admission-policy unit tests plus
ten existing configuration/TP checks passed in the matching image; GPU accuracy,
mixed-adapter throughput and peak-memory comparisons remain pending. Keep the
policy opt-in until those comparisons complete.

Choose this budget from the adapted projection shapes, then check the resolved
KV and request limits after startup. On the dedicated B300 TP1 experiment,
Qwen3.8-27B's real dense adapter changes 256 BF16 tensors totaling 35 GiB.
`--lora-merged-bf16-cache-size-mb 36864 --max-total-tokens 65536` retained a
65,536-row KV pool with 256 Mamba slots at static memory fraction 0.5. The first
40 GiB attempt left only 1,565 KV rows and rejected a 1,630-token test input;
that attempt is a capacity failure, not a passing comparison. These numbers
describe this model, adapter and GPU configuration, not a general default.

The 36 GiB retry passed the independent-merge comparison: 240 pairs and
111,184 scores for the real adapter, plus eight pairs and 3,508 scores for the
fused GDN adapter, all with zero logprob differences and matching argmax.
Mixed routing and 108 LRU reload requests also passed their consistency checks.
Those checks do not establish actual KV-prefix restoration or a speedup. Measure
multiple adapters too: a budget that holds one adapter can evict weights during
every forward when several active adapters exceed it. Keep the KV pool equal
in cached, uncached and merged speed controls; 32 requests of 2,048 input plus
256 output tokens exceed a 65,536-row pool and can queue despite a request
limit of 32.

The subsequent cached/uncached/independent-merge natural SSE controls each
completed 636 requests at 16 points on the same GPU2, R34 runtime, 65,536-row
KV pool and 256 Mamba slots. All 636 outputs matched across the three controls.
After two warmup batches, three cold and hot repetitions gave these cold,
concurrency-32 observations:

| Input / output tokens | Uncached tok/s | Cached tok/s | Merged tok/s | Cached P95 / merged P95, seconds |
| --- | ---: | ---: | ---: | ---: |
| 128 / 128 | 353.29 | 555.59 | 570.94 | 7.40 / 7.26 |
| 2048 / 256 | 166.88 | 260.35 | 265.08 | 31.46 / 30.88 |

Cached throughput was 97.31% and 98.22% of the merged controls respectively.
Other experiments ran concurrently, so these are observed ratios; dedicated
machine repeats in both cache-control orders remain pending. The short profiles
contained 16,399 / 8,719 / 8,537 CUDA kernel events for uncached / cached / merged.
The cached trace had no repeated FP32 weight-merge or rounding kernels, versus
2,560 of each in the uncached trace. All three retained 4,340 activation GEMM
and 1,056 GDN/convolution events. Trace duration sums are not end-to-end wall-time
shares. These warmed measurements do not cover first projection fill, multiple
active adapters, actual KV-prefix restoration or their lifecycle acceptance.
The [aggregate result snapshot](results/qwen38-27b-b300/2026-10-19/manifest.json)
includes resolved execution settings, these metrics and raw-artifact hashes;
it contains no prompts, output token IDs or weights. Complete raw requests and
traces remain under the dedicated experiment's
`qwen38-mol-multigpu-20261019` evidence root.

The snapshot also records the first completed million-total-context warmup on
R26: D, M and base each returned 256 tokens after the same 999,744-token input,
with D/M output IDs matching. First-warm TTFT was 2,001.23 / 1,993.29 seconds
for D/M, explaining the earlier 1,800-second client read timeout. This evidence
covers one fixed input and its first warmup; the repeated 18-request speed
matrix and full-prompt/multi-input long-context accuracy gates remain pending.

Its current supported configuration is DP1/PP1, unquantized BF16 weights,
deterministic inference, eager execution, an explicit maximum LoRA rank of
1 through 64, and dense targets only:

```text
--lora-backend merged_bf16 --max-lora-rank 32
--lora-target-modules qkv_proj o_proj gate_up_proj down_proj in_proj_qkvz in_proj_ba out_proj
--enable-deterministic-inference
--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
--attention-backend triton --linear-attn-prefill-backend triton
--bf16-gemm-backend torch
```

Speculative decoding, overlap loading, embeddings, LM head and MoE targets
are rejected. Qwen GDN input/output projections use the same packed dense
path. TF32 must remain disabled. Existing backends keep their original
factor dtype and projection path. Packed QKV/gate-up projections retain their
separate A factors and leave any unadapted output suffix unchanged. Native
cuBLAS FP32 `mm.out` and a separate scale/add precede BF16 rounding. The merge
GEMM uses fixed weight shapes and retains cuBLAS under the batch-invariant
operator overrides; the BF16 activation GEMM remains deterministic. Equality with
the independent reference must still be established by the full comparison
matrix above, including real adapters, mixed routing and long contexts.

The pool records target presence per layer and slot on every adapter load;
projections absent from that adapter use the unchanged base weight. Unknown
metadata keeps the merge path. A single-adapter batch avoids routing copies.
On CUDA, scaling, adding the base and BF16 writeback share one kernel with FMA
disabled, preserving the separate FP32 multiply/add rounding. The native FP32
merge GEMM and transient-weight lifetime remain unchanged.

For GDN models, check the resolved request limit in startup logs as well as
`--max-running-requests`: the Mamba state pool can cap it below the requested
value. Reserve enough state slots for the concurrent requests and cache
snapshots, and report observed GPU batch sizes separately from HTTP concurrency.

The merged row-parallel projection reduces the locally merged output once;
it must not enter the factorized A/B reduction path afterwards. Attention DP
uses local attention routing and globally gathered MLP routing. Even an idle or
base-only local attention rank must join those global metadata collectives.
The existing attention-DP LoRA restriction still applies: adapters must be
pinned, and the supported layout has `--attn-dp-size` equal to `--tp-size`.

For Qwen3.8-27B, TP2 and TP2/attention-DP2 each passed the same-topology
30-input, 256-token, four-round cold/hot comparison: 240 pairs and 111,184
scores, with zero differences in all three phases. This covers the real dense
adapter under deterministic eager BF16 execution. TP4 also passed the same
240-pair / 111,184-score main comparison, plus eight pairs / 3,508 scores for
the fused GDN adapter, all with zero differences and matching argmax. Its D/M
natural performance controls each completed 636 requests with identical output
tokens. Four dynamic rank traces contained 70,380 CUDA kernel events; each rank
had 2,560 FP32 weight merges, 2,560 rounding kernels and 1,290 NCCL BF16 tree
all-reduces. This short profile does not establish a wall-time bottleneck or a
TP speedup. TP4's pure-base performance control remains pending. Larger
TP/attention-DP layouts, other adapted GDN layouts, DCP and PD require their own
acceptance results; accepting a configuration at startup is not such a result.
The same result snapshot includes TP4's per-phase accuracy audit and all four
dynamic rank trace hashes and kernel counts.

## Compare sequential instances on the same GPUs

When D and M cannot coexist, export and seal M's original scores, stop M, then
replay the sealed reference on D with the same GPU placement and configuration:

```bash
python benchmark/lora/compare_frozen.py reference \
  --url http://127.0.0.1:30002 --topology tp2-dp1-pp1 \
  --corpus /evidence/corpus.json --provenance /evidence/source.json \
  --output /evidence/merged-reference
# Stop M, then launch D on the same GPUs and with the same numerical settings.
python benchmark/lora/compare_frozen.py dynamic \
  --url http://127.0.0.1:30000 --topology tp2-dp1-pp1 \
  --reference /evidence/merged-reference \
  --adapter-path /adapters/example --provenance /evidence/source.json \
  --output /evidence/dynamic-replay
```

The provenance JSON is a user-supplied source/image and input manifest. The
client retains the resolved server configuration, responses and per-rank
captures, hashes every reference artifact, and verifies that seal before
loading D's adapter. It checks execution-setting equality, step/token indices
and identical scores across sampling TP replicas. A reference-only run has
no D/M score result. Cold/hot labels still do not establish actual cache reuse.
Attention-DP replay loads a pinned adapter automatically. Output directories
must be new, and the reference must remain unchanged.

The experimental direct PD client adds `--prefill-url` and `--bootstrap-port`
to both stages. It sends paired native requests with the same bootstrap room,
retains both sides' raw responses, and joins prefill step zero with decode
steps one onward before comparing scores. It rejects a fake transfer backend.
The protocol unit tests use mocked endpoints; actual KV/SSM transfer,
same-layout D/M accuracy, transfer costs and 4P4D kernels must be validated on
real servers before claiming PD support or speed. Stop and start both sides
of an experimental PD pair together.

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

## Runnable lifecycle and KV-prefix diagnostics

The following TP1 clients use only the Python standard library and can also be
copied into a checkout pinned to an earlier serving commit. Run them on an
otherwise idle, dedicated D/M experiment after creating the frozen corpus and
independent merged model described above. Output directories must be new.

```bash
python3 benchmark/lora/check_lifecycle.py \
  --url http://127.0.0.1:30000 \
  --adapters /work/mol/evidence/adapters.json \
  --corpus /work/mol/evidence/corpus.json \
  --output /work/mol/evidence/lifecycle \
  --cycles 100 --bad-loads 100 --aliases 12 --lru-rounds 9

python3 benchmark/lora/check_cache_restore.py \
  --dynamic-url http://127.0.0.1:30000 \
  --merged-url http://127.0.0.1:30002 \
  --adapter-path /work/mol/adapter-A \
  --corpus /work/mol/evidence/corpus.json \
  --output /work/mol/evidence/cache-restore
```

For lifecycle testing, `adapters.json` is a list of distinct adapter paths such
as A, B and zero. `--aliases` must exceed the server's `max_loaded_loras` budget;
12 aliases and 9 rounds produce 108 LRU requests with the example budget of 10.
The client checks 100 alias reuse cycles, 100 rejected loads interleaved with
healthy requests, 256-token generation overlapping unload, and rejection of new
requests while the old generation drains. Natural output IDs and logprobs must
match that adapter's same-service deterministic reference exactly. This does
not establish independent merged-model precision or every malformed export's
validation behavior. Keep all other load/unload clients stopped during this run.

The cache client requires at least one corpus prompt of 128 tokens, real hot
`cached_tokens > 0` on both D and M, and identical 16-token natural continuations.
It checks output-token logprobs (mean <= 0.01, P99 <= 0.1 nat), plus exact
cold/hot self-repeat equality. A divergent natural path is a failed diagnostic;
use the forced-prefix comparison to investigate it. This single-prefix check
does not score full prefill or establish cross-adapter cache isolation. Both
clients write raw responses and only write `summary.json` after passing; failed
runs remain for inspection. The CPU fake-HTTP tests validate client protocol
and rejection paths, not GPU serving correctness.
