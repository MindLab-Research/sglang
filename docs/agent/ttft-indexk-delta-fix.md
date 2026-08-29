# TTFT ∝ Context Length on PD Near-Full-Hit Requests — index_k Full-Transfer Root Cause (2026-08-29)

## Symptom

`.6/.7/.9` GLM-5.3 2P1D (B200, mooncake TCP, DCP=8): TTFT grew linearly with
context length even at ~99% radix hit. An 827K-ctx request (t213-607,
cached=825920/827K) took ~20s to first token: prefill delta forward was 1s,
but the prefill-batch → kv-arrived window was **14s**. 190K ctx was fine (~3s).

## Root Cause

`_dsa_payload` built the DSA state (index_k) page list over the **full
sequence** (`[:seq_len]`) on both sides, with no prefix truncation — unlike
the main-KV delta slice (`kv_indices[:origin_input_len - prefix_len]`).

Even when decode radix held 99.8% of the context, the whole sequence's
index_k was re-shipped per request: ~8.4KB/page × 12922 pages × 21 physical
indexer layers ≈ **2.3GB over mooncake TCP (~163MB/s) ≈ 14s**. Linear in ctx
→ TTFT ∝ ctx. index_k rows share the main-KV page lifecycle, so the radix-hit
rows are already resident on decode; re-shipping them is pure waste.

## Fix (commit 53d9447b6d)

Skip `[0, decode_prefix_len)` in both `_dsa_payload` builders —
`disaggregation/decode.py` (uses `total_prefix_len`) and
`disaggregation/prefill.py` (uses `req.disagg_decode_prefix_len` via
`pop_decode_prefix_len`). The src/dst page lists pair 1:1 positionally, so
both sides MUST truncate identically.

Kill-switch: `SGLANG_DSA_INDEX_K_FULL_TRANSFER=1` restores legacy full-seq
transfer.

## Verification (.6/.7/.9 deployed, two-round 686K test)

| Round | Before | After |
|---|---|---|
| Cold (radix empty) | ~51s E2E (unchanged — full path by design) | 51.1s |
| Near-full-hit | ~20s TTFT (t213-607 observed) | **1.99s TTFT** (10×) |

Output correctness: reasoning (898 chars) + content (343 chars) clean and
accurate — confirms the index_k row-retention premise (radix-held pages keep
their index_k rows intact; skipped transfer is safe).

## Deploy Notes

- 3 containers (`.6/.7` glm53-prefill, `.9` glm53-decode) via docker cp +
  paired restart (mooncake session invalidation rule).
- `docker restart` can fail with "container is zombie and can not be killed"
  on these nodes: the zombie gets reaped within a minute, then
  `docker start <c>` works (no need to recreate the container).
- Router restart after engine restart clears stale health-cache / circuit
  state (sglang-router on `.6:30000`, no api-key).

## Related Threads (not this fix)

- First-attempt prefill send failure (`error sending request` on 827K
  bodies, smg→engine connection layer) contributes ~2-3s occasionally.
- `Decode transfer failed` (t213-608, 827K/92% hit) — separate transfer
  timeout family, not addressed by this commit.
