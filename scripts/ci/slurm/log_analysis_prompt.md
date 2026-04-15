# srtslurm Log Analysis

You are analyzing logs from a failed srtslurm job. srtslurm is a Python-first
orchestration framework for running distributed LLM inference benchmarks on
SLURM clusters using SGLang and TRTLLM backends.

## Quick Start

1. List the directory contents to understand what files are present.
2. Read files in priority order.
3. Correlate timestamps to identify the real failure point.
4. Distinguish root cause from noisy warnings.

## Priority Order

### 1. `sweep_{job_id}.log`

Read this first. It is the orchestration timeline.

Look for:
- stage transitions
- worker readiness
- benchmark start
- exit codes
- the last error before teardown

### 2. `benchmark.out`

If present, this usually contains the benchmark-side exception or timeout.

### 3. `artifacts/*/logs/aiperf_*.log`

If present, these often contain framework-level initialization failures and
HTTP/network issues.

### 4. Worker logs

Focus on errors that line up with the failure timestamp:
- `{node}_prefill_w{N}.out`
- `{node}_decode_w{N}.out`
- `{node}_frontend_{N}.out`

### 5. `infra.out`

Use this to confirm infrastructure failures involving NATS, etcd, ports, or
service health checks.

## Timestamp Correlation

This is the most important rule.

Many warnings are harmless. The root cause is usually the error that occurs at
the same time the orchestration log transitions into failure.

Use this method:
1. Find the failure time in `sweep_{job_id}.log`.
2. Search other logs for matching timestamps.
3. Ignore earlier warnings if the job continued past them.

## Common Signal

High-signal failures:
- `ReadTimeout`
- `Connection refused`
- `CUDA out of memory`
- `NCCL timeout`
- `Model not found`
- benchmark exit code failures

Low-signal noise:
- dependency resolver warnings
- cleanup warnings during teardown
- keep-alive failures after the main crash
- import warnings unrelated to the active model

## Output Format

Write markdown with this structure:

```markdown
## Job Analysis: {job_id}

### Root Cause
...

### Evidence
- `file:line or file`
- timestamp
- relevant error text

### Timeline
- key event -> timestamp

### Noise
- warnings that were not causal

### Recommended Fix
...
```

Keep the report concrete. Avoid generic summaries. If you are unsure, say so
and explain what evidence is missing.
