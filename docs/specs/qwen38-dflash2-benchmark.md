# Qwen3.8 DFlash2 native commit — measured benchmark

Companion to `qwen38-dflash2.md` (design and correctness argument): this
page records the controlled measurement for the validated configuration.
Produced with `benchmarks/run_qwen38_dflash2.py` from a clean test build;
conditions ran in Baseline A1 -> DFlash2 -> Baseline A2 order, each
against a fresh server process on one NVIDIA GeForce RTX 3090 (24 GiB,
driver 580.173.02). One warmup plus three measured repetitions per
workload per condition; greedy decoding (temperature 0, top-p 1, seed
123), context 512, 192 generated tokens per request. Target:
Qwen3.8-27B Q4_K_L; draft: Q4_K_M DFlash2. The DFlash2 condition uses
the experimental opt-in (`DFLASH_DFLASH2_NATIVE_COMMIT=1` with the
pre-existing `DFLASH_SINGLE_CHAIN_CHECKPOINT_F32=1` and
`DFLASH_FAST_ROLLBACK_THRESHOLD=1` fast-rollback knobs); the baselines
are target-only autoregressive decode.

## Throughput (mean decode tok/s over 3 repetitions)

| workload | Baseline A1 | Baseline A2 | midpoint | DFlash2 | vs midpoint |
|---|---:|---:|---:|---:|---:|
| prose | 36.50 | 36.40 | 36.45 | 37.43 | 1.03x |
| code | 36.33 | 36.30 | 36.32 | 58.10 | 1.60x |
| repeated_context | 36.60 | 36.50 | 36.55 | 58.90 | 1.61x |

DFlash2 mean decode throughput exceeds the Baseline A1 / Baseline A2
midpoint on every workload. Peak sampled VRAM: 22440 MiB for the DFlash2
condition vs 20698 MiB for the baselines.

## Exact parity

Exact parity holds: every repetition of every condition produced the
identical output SHA-256 per workload — Baseline A1, DFlash2, and
Baseline A2 match byte-for-byte (prose `59dab591ca362036…`, code
`464050b7e1cff0fc…`, repeated_context `ead3fff042944cf2…`).

## Speculation telemetry (3 measured DFlash2 repetitions per workload)

| workload | proposed | accepted | distinct adaptive depths |
|---|---:|---:|---|
| prose | 612 | 303 | 1, 2, 3 |
| code | 456 | 426 | 2, 3 |
| repeated_context | 204 | 198 | 2, 3 |

Every DFlash2 repetition ran with the native commit active: accepted
tokens > 0, no full-request AR replay marker in the request log, and
clean server exits for all three conditions. Prose accelerates least
(draft acceptance is content-limited); code and repeated context show
the full native-commit benefit.

## Scope

These numbers cover exactly the validated envelope stated in
`qwen38-dflash2.md`: one RTX 3090 (SM 8.6), Q4_K_L target, Q4_K_M
DFlash2 draft, greedy decoding, full attention, dense KV cache. Other
GPUs, quantizations, sampled decoding, or KVFlash-paged runs are not
claimed by this measurement; under paging the native commit is refused
and the exact restore+replay path serves speculation.
