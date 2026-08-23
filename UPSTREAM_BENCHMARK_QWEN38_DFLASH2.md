# Qwen3.8 DFlash2 native commit — controlled benchmark summary

Measured on the rebased branch at `54c3525` with
`benchmarks/run_qwen38_dflash2_benchmark.py`. Conditions ran in
Baseline A1 -> DFlash2 -> Baseline A2 order, each against a fresh server
process on one NVIDIA GeForce RTX 3090 (24 GiB, driver 580.173.02,
physical GPU 0). One warmup plus three measured repetitions per workload
per condition; greedy decoding (temperature 0, top-p 1, seed 123),
context 512, 192 generated tokens per request.

Models: Qwen3.8-27B Q4_K_L target
(SHA-256 `6d9ee93a089dff34a901a8f18613bb4f3b124856410057460f401863dd6ed2f0`)
and Q4_K_M DFlash2 draft
(SHA-256 `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`).
The DFlash2 condition uses the experimental opt-in
(`DFLASH_DFLASH2_NATIVE_COMMIT=1`, plus the pre-existing
`DFLASH_SINGLE_CHAIN_CHECKPOINT_F32=1` and
`DFLASH_FAST_ROLLBACK_THRESHOLD=1` fast-rollback knobs); the baselines
are target-only autoregressive decode.

## Throughput (mean decode tok/s over 3 repetitions)

| workload | Baseline A1 | Baseline A2 | midpoint | DFlash2 | vs midpoint |
|---|---:|---:|---:|---:|---:|
| prose | 36.50 | 36.43 | 36.47 | 37.63 | 1.03x |
| code | 36.40 | 36.30 | 36.35 | 58.27 | 1.60x |
| repeated_context | 36.60 | 36.53 | 36.57 | 59.10 | 1.62x |

DFlash2 mean throughput exceeds the Baseline A1 / Baseline A2 midpoint
on all three workloads. Per-repetition standard deviation is at most
0.12 tok/s. Peak sampled VRAM: 22441 MiB for the DFlash2 condition vs
20698 MiB for the baselines.

## Correctness

Exact parity: every repetition of every condition produced the
identical output SHA-256 per workload — Baseline A1, DFlash2, and
Baseline A2 hashes match byte-for-byte (prose `59dab591ca362036…`, code
`464050b7e1cff0fc…`, repeated_context `ead3fff042944cf2…`).

## Speculation telemetry (3 measured DFlash2 repetitions per workload)

| workload | proposed | accepted | distinct adaptive depths |
|---|---:|---:|---|
| prose | 612 | 303 | 1, 2, 3 |
| code | 456 | 426 | 2, 3 |
| repeated_context | 204 | 198 | 2, 3 |

Every DFlash2 repetition ran with the native commit active: accepted
tokens > 0, no full-request AR replay marker in the request log, and
clean server exits (exit code 0) for all three conditions.

## Validated envelope

These results and the exactness argument cover exactly this
configuration: one RTX 3090 (SM 8.6), Qwen3.8-27B Q4_K_L target,
Q4_K_M DFlash2 draft, greedy decoding, full attention. Other GPUs,
quantizations, or sampled decoding are not claimed. The three-server
smoke in `VERIFY_UPSTREAM_QWEN38_DFLASH2.sh` additionally proves the
default-off behavior: without the opt-in, the same draft produces the
same bytes through the diagnostic AR-fallback path.
