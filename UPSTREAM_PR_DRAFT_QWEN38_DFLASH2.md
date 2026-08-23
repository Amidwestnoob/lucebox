# PR draft: Qwen3.8 DFlash2 drafts with opt-in native commit

Two commits, reviewable independently:

1. `fix(qwen35): upload AR step inputs only after the step-graph rebuild`
2. `feat(qwen35): experimental Qwen3.8 DFlash2 drafts with opt-in native commit`

plus a small third commit with the benchmark harness, the benchmark
summary, and this draft.

## Summary

Commit 1 is a narrow correctness fix in the autoregressive (AR) decode
loop that affects every Qwen3.5-backend generation. `do_ar_decode`
uploaded `inp_embed`/`positions` into the step graph before
`build_target_step` rebuilt it. A rebuild with a different token count
relocates the input tensors inside the shared allocator arena, so the
first decode forward after every prefill read stale bytes as M-RoPE
positions and permanently tilted the first generated K row in the KV
cache. Whole greedy trajectories shifted as a result. The fix moves the
uploads after the rebuild.

Commit 2 adds experimental support for published Qwen3.8 DFlash2 Q4
draft models through the existing `--draft` flag: DFlash2 GGUF
detection, the grouped-convolution draft decoder and selector graphs, an
adaptive proposal depth controller, batched target verification, and an
opt-in "native commit" that advances the target's recurrent/KV state
directly from verified rows instead of replaying the request.

## Correctness

The native commit is exact by construction inside its validated
envelope, not exact by replay:

- Verify rows are built to be bit-identical to the one-token AR step
  graph: per-row interleaved `set_rows` KV writes, per-row maskless
  flash attention over the AR graph's padded span, per-column MMVQ for
  quantized projections, and a verify width cap of 3 proposals + seed
  (`kDflash2NativeCommitMaxDepth`) so the MMVQ kernel keeps one launch
  geometry. Wider batches change the kernel's reduction geometry and
  drift in the low bits, which is why the cap is a correctness pin, not
  a tuning knob.
- Accepted prefixes commit from F32 verify checkpoints (fast rollback);
  exact restore+replay is the automatic fallback. `rollback_to`
  zero-truncates attention rows past the commit point so later padded
  spans read zeros exactly as AR would.
- Model-backed proof on the rebased branch: byte-exact SHA-256 output
  parity between target-only AR, the non-opted-in draft path, and the
  native commit; details and hashes in
  `UPSTREAM_BENCHMARK_QWEN38_DFLASH2.md` (Exact parity on every
  workload, Baseline A1 and Baseline A2 hashes identical to DFlash2).
- `DFLASH_DFLASH2_EXACT_MARGIN=<logits>` optionally re-derives
  near-tie decisions token-by-token through the exact AR graph as
  defense in depth on unvalidated hardware.

## Performance

Controlled A1/B/A2 measurement (fresh server per condition, one warmup,
three measured repetitions per workload, greedy, seed 123, context 512)
on one RTX 3090 with a Qwen3.8-27B Q4_K_L target and Q4_K_M DFlash2
draft:

| workload | Baseline A1 tok/s | Baseline A2 tok/s | DFlash2 tok/s | vs midpoint |
|---|---:|---:|---:|---:|
| prose | 36.50 | 36.43 | 37.63 | 1.03x |
| code | 36.40 | 36.30 | 58.27 | 1.60x |
| repeated_context | 36.60 | 36.53 | 59.10 | 1.62x |

DFlash2 mean decode throughput exceeds the A1/A2 midpoint on all three
workloads with byte-exact outputs. Prose gains least (draft acceptance
is content-limited); code and repeated context show the full benefit.
Reproduce with `benchmarks/run_qwen38_dflash2_benchmark.py` (model
paths via `--target-gguf/--draft-gguf` or `QWEN38_TARGET_GGUF` /
`QWEN38_DRAFT_GGUF`).

## Opt-in

The native commit is explicitly opt-in and off by default:
`DFLASH_DFLASH2_NATIVE_COMMIT=1`. Without the opt-in, a DFlash2 draft
runs a diagnostic probe mode — real proposal/verify probes for
telemetry, then an exact one-token AR commit — so output is identical
on any hardware and no speedup is claimed. Ordinary (non-DFlash2)
drafts and the no-draft AR path are unchanged. Fast rollback additionally
uses the pre-existing `DFLASH_SINGLE_CHAIN_CHECKPOINT_F32=1` and
`DFLASH_FAST_ROLLBACK_THRESHOLD=1` knobs; without them the native
commit falls back to exact restore+replay (slower, same output).

## Testing

- `test_ar_step_input_layout` (commit 1): pins the upload-after-rebuild
  protocol on both CPU and CUDA backends against the real allocator.
- `test_dflash2_contract` (commit 2, 8 cases): tensor alias resolution,
  adaptive contraction/expansion including the depth-floor regression
  (depth 0 was an absorbing state), the width-cap pin, disabled-path
  behavior, telemetry accounting, and exact prefix/replay helpers.
- Complete server unit suite: 401 cases pass.
- `VERIFY_UPSTREAM_QWEN38_DFLASH2.sh`: builds, runs the tests above,
  then a three-server model smoke (AR baseline, default-off draft,
  opted-in native) asserting byte parity, accepted tokens, at least two
  distinct adaptive depths, the AR fallback marker in the default-off
  run, no full-request AR replay in the opted-in run, and clean exits.

## Limitations

- Validated envelope only: one RTX 3090 (SM 8.6), Qwen3.8-27B Q4_K_L
  target, Q4_K_M DFlash2 draft, greedy decoding. Other GPUs, quant
  layouts, or sampled decoding are NOT claimed exact; the width-cap
  argument depends on the validated MMVQ launch-geometry behavior.
- Sampled decoding falls back to AR unless the pre-existing
  `DFLASH_SAMPLED_VERIFY` path is used, which is outside this PR's
  validation.
- The kvflash pooled path and finite `--fa-window` keep masked verify
  and restore+replay commit (AR-exact rows are disabled there by
  construction).
- Prose-like content with low draft acceptance sees small gains; the
  win comes from code-like and repeated content.
