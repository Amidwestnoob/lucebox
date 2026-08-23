# Qwen3.8 Q4 DFlash2 in Lucebox

Lucebox loads the published Qwen3.8 DFlash2 Q4_K_M draft through the existing
`--draft` path. The target remains the native Qwen3.5 hybrid backend; no
external llama.cpp or buun process is required at runtime.

## Launch

```text
build/dflash_server \
  /path/to/Qwen3.8-27B-Q4_K_L.gguf \
  --target-device cuda:0 \
  --draft /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --draft-device cuda:0 \
  --host 127.0.0.1 --port 8080
```

Omitting `--draft` keeps the existing autoregressive path. The DFlash2 draft
is selected from GGUF metadata (`general.architecture=dflash`), not from a
machine-specific filename.

## Compatibility

The loader accepts the published flattened names and the older nested names:

- `selector_hidden.weight` / `selector.hidden_proj.weight`
- `selector_predecessor.weight` / `selector.pred_codebook`
- `selector_successor.weight` / `selector.succ_codebook`
- `blk.N.attn_conv_base` / `blk.N.attn_conv.base`
- `blk.N.attn_conv_proj.weight` / `blk.N.attn_conv.proj.weight`
- corresponding `ffn_conv` names

It reads `dflash.target_layers`, `dflash.block_size`, the grouped-convolution
geometry, and selector geometry from the draft. Those target-layer IDs replace
the legacy evenly spaced capture IDs before the target cache is allocated.

## Adaptive depth and correctness

DFlash2 drafts the trained eight-token block, but Lucebox keeps the target
output exact. The controller starts at depth 2, contracts toward a floor of
one proposal when the observed draft prefix acceptance is at most 35% (depth
zero would never observe acceptance again and could not recover), and expands
toward the trained maximum when acceptance is at least 80%; the deterministic
controller is covered by host tests.

The native commit is experimental and off by default; opt in with
`DFLASH_DFLASH2_NATIVE_COMMIT=1`. The historical divergence between the
native replay and the target-only AR graph (first fault at position 69:
replay 55404 vs AR 421) was root-caused to the AR decode loop itself: it
uploaded `inp_embed`/`positions` before rebuilding the step graph, so the
first decode forward after every prefill read stale bytes as M-RoPE
positions and permanently tilted the first generated K row. With the upload
order fixed, the native path is exact and the AR baseline changes to the
truly correct trajectory.

The native commit verifies through AR-exact batched rows: full-attention
layers interleave one per-row `set_rows` KV write with one per-row maskless
flash-attention call over the same 256-padded span the AR decode graph uses,
quantized projections force the per-column MMVQ path, and the verify width is
capped at `kDflash2NativeCommitMaxDepth` (3 proposals + seed) so every
projection keeps the single-geometry MMVQ launch class. Under those
constraints each verify row's logits and state advance are bit-identical to
sequential AR decode; greedy acceptance, the correction token, and the
committed recurrent state (fast rollback from F32 checkpoints, with exact
restore+replay as the fallback) all match target-only AR exactly.
`DFLASH_DFLASH2_EXACT_MARGIN=<logits>` optionally re-derives near-tie steps
token by token through the exact AR graph as defense in depth on unvalidated
hardware. Without the `DFLASH_DFLASH2_NATIVE_COMMIT=1` opt-in, a DFlash2
draft runs the diagnostic probe mode (verify probes plus a fresh-prefill AR
commit, identical output, no speedup claim). The bit-exactness argument for
the native commit is validated on an RTX 3090 (SM 8.6) with a Q4_K_L target
and Q4_K_M draft under greedy decoding; other GPUs, quantizations, or
sampled decoding are outside the validated envelope.

Set `DFLASH_DFLASH2_ADAPTIVE=0` to disable only adaptive depth and retain the
full native draft depth. This does not change target verification or the
no-draft autoregressive behavior.

The non-streaming API reports bounded telemetry under `usage.dflash2`:

```json
{
  "proposed_tokens": 42,
  "accepted_tokens": 31,
  "observed_depths": [2, 3, 4, 3]
}
```

`observed_depths` is capped at 256 entries per request. The server emits
`[dflash2] proposed=N accepted=N depth=N` lines for raw benchmark evidence;
with the native commit opted in there is no `[ar-decode]` line for a DFlash2
request (its presence marks the default diagnostic mode or an unrelated AR
tail-off).

## Deterministic verification

The focused host contract tests cover alias resolution, adaptive bounds and
expansion/contraction (including the depth floor regression), the native
commit width cap, disabled-path behavior, telemetry accounting, and exact
prefix/replay semantics. `test_ar_step_input_layout` pins the
upload-after-rebuild input protocol behind the AR positions fix. The
model-backed acceptance package is produced by:

```text
python3 benchmarks/run_qwen38_dflash2_benchmark.py
bash VERIFY_UPSTREAM_QWEN38_DFLASH2.sh
```

The benchmark uses one RTX 3090, one warmup, three measured repetitions,
fixed greedy settings, fresh server processes for Baseline A1, adaptive Q4
DFlash2, and Baseline A2, and exact output SHA-256 comparisons for prose, code,
and repeated-context workloads.
