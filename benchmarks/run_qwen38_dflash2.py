#!/usr/bin/env python3
"""Controlled Baseline A1 / DFlash2 / Baseline A2 benchmark for the
experimental Qwen3.8 DFlash2 native commit.

Conditions run in A1 -> DFlash2 -> A2 order, each against a fresh server
process on one CUDA device. The DFlash2 condition opts into the native
recurrent commit (DFLASH_DFLASH2_NATIVE_COMMIT=1); the baselines are
target-only autoregressive decode. Every run records commands, raw
responses, output hashes, dflash2 telemetry, VRAM samples, and server exit
codes. The assembled result asserts byte-exact SHA-256 output parity across
all three conditions and requires the DFlash2 mean decode throughput to
exceed the A1/A2 midpoint on every workload.

Model paths are required and must be supplied explicitly:

    python3 benchmarks/run_qwen38_dflash2.py \
        --server-bin build/dflash_server \
        --target /path/to/target.gguf --draft /path/to/dflash2-draft.gguf \
        --gpu 0 --output-dir bench-out/qwen38-dflash2

Conditions can run in separate invocations (--only <condition>, then
--assemble) so an interrupted session never loses completed evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "benchmarks/qwen38_dflash2_workloads.json"
REPS = 3
WARMUPS = 1
WORKLOAD_NAMES = ["prose", "code", "repeated_context"]
AR_FALLBACK_MARKERS = ("mode=ar_exact_fallback", "[ar-decode]")

CONDITIONS: dict[str, dict[str, Any]] = {
    "baseline_a1": {"draft": False, "env": {}},
    "dflash2_q4_native": {
        "draft": True,
        "env": {
            "DFLASH_DFLASH2_NATIVE_COMMIT": "1",
            "DFLASH_DFLASH2_ADAPTIVE": "1",
            # Exact F32 recurrent checkpoints let every accepted prefix
            # commit from the verify forward (fast rollback) instead of a
            # second replay forward; threshold 1 applies it at any width.
            "DFLASH_SINGLE_CHAIN_CHECKPOINT_F32": "1",
            "DFLASH_FAST_ROLLBACK_THRESHOLD": "1",
        },
    },
    "baseline_a2": {"draft": False, "env": {}},
}


class Config:
    """Resolved benchmark inputs and output locations."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.server_bin = Path(args.server_bin)
        self.target = Path(args.target)
        self.draft = Path(args.draft)
        self.gpu = str(args.gpu)
        self.expect_target_sha = args.expect_target_sha256 or ""
        self.expect_draft_sha = args.expect_draft_sha256 or ""
        self.output_dir = Path(args.output_dir)
        self.out_json = self.output_dir / "benchmark_qwen38_dflash2.json"
        self.out_raw = self.output_dir / "benchmark_qwen38_dflash2_raw.log"
        self.port_base = args.port_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def gpu_info() -> dict[str, Any]:
    return run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )


def gpu_used_mib() -> int | None:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        return None
    values = []
    for line in p.stdout.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            pass
    return max(values) if values else None


class VramSampler:
    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = gpu_used_mib()
            if value is not None:
                self.samples.append((time.time(), value))
            self.stop_event.wait(0.10)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def peak(self, start: float, end: float) -> int | None:
        values = [v for t, v in self.samples if start <= t <= end]
        return max(values) if values else gpu_used_mib()


def http_json(port: int, payload: dict[str, Any]) -> tuple[int, str, dict[str, Any] | None, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = None
            return response.status, raw, obj, ""
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), None, str(error)
    except Exception as error:
        return 599, "", None, repr(error)


def wait_ready(proc: subprocess.Popen[str], port: int) -> tuple[bool, str]:
    deadline = time.time() + 1800
    detail = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}: {detail}"
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/health", method="GET"
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status == 200:
                    return True, "ready"
        except Exception as error:
            detail = repr(error)
        time.sleep(1)
    return False, f"health timeout: {detail}"


def request_payload(workload: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": workload["prompt"]}],
        "max_tokens": workload["n_predict"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 123,
        "stream": False,
    }


def response_content(obj: dict[str, Any] | None) -> str:
    if not obj:
        return ""
    choices = obj.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", ""))


def stderr_ar_fallback_delta(stderr_path: Path, offset: int) -> tuple[bool, int]:
    """Scan server stderr text written after `offset` for AR-fallback markers.

    Returns (marker_seen, new_offset).
    """
    try:
        data = stderr_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, offset
    chunk = data[offset:]
    seen = any(marker in chunk for marker in AR_FALLBACK_MARKERS)
    return seen, len(data)


def make_run(
    condition: str,
    workload: str,
    rep: int,
    status: int,
    raw: str,
    obj: dict[str, Any] | None,
    error: str,
    peak: int | None,
    elapsed: float,
    run_kind: str,
    native_commit_env: bool,
    ar_marker_seen: bool,
) -> dict[str, Any]:
    usage = obj.get("usage", {}) if obj else {}
    timings = usage.get("timings", {}) if isinstance(usage, dict) else {}
    dflash2 = usage.get("dflash2", {}) if isinstance(usage, dict) else {}
    text = response_content(obj)
    return {
        "run_kind": run_kind,
        "condition": condition,
        "workload": workload,
        "repetition": rep,
        "exit_code": 0 if status == 200 and obj is not None else int(status or 1),
        "http_status": status,
        "decode_tokens_per_second": timings.get("decode_tokens_per_sec", 0.0),
        "peak_vram_mib": peak,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_text": text,
        "proposed_tokens": int(dflash2.get("proposed_tokens", 0) or 0),
        "accepted_tokens": int(dflash2.get("accepted_tokens", 0) or 0),
        "observed_depths": list(dflash2.get("observed_depths", []) or []),
        "accept_rate": usage.get("accept_rate", 0.0) if isinstance(usage, dict) else 0.0,
        "spec_decode_ran": bool(usage.get("spec_decode_ran", False))
        if isinstance(usage, dict)
        else False,
        "native_commit": bool(native_commit_env and not ar_marker_seen),
        "ar_full_request_replay": bool(ar_marker_seen),
        "elapsed_wall_seconds": elapsed,
        "response_json": obj,
        "response_raw": raw,
        "error": error,
    }


def server_command(cfg: Config, draft: bool, port: int) -> list[str]:
    command = [
        str(cfg.server_bin),
        str(cfg.target),
        "--target-device",
        "cuda:0",
        "--max-ctx",
        "512",
        "--default-max-tokens",
        "256",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if draft:
        command += ["--draft", str(cfg.draft), "--draft-device", "cuda:0"]
    return command


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_condition(cfg: Config, condition: str, spec: dict[str, Any],
                  sampler: VramSampler, workloads: dict[str, Any], index: int,
                  raw: list[str]) -> dict[str, Any]:
    """Run one condition's warmup + measured requests against a fresh server.

    Appends raw-evidence lines to `raw` and returns the condition record.
    """
    port = cfg.port_base + index
    command = server_command(cfg, spec["draft"], port)
    env_overrides = {"CUDA_VISIBLE_DEVICES": cfg.gpu, **spec["env"]}
    env = os.environ.copy()
    env.update(env_overrides)
    native_commit_env = env.get("DFLASH_DFLASH2_NATIVE_COMMIT") == "1"
    stdout_path = cfg.output_dir / f"{condition}.stdout.log"
    stderr_path = cfg.output_dir / f"{condition}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        stdout.write("COMMAND: " + json.dumps(command) + "\n")
        stdout.write("ENV_OVERRIDES: " + json.dumps(env_overrides) + "\n")
        stdout.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=True,
        )
        ready, detail = wait_ready(proc, port)
        raw.append(
            f"\n===== CONDITION {condition} =====\n"
            f"COMMAND: {json.dumps(command)}\n"
            f"ENV_OVERRIDES: {json.dumps(env_overrides)}\n"
            f"READY: {ready} {detail}\n"
        )
        condition_result: dict[str, Any] = {
            "command": command,
            "environment_overrides": env_overrides,
            "server_stdout": str(stdout_path),
            "server_stderr": str(stderr_path),
            "warmups": {},
        }
        for name in WORKLOAD_NAMES:
            condition_result[name] = []
        stderr_offset = 0
        for workload_name in WORKLOAD_NAMES:
            workload = workloads[workload_name]
            payload = request_payload(workload)
            for rep, kind in [
                (0, "warmup"),
                *[(n, "measured") for n in range(1, REPS + 1)],
            ]:
                started = time.time()
                status, response_raw, obj, error = (
                    http_json(port, payload) if ready else (598, "", None, detail)
                )
                elapsed = time.time() - started
                ar_seen, stderr_offset = stderr_ar_fallback_delta(
                    stderr_path, stderr_offset
                )
                run = make_run(
                    condition,
                    workload_name,
                    rep,
                    status,
                    response_raw,
                    obj,
                    error,
                    sampler.peak(started, time.time()),
                    elapsed,
                    kind,
                    native_commit_env,
                    ar_seen,
                )
                raw.append(
                    f"\n-- {kind.upper()} {condition}/{workload_name}/rep{rep} --\n"
                )
                raw.append(
                    "REQUEST: " + json.dumps(payload, ensure_ascii=False) + "\n"
                )
                raw.append(
                    "RUN: "
                    + json.dumps(run, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                if kind == "warmup":
                    condition_result["warmups"][workload_name] = run
                else:
                    condition_result[workload_name].append(run)
        stop_process(proc)
    condition_result["server_exit_code"] = proc.returncode
    raw.append(f"server_exit_code={proc.returncode}\n")
    raw.append(
        "SERVER_STDOUT:\n"
        + stdout_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
    )
    raw.append(
        "SERVER_STDERR:\n"
        + stderr_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
    )
    return condition_result


def preflight(cfg: Config) -> tuple[str, str, dict[str, Any]]:
    required = [cfg.server_bin, cfg.target, cfg.draft, WORKLOADS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required inputs: " + ", ".join(missing))
    target_sha = sha256_file(cfg.target)
    draft_sha = sha256_file(cfg.draft)
    if cfg.expect_target_sha and target_sha != cfg.expect_target_sha:
        raise SystemExit(
            f"target model hash mismatch: {target_sha} != {cfg.expect_target_sha}")
    if cfg.expect_draft_sha and draft_sha != cfg.expect_draft_sha:
        raise SystemExit(
            f"draft model hash mismatch: {draft_sha} != {cfg.expect_draft_sha}")
    workloads = json.loads(WORKLOADS.read_text(encoding="utf-8"))["workloads"]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return target_sha, draft_sha, workloads


def result_skeleton(cfg: Config, target_sha: str, draft_sha: str,
                    workloads: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    gpu = gpu_info()
    return {
        "schema_version": 2,
        "source": {"repository": "lucebox", "commit": source_commit},
        "hardware": gpu,
        "models": {
            "target": {"path": str(cfg.target), "sha256": target_sha},
            "dflash2_q4": {"path": str(cfg.draft), "sha256": draft_sha},
        },
        "settings": {
            "warmups": WARMUPS,
            "repetitions": REPS,
            "greedy": True,
            "seed": 123,
            "cuda_visible_devices": cfg.gpu,
            "context": 512,
            "endpoint": "/v1/chat/completions",
            "native_commit_env": "DFLASH_DFLASH2_NATIVE_COMMIT=1",
        },
        "workloads": workloads,
        "conditions": {},
        "correctness": {},
        "adaptive_depth": {},
        "summary": {},
    }


def run_one(cfg: Config, condition: str) -> int:
    """Run one condition and persist its record + raw fragment."""
    if condition not in CONDITIONS:
        print(f"unknown condition {condition}", file=sys.stderr)
        return 2
    target_sha, draft_sha, workloads = preflight(cfg)
    index = list(CONDITIONS).index(condition)
    raw: list[str] = []
    sampler = VramSampler()
    sampler.start()
    try:
        record = run_condition(cfg, condition, CONDITIONS[condition], sampler,
                               workloads, index, raw)
    finally:
        sampler.stop()
    record["sampler_peak_vram_mib"] = max(
        (value for _, value in sampler.samples), default=None
    )
    (cfg.output_dir / f"{condition}.runs.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")
    (cfg.output_dir / f"{condition}.raw.txt").write_text(
        "".join(raw), encoding="utf-8")
    ok = record["server_exit_code"] == 0 and all(
        run["exit_code"] == 0
        for name in WORKLOAD_NAMES
        for run in record[name]
    )
    print(json.dumps({
        "condition": condition,
        "server_exit_code": record["server_exit_code"],
        "all_requests_ok": ok,
        "peak_vram_mib": record["sampler_peak_vram_mib"],
    }))
    return 0 if ok else 1


def assemble(cfg: Config) -> int:
    target_sha, draft_sha, workloads = preflight(cfg)
    result = result_skeleton(cfg, target_sha, draft_sha, workloads)
    raw: list[str] = [
        "QWEN38 DFLASH2 BENCHMARK RAW EVIDENCE\n",
        f"source_commit={result['source']['commit']}\n",
        f"target={cfg.target}\ntarget_sha256={target_sha}\n",
        f"dflash2_q4={cfg.draft}\ndflash2_q4_sha256={draft_sha}\n",
        f"gpu_info={json.dumps(result['hardware'], sort_keys=True)}\n",
        f"settings={json.dumps(result['settings'], sort_keys=True)}\n",
    ]
    peaks: list[int] = []
    for condition in CONDITIONS:
        runs_path = cfg.output_dir / f"{condition}.runs.json"
        raw_path = cfg.output_dir / f"{condition}.raw.txt"
        if not runs_path.exists() or not raw_path.exists():
            print(f"missing condition evidence: {condition}", file=sys.stderr)
            return 2
        record = json.loads(runs_path.read_text(encoding="utf-8"))
        peak = record.pop("sampler_peak_vram_mib", None)
        if isinstance(peak, int):
            peaks.append(peak)
        result["conditions"][condition] = record
        raw.append(raw_path.read_text(encoding="utf-8"))
    result["hardware"]["sampler_peak_vram_mib"] = max(peaks, default=None)
    all_parity = True
    all_accel = True
    result["summary"]["workloads"] = {}
    for workload_name in WORKLOAD_NAMES:
        a1 = result["conditions"]["baseline_a1"][workload_name]
        a2 = result["conditions"]["baseline_a2"][workload_name]
        b = result["conditions"]["dflash2_q4_native"][workload_name]
        a1_hashes = [run["output_sha256"] for run in a1]
        a2_hashes = [run["output_sha256"] for run in a2]
        b_hashes = [run["output_sha256"] for run in b]
        parity = (
            len(b) == REPS
            and len(set(a1_hashes)) == 1
            and a1_hashes == a2_hashes == b_hashes
        )
        result["correctness"][workload_name] = {
            "exact_parity": parity,
            "baseline_a1_output_sha256": a1_hashes,
            "baseline_a2_output_sha256": a2_hashes,
            "dflash2_output_sha256": b_hashes,
        }
        depths = [depth for run in b for depth in run["observed_depths"]]
        result["adaptive_depth"][workload_name] = {
            "observed_depths": depths[:256],
            "distinct_depths": sorted(set(depths)),
            "min_observed_depth": min(depths) if depths else 0,
            "max_observed_depth": max(depths) if depths else 0,
            "proposed_tokens": sum(run["proposed_tokens"] for run in b),
            "accepted_tokens": sum(run["accepted_tokens"] for run in b),
        }
        a1_mean = statistics.mean(
            [float(run["decode_tokens_per_second"] or 0.0) for run in a1]
        )
        a2_mean = statistics.mean(
            [float(run["decode_tokens_per_second"] or 0.0) for run in a2]
        )
        b_mean = statistics.mean(
            [float(run["decode_tokens_per_second"] or 0.0) for run in b]
        )
        midpoint = (a1_mean + a2_mean) / 2.0
        accelerated = b_mean > midpoint
        result["summary"]["workloads"][workload_name] = {
            "baseline_a1_mean_tps": a1_mean,
            "baseline_a2_mean_tps": a2_mean,
            "baseline_midpoint_mean_tps": midpoint,
            "dflash2_mean_tps": b_mean,
            "speedup_vs_midpoint": (b_mean / midpoint) if midpoint > 0 else 0.0,
            "accelerated": accelerated,
        }
        all_parity = all_parity and parity
        all_accel = all_accel and accelerated

    b_runs_ok = all(
        run["exit_code"] == 0
        and run["proposed_tokens"] > 0
        and run["accepted_tokens"] > 0
        and run["native_commit"] is True
        and run["ar_full_request_replay"] is False
        for name in WORKLOAD_NAMES
        for run in result["conditions"]["dflash2_q4_native"][name]
    )
    baseline_runs_ok = all(
        run["exit_code"] == 0
        for cond in ("baseline_a1", "baseline_a2")
        for name in WORKLOAD_NAMES
        for run in result["conditions"][cond][name]
    )
    result["summary"]["all_exact_parity"] = all_parity
    result["summary"]["all_accelerated"] = all_accel
    result["summary"]["dflash2_runs_native_ok"] = b_runs_ok
    result["summary"]["baseline_runs_ok"] = baseline_runs_ok
    result["summary"]["classification"] = (
        "exact_parity_native_commit_accelerated"
        if (all_parity and all_accel and b_runs_ok and baseline_runs_ok)
        else "failed_or_divergent"
    )

    cfg.out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg.out_raw.write_text("".join(raw).rstrip("\n") + "\n", encoding="utf-8")
    ok = all_parity and all_accel and b_runs_ok and baseline_runs_ok
    print(
        json.dumps(
            {
                "result": str(cfg.out_json),
                "raw": str(cfg.out_raw),
                "all_exact_parity": all_parity,
                "all_accelerated": all_accel,
                "dflash2_runs_native_ok": b_runs_ok,
                "peak_vram_mib": result["hardware"]["sampler_peak_vram_mib"],
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A1/DFlash2/A2 controlled benchmark for the Qwen3.8 "
                    "DFlash2 native commit (exact-parity gate included)")
    parser.add_argument("--server-bin", default=str(ROOT / "build/dflash_server"),
                        help="dflash_server binary to benchmark")
    parser.add_argument("--target", required=True,
                        help="target model GGUF path (required)")
    parser.add_argument("--draft", required=True,
                        help="DFlash2 draft model GGUF path (required)")
    parser.add_argument("--gpu", default="0",
                        help="CUDA_VISIBLE_DEVICES value for the server (default 0)")
    parser.add_argument("--output-dir", default=str(ROOT / "bench-out/qwen38-dflash2"),
                        help="directory for per-condition records, server logs, "
                             "and assembled results")
    parser.add_argument("--expect-target-sha256", default="",
                        help="optional expected SHA-256 of the target model")
    parser.add_argument("--expect-draft-sha256", default="",
                        help="optional expected SHA-256 of the draft model")
    parser.add_argument("--port-base", type=int, default=18260)
    parser.add_argument("--only", metavar="CONDITION",
                        help="run a single condition: " + ", ".join(CONDITIONS))
    parser.add_argument("--assemble", action="store_true",
                        help="assemble previously recorded conditions")
    args = parser.parse_args()
    cfg = Config(args)
    if args.only:
        return run_one(cfg, args.only)
    if args.assemble:
        return assemble(cfg)
    for condition in CONDITIONS:
        code = run_one(cfg, condition)
        if code != 0:
            return code
    return assemble(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
