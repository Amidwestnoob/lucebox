#!/usr/bin/env bash
# Independent verification for the experimental Qwen3.8 DFlash2 native commit.
#
# Checks, in order:
#   1. The focused deterministic regression tests build and pass:
#      test_dflash2_contract, and test_ar_step_input_layout on CUDA and CPU.
#   2. The complete server unit suite passes.
#   3. Model-backed smoke on one CUDA GPU, three greedy chat requests:
#        a. target-only autoregressive (AR) baseline;
#        b. DFlash2 draft WITHOUT the opt-in env var: output must stay
#           byte-exact with AR and the exact AR fallback must run — this
#           proves the native commit is off by default;
#        c. DFlash2 draft WITH DFLASH_DFLASH2_NATIVE_COMMIT=1: byte-exact
#           SHA-256 output parity with AR, accepted_tokens > 0, at least
#           two distinct adaptive depths, no full-request AR replay after
#           speculation starts, and a clean server exit.
#
# Model paths are required and machine-neutral. Set:
#   QWEN38_TARGET_GGUF   target model GGUF (validated: Qwen3.8-27B Q4_K_L)
#   QWEN38_DRAFT_GGUF    DFlash2 draft GGUF (validated: Q4_K_M)
# Optional:
#   QWEN38_TARGET_SHA256 / QWEN38_DRAFT_SHA256   pin exact model files
#   QWEN38_CUDA_ARCH                             CUDA arch for a fresh build
#
# Exit 0 = PASS. Any other exit = FAIL with the reason on stderr.
set -u
cd "$(dirname "$0")"

# Optional developer-local model paths (untracked file).
[ -f .hermes/local-models.env ] && . .hermes/local-models.env

TARGET_GGUF="${QWEN38_TARGET_GGUF:-}"
DRAFT_GGUF="${QWEN38_DRAFT_GGUF:-}"

fail() { echo "VERIFY: FAIL: $*" >&2; exit 1; }
note() { echo "VERIFY: $*"; }

if [ -z "$TARGET_GGUF" ] || [ -z "$DRAFT_GGUF" ]; then
    echo "usage: set QWEN38_TARGET_GGUF and QWEN38_DRAFT_GGUF to the target" >&2
    echo "and DFlash2 draft GGUF paths, then rerun this script." >&2
    exit 2
fi
[ -f "$TARGET_GGUF" ] || fail "target model missing: $TARGET_GGUF"
[ -f "$DRAFT_GGUF" ] || fail "draft model missing: $DRAFT_GGUF"

if [ -n "${QWEN38_TARGET_SHA256:-}" ]; then
    note "hashing target model"
    tsha=$(sha256sum "$TARGET_GGUF" | cut -d' ' -f1)
    [ "$tsha" = "$QWEN38_TARGET_SHA256" ] || fail "target sha256 mismatch: $tsha"
fi
if [ -n "${QWEN38_DRAFT_SHA256:-}" ]; then
    note "hashing draft model"
    dsha=$(sha256sum "$DRAFT_GGUF" | cut -d' ' -f1)
    [ "$dsha" = "$QWEN38_DRAFT_SHA256" ] || fail "draft sha256 mismatch: $dsha"
fi

if [ ! -x build/dflash_server ] || [ ! -x build/test_dflash2_contract ] || \
   [ ! -x build/test_ar_step_input_layout ] || [ ! -x build/test_server_unit ]; then
    note "building server and tests"
    cmake -B build -S server -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${QWEN38_CUDA_ARCH:-native}" \
        > /dev/null 2>&1 || fail "cmake configure"
    cmake --build build --target dflash_server test_dflash2_contract \
        test_ar_step_input_layout test_server_unit -j"$(nproc)" \
        > /dev/null 2>&1 || fail "build"
fi

note "running focused deterministic regression tests"
./build/test_dflash2_contract || fail "test_dflash2_contract"
./build/test_ar_step_input_layout || fail "test_ar_step_input_layout (cuda+cpu)"
./build/test_ar_step_input_layout --cpu || fail "test_ar_step_input_layout --cpu"

note "running complete server unit suite"
./build/test_server_unit > /tmp/qwen38_verify_unit.log 2>&1 \
    || fail "test_server_unit (log: /tmp/qwen38_verify_unit.log)"
tail -4 /tmp/qwen38_verify_unit.log

note "model-backed smoke: AR baseline, default-off draft, opted-in native"
QWEN38_TARGET_GGUF="$TARGET_GGUF" QWEN38_DRAFT_GGUF="$DRAFT_GGUF" \
python3 - <<'PYEOF' || fail "model-backed smoke"
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.getcwd()
SERVER = os.path.join(ROOT, "build/dflash_server")
TARGET = os.environ["QWEN38_TARGET_GGUF"]
DRAFT = os.environ["QWEN38_DRAFT_GGUF"]
PROMPT = json.load(open(os.path.join(ROOT, "benchmarks/qwen38_dflash2_workloads.json")))[
    "workloads"]["prose"]["prompt"]
AR_MARKERS = ("mode=ar_exact_fallback", "[ar-decode]")


def run(tag, port, use_draft, env_extra):
    cmd = [SERVER, TARGET, "--target-device", "cuda:0", "--max-ctx", "512",
           "--default-max-tokens", "96", "--host", "127.0.0.1", "--port", str(port)]
    if use_draft:
        cmd += ["--draft", DRAFT, "--draft-device", "cuda:0"]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.update(env_extra)
    err_path = f"/tmp/qwen38_smoke_{tag}.stderr.log"
    with open(err_path, "w") as se:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL, stderr=se,
                                start_new_session=True, text=True)
        deadline = time.time() + 900
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                if urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            print(f"{tag}: server not ready", file=sys.stderr)
            return None
        payload = {"messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": 96, "temperature": 0.0, "top_p": 1.0,
                   "seed": 123, "stream": False}
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=900) as resp:
            obj = json.loads(resp.read().decode())
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
        text = obj["choices"][0]["message"]["content"]
        return {
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "usage": obj.get("usage", {}),
            "exit": proc.returncode,
            "stderr": open(err_path, errors="replace").read(),
        }


ar = run("ar", 18601, False, {})
assert ar is not None, "AR run failed"
default_off = run("default_off", 18602, True, {})
assert default_off is not None, "default-off run failed"
native = run("native", 18603, True, {
    "DFLASH_DFLASH2_NATIVE_COMMIT": "1",
    "DFLASH_SINGLE_CHAIN_CHECKPOINT_F32": "1",
    "DFLASH_FAST_ROLLBACK_THRESHOLD": "1",
})
assert native is not None, "native run failed"

problems = []
for tag, res in (("ar", ar), ("default_off", default_off), ("native", native)):
    if res["exit"] != 0:
        problems.append(f"{tag} server exit {res['exit']}")
if ar["sha256"] != default_off["sha256"]:
    problems.append("default-off parity FAILED: "
                    f"ar={ar['sha256']} default_off={default_off['sha256']}")
if ar["sha256"] != native["sha256"]:
    problems.append(f"native parity FAILED: ar={ar['sha256']} native={native['sha256']}")

# Default-off proof: without the opt-in the request must take the exact AR
# fallback (its [ar-decode] line appears in the request section).
off_req_log = default_off["stderr"].split("chat START", 1)[-1]
if not any(marker in off_req_log for marker in AR_MARKERS):
    problems.append("default-off run shows no AR fallback marker: "
                    "native commit appears active without the opt-in")

d = native["usage"].get("dflash2", {})
accepted = int(d.get("accepted_tokens", 0) or 0)
proposed = int(d.get("proposed_tokens", 0) or 0)
depths = sorted(set(d.get("observed_depths", []) or []))
if accepted <= 0:
    problems.append("accepted_tokens == 0")
if len(depths) < 2:
    problems.append(f"fewer than two distinct adaptive depths: {depths}")
# The native request section must not contain any AR replay marker.
req_log = native["stderr"].split("chat START", 1)[-1]
for marker in AR_MARKERS:
    if marker in req_log:
        problems.append(f"AR full-request replay marker present: {marker}")

print(json.dumps({
    "ar_sha256": ar["sha256"],
    "default_off_sha256": default_off["sha256"],
    "native_sha256": native["sha256"],
    "exact_parity": ar["sha256"] == native["sha256"] == default_off["sha256"],
    "default_off_ar_fallback": True,
    "accepted_tokens": accepted,
    "proposed_tokens": proposed,
    "observed_depths": depths,
    "ar_full_request_replay": False,
    "server_exits": [ar["exit"], default_off["exit"], native["exit"]],
}, indent=1))
if problems:
    for p in problems:
        print("SMOKE PROBLEM:", p, file=sys.stderr)
    sys.exit(1)
PYEOF

note "PASS: native commit is opt-in, exact, and accepted draft tokens observed"
exit 0
