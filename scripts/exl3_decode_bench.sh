#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# EXL3 native-decode speed bench for Qwen3.8-Flash-Next on GB10 (DGX Spark).
#
# Starts ONE `spark serve`, sends 6 fixed prompts one at a time (then, unless
# SKIP_CONC=1, 2 prompts concurrently), saves each JSON response and prints
# the server's per-request `Done:` line (decode tok/s, TTFT, MTP acceptance).
# The serve is killed at the end.
#
# NOTE: MTP speculative decode only runs on /v1/chat/completions. Requests to
# /v1/completions take the plain decode path and will not show the MTP gain.
#
# Build first (the target env is required, or the binary rejects the EXL3
# checkpoint):
#   source ~/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH
#   export LIBRARY_PATH=/path/to/nccl-stub     # single-GPU build
#   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
#     ATLAS_TARGET_QUANT=exl3 cargo build --release -p spark-server
#
# Usage:
#   MODEL=/path/to/Qwen3.8-Flash-Next-EXL3-3.05bpw OUT=/tmp/bench-a \
#     scripts/exl3_decode_bench.sh
# Env: MODEL (required), OUT (default /tmp/exl3-decode-bench), PORT (8893),
#      BIN (./target/release/spark), LD_LIBRARY_PATH as needed,
#      SKIP_CONC=1 to skip the 2-concurrent round.
# Kill switch for the MoE expert kernels: ATLAS_MOE_B2_V4=0.
set -euo pipefail
: "${MODEL:?set MODEL to the EXL3 checkpoint dir}"
OUT="${OUT:-/tmp/exl3-decode-bench}"
PORT="${PORT:-8893}"
BIN="${BIN:-./target/release/spark}"
mkdir -p "$OUT"

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "port $PORT already serving; stop it first" >&2; exit 1
fi

"$BIN" serve --model-from-path "$MODEL" --port "$PORT" \
  --exl3-native-decode --kv-cache-dtype bf16 --gpu-memory-utilization 0.92 \
  --max-seq-len 8192 --speculative --num-drafts 1 --mtp-gate force \
  > "$OUT/serve.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null || true' EXIT

for _ in $(seq 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "serve exited; see $OUT/serve.log" >&2; exit 1; }
  sleep 2
done

P=("Explain how a refrigerator works, step by step."
   "Summarize the causes of the First World War in five bullet points."
   "Write a Python function that checks whether a string is a palindrome, with tests."
   "Translate into French: The children are making supper while the dog sleeps by the stove."
   "What is 17 * 23? Show your work."
   "Describe the water cycle to a seven-year-old.")

req() { # req INDEX OUTFILE
  python3 -c 'import json,sys;print(json.dumps({"model":"x","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}))' "${P[$1]}" \
    | curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d @- > "$2"
}

for k in 0 1 2 3 4 5; do req $k "$OUT/solo-p$k.json"; done
if [ "${SKIP_CONC:-0}" != 1 ]; then
  req 0 "$OUT/conc-p0.json" & A=$!
  req 5 "$OUT/conc-p5.json" & B=$!
  wait $A $B
fi

echo "== server Done lines (solo p0..p5, then concurrent) =="
grep -a "Done:" "$OUT/serve.log" || true
echo "outputs: $OUT"
