#!/usr/bin/env bash
# Batch 3: exact-tail queries on top of the vs arm at ~127K (same 6 CCDB prompts).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
arm() {  # tag json [bytes]
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes ${3:-512000} --n 6 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
arm b3_v2k_w4k_t64   '{"sink":64,"window":4096,"vertical":2048,"exact_tail":64}'
arm b3_v2k_w4k_t256  '{"sink":64,"window":4096,"vertical":2048,"exact_tail":256}'
arm b3_v4k_w4k_t64   '{"sink":64,"window":4096,"vertical":4096,"exact_tail":64}'
arm b3_v8k_w8k_t64   '{"sink":64,"window":8192,"vertical":8192,"exact_tail":64}'
arm b3_v1k_w2k_t64   '{"sink":64,"window":2048,"vertical":1024,"exact_tail":64}'
arm b3_v0_w4k_t64    '{"sink":64,"window":4096,"vertical":64,"exact_tail":64}'
wait
