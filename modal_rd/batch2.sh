#!/usr/bin/env bash
# Batch 2: routing sweep for the all-sparse vs arm at ~127K (same 6 CCDB prompts).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p modal_rd/logs
arm() {  # tag json
  modal run --detach modal_rd/app.py --module modal_rd.eval_niah \
    --args "--arm vs --attn '$2' --bytes 512000 --n 6 --tag $1" > "modal_rd/logs/$1.log" 2>&1 &
  sleep 3
}
arm b2_v4k_w4k        '{"sink":64,"window":4096,"vertical":4096}'
arm b2_v8k_w8k        '{"sink":64,"window":8192,"vertical":8192}'
arm b2_v2k_w4k_max    '{"sink":64,"window":4096,"vertical":2048,"route_pool":"max"}'
arm b2_v4k_w4k_max    '{"sink":64,"window":4096,"vertical":4096,"route_pool":"max"}'
arm b2_v2k_w4k_r256   '{"sink":64,"window":4096,"vertical":2048,"n_route":256}'
arm b2_v4k_w4k_max_r16 '{"sink":64,"window":4096,"vertical":4096,"route_pool":"max","n_route":16}'
wait
